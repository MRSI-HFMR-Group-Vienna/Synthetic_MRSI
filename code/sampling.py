from numba.np.arrayobj import array_shape
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from Synthetic_MRSI.code.OLD.main_notebook__BACKUP2 import coil_sensitivity_maps

from Synthetic_MRSI.code.main_notebook___BACKUP2 import target_size
from file import Configurator
from printer import Console
import dask.array as da
from tqdm import tqdm
import numpy as np
import cupy as cp
import pint
import dask
import file
import sys
from tools import InterpolationTools, DaskTools, GPUTools, FourierTools
from interfaces import Interpolation
import os
from pathlib import Path
from sampling import CoilSensitivityVolume



class Model:

    def __init__(self,
                 block_size: tuple[int, int, int, int, int] = None,
                 target_gpu_smaller_tasks: int = None,
                 path_cache: str = None,
                 data_type: np.dtype = None,
                 coil_sensitivity_volume: CoilSensitivityVolume = None,
                 spectral_spatial_volume: da.Array = None):


        self.path_cache = None
        if path_cache is not None:
            if os.path.exists(path_cache):
                self.path_cache = path_cache
            else:
                Console.printf("error", f"Terminating the program. Path does not exist: {path_cache}")
                sys.exit()
            dask.config.set(temporary_directory=path_cache)


        self.block_size = block_size # (coil, time, X, Y, Z)
        self.data_type = data_type

        # Need the coil sensitivity maps not as dask array for performance reasons
        if not isinstance(coil_sensitivity_volume.volume, CoilSensitivityVolume):
            Console.printf("error", f"coil sensitivity volume must be of class sampling.CoilSensitivityVolume, but got: {type(coil_sensitivity_volume.volume)}")
            sys.exit()
        # Need the spectral spatial model as dask array. Automatically converts it to dask array if not already one.
        if not isinstance(spectral_spatial_volume, da.Array):
            chunksize = self.block_size[1:]
            Console.printf("warning", f"spectral spatial volume is of type {type(spectral_spatial_volume)} and will be converted to dask array with blocksize {chunksize}")
            spectral_spatial_volume = DaskTools.to_dask(self.spectral_spatial_volume, chunksize=chunksize) # chunksize=(time, X, Y, Z)

        # Assigning to the two main volume to object variables
        self.coil_sensitivity_volume: CoilSensitivityVolume = coil_sensitivity_volume
        self.spectral_spatial_volume_dask: da.Array = spectral_spatial_volume # volume/graph from previous Spectral Spatial Model

        # This volume is for intermediate steps (e.g., coil combination, FFT, iFFT, cropping, ... and so on and will be dynamically updated!)
        self.working_volume = None # The volume which is already (coil, time, X, Y, Z), nd where also other functions may be already applied
        Console.printf("info", "Please note that the 'working_volume' of the Sampling Model will be updated with each operation!\n")

        # Target GPU index for smaller tasks if GPU is used
        self.target_gpu_smaller_tasks = target_gpu_smaller_tasks

        self.model_summary()



    def model_summary(self):
        Console.add_lines("Sampling Model:")
        Console.add_lines(f" => Got spectral spatial volume:")
        Console.add_lines(f"    class type: .......... {type(self.spectral_spatial_volume_dask)}")
        Console.add_lines(f"    data type: ........... {self.spectral_spatial_volume_dask.dtype}")
        Console.add_lines(f"    shape: ............... {self.spectral_spatial_volume_dask.shape}\n")

        Console.add_lines(f" => Got coil sensitivity volume:")
        Console.add_lines(f"    class type: .......... {type(self.coil_sensitivity_volume)}")
        Console.add_lines(f"    volume data type: .... {self.coil_sensitivity_volume.volume.dtype}")
        Console.add_lines(f"    volume shape: ........ {self.coil_sensitivity_volume.volume.shape}\n")

        Console.add_lines(f" => Desired overall block size for dask: ... {self.block_size}")
        Console.add_lines(f" => Cache path  ............................ {Path(*Path(self.path_cache).parts[-2:])} (shortened)")
        Console.add_lines(f" => Target GPU smaller tasks: .............. {self.target_gpu_smaller_tasks}")
        Console.printf_collected_lines("info")

        ###if not isinstance(self.coil_sensitivity_volume, da.Array):
        ###    chunksize = (self.block_size[0], *self.block_size[2:])
        ###    self.coil_sensitivity_volume = DaskTools.to_dask(self.coil_sensitivity_volume, chunksize=chunksize) # chunksize=(coils,X,Y,Z)
        ###    Console.add_lines(f"Transformed coil sensitivity volume of type '{type(coil_sensitivity_volume)}' => dask array with chunksize: {chunksize}") # TODO: maybe print shape too
        ###if not isinstance(self.spectral_spatial_volume, da.Array):
        ###    chunksize = self.block_size[1:]
        ###    self.spectral_spatial_volume = DaskTools.to_dask(self.spectral_spatial_volume, chunksize=chunksize) # chunksize=(time, X, Y, Z)
        ###    Console.add_lines(f"Transformed spectral spatial volume of type '{type(coil_sensitivity_volume)}' => dask array with chunksize: {chunksize}") # TODO: maybe print shape too
        ###
        ###Console.printf_collected_lines("info")


    def apply_coil_sensitivity(self, compute_on_device: str = 'gpu', return_on_device: str = 'cpu') -> da.Array:
        """
        To apply the coil sensitivity maps (= B1- maps) to the volume in this sampling model.

        Basically, it is just:
            result = volume[1, time, X, Y, Z] * coil_sensitivity_volume[coils, 1, X, Y, Z]

        :param compute_on_device: either 'cpu' or 'gpu'/'cuda'
        :param return_on_device: either 'cpu' or 'gpu'/'cuda'
        :return: dask array with cupy or numpy inside
        """
        # convert coil sensitivity volume to dask array
        chunksize = (self.block_size[0], *self.block_size[2:])
        coil_sensitivity_volume_dask = DaskTools.to_dask(self.coil_sensitivity_volume.volume, chunksize=chunksize) # chunksize=(coils,X,Y,Z)

        # first bring to respective device!
        coil_sensitivity_volume, spectral_spatial_volume = GPUTools.dask_map_blocks_many(
            coil_sensitivity_volume_dask, self.spectral_spatial_volume_dask,
            device=compute_on_device)

        # Broadcasting to be able to multiply at once:
        #    coil sensitivity maps shape   : e.g., (coil, X, Y, Z) --> (coil, 1,    X, Y, Z)
        #    spectral spatial volume shape : e.g., (time, X, Y, Z) --> (coil, time, X, Y, Z)
        volume = (coil_sensitivity_volume[:, None, ...]  *     # (coils, 1,    X, Y, Z)
        spectral_spatial_volume[None, ...])
        # (1,     time, X, Y, Z)

        self.working_volume = GPUTools.dask_map_blocks(volume, device=return_on_device)
        Console.printf("success", "Added to apply coil sensitivity maps computational graph.")

        return self.working_volume
    
    
    def apply_cartesian_FT(self, direction: str = "forward", fft_shift: bool = True, compute_on_device: str ='gpu', return_on_device: str ='cpu') -> da.Array:
        """
        This method serves as forward and backward fourier transformation and also includes the fft-shift and ifftshift. This method is only for cartesian fourier
        transformation and should not be extended to a non-cartesian fourier transformation.

        :param direction: "forward" and "inverse" is allowed
        :param fft_shift: if True and "forward" is selected then it performs fftshift and if True and "inverse" ifftshift is performed
        :param compute_on_device: either 'cpu' or 'gpu'/'cuda'
        :param return_on_device: either 'cpu' or 'gpu'/'cuda'
        :return: dask array
        """

        self.working_volume = FourierTools.cartesian_FT_dask(array=self.working_volume, direction=direction, axes=(2,3,4), fft_shift=fft_shift, device=compute_on_device)
        self.working_volume = GPUTools.dask_map_blocks(self.working_volume, device=return_on_device)
        return self.working_volume


    def crop_k_space_center(self, crop_center_shape=(64, 64, 40), verbose:bool = True) -> da.Array:
        """
        This requires an array of input shape (coil, time, X, Y, Z), and the axes X, Y, Z of this array should be also in the k-space.
        Further, also the low frequencies should be shifted to the center (e.g., fftshift) after the fourier transformation.

        Then, the cropping will create from (coil, time, X1, Y1, Z1) ==> (coil, time, X2, Y2, Z2)

        :param crop_center_shape: the spatial shape of the volume which should be cropped (X, Y, Z)
        :param verbose: if True then print to console
        :return: dask Array
        """

        array = self.working_volume
        array_shape_before = array.shape

        # TODO: Check that shape is really (coil, time, X, Y, Z)
        if len(array.shape) != 5:
            Console.printf("error", f"Cannot crop k-space center in this case. Require shape (coil, time, X, Y, Z). Given was: {array.shape}")
            return

        cx, cy, cz = crop_center_shape
        X, Y, Z = array.shape[-3:]

        if None in (X, Y, Z):
            Console.printf("error", "Center crop needs known X/Y/Z sizes (shape can't contain None).")
            return

        if cx > X or cy > Y or cz > Z:
            Console.printf("error", f"Crop {crop_center_shape} larger than spatial shape {(X, Y, Z)}")
            return

        sx = (X - cx) // 2
        sy = (Y - cy) // 2
        sz = (Z - cz) // 2

        array = array[..., sx:sx + cx, sy:sy + cy, sz:sz + cz]
        array_shape_after = array.shape

        Console.printf("success", f"Cropping k-space center (..., X,Y,Z) from {array_shape_before} => {array_shape_after}", mute=not verbose)

        self.working_volume = array

        return array

            # maybe here gpu tools? numpy or cupy

        #self.working_volume

###    def apply_cartesian_FFT(self, compute_on_device: str = 'gpu', return_on_device: str = 'cpu', apply_fftshift:bool = True, crop_center_k_space_xyz: tuple = None, rechunking: bool = False) -> da.Array | None:
###        """
###        Applies a FFT along X,Y,Z of volume (coil, time, X,Y,Z).
###
###        :return: dask.array
###        """
###        if self.working_volume is None:
###            Console.printf("error", "Apply first the function 'apply_coil_sensitivity', then re-run this function.")
###            return None
###        else:
###            # Bring to the desired device to compute
###            i_space_volume = GPUTools.dask_map_blocks(self.working_volume, device=compute_on_device)
###            # Re-chunk for performance reasons
###            chunks_before = i_space_volume.chunksize
###            if rechunking:
###                i_space_volume = i_space_volume.rechunk(("auto", "auto", -1,-1,-1)) # to ensure that coil and time is chunked, but X;Y;Z is a single chunk
###                Console.printf("success", f"Re-chunked for better performance from {chunks_before} => {i_space_volume.chunksize}")
###
###            # Perform FFT on desired axes
###            k_space_volume = FourierTools.dask_fftn(
###                array=i_space_volume,
###                axes=(2,3,4),
###                compute_on_device=compute_on_device,
###                return_on_device=compute_on_device, # still use compute on device, since further operations might be possible afterward on desired device
###                fftshift=apply_fftshift,
###                verbose=True)
###
###            # Cropping of k-space center if desired
###            if crop_center_k_space_xyz is not None:
###                if len(crop_center_k_space_xyz) != 3:
###                    Console.printf("error", f"Required tuple of length 3 for k-space center cropping. But was given: {len(crop_center_k_space_xyz)}")
###                else:
###                    k_space_volume_shape_before = k_space_volume.shape
###                    k_space_volume = self.crop_center_volume(array=k_space_volume, output_shape_xyz=crop_center_k_space_xyz)
###                    Console.printf("success", f"Cropping k-space (X,Y,Z) in (coil, time, X, Y, Z): {k_space_volume_shape_before} => {k_space_volume.shape}")
###
###            # Bring FFT-volume to desired target device (output)
###            k_space_volume = GPUTools.dask_map_blocks(k_space_volume, device=return_on_device)
###            #Console.printf("success", f"Added FFT to the graph.")
###
###            return k_space_volume

###    def crop_center_volume(self, array: da.Array, output_shape_xyz):
###        """
###        To crop a given volume of shape (coil, time, X, Y, Z) in a way that only (X,Y,Z) is cropped. This is useful to crop e.g.,
###        fourier transformed volume. Note: It only crops the center!
###
###        :param output_shape_xyz: (X, Y, Z), thus only the spatial dimensions
###        :return: Nothing
###        """
###
###        X, Y, Z = array.shape[2], array.shape[3], array.shape[4]    # the actual shape of the volume
###        Xc, Yc, Zc = output_shape_xyz                   # the desired cropping shape volume
###        if Xc > X or Yc > Y or Zc > Z:
###            Console.printf("error",f"Crop size must be <= original size on each axis. Original size xyz: {self.working_volume[2:]}; Desired shape xyz: {output_shape_xyz}")
###
###        sx = (X - Xc) // 2
###        ex = sx + Xc
###        sy = (Y - Yc) // 2
###        ey = sy + Yc
###        sz = (Z - Zc) // 2
###        ez = sz + Zc
###
###        return array[:, :, sx:ex, sy:ey, sz:ez]

    def coil_combination(self, compute_on_device: str, return_on_device: str):

        # (1) Interpolate to target size (of other volume). No task yet required!
        working_volume = self.working_volume.shape
        coil_sensitivity_volume_shape_before = self.coil_sensitivity_volume.volume.shape
        #     The following: compute_on_device, return_on_device => both compute_on_device input argument since not lst operation in this method!
        coil_sensitivity_volume = self.coil_sensitivity_volume.interpolate_volume(compute_on_device=compute_on_device, return_on_device=compute_on_device, target_size=working_volume.shape[2:], target_gpu=self.target_gpu_smaller_tasks, verbose=False)
        Console.printf("success", f"Interpolated coil sensitivity volume: {coil_sensitivity_volume_shape_before} => {coil_sensitivity_volume.volume.shape}")
        # (2) Need complex conjugated coil sensitivity volume
        coil_sensitivity_volume_conjugated = coil_sensitivity_volume.conjugate()
        # (3) Create dask arrays (+ good junks regarding performance)
        coil_sensitivity_volume_dask = da.asarray(coil_sensitivity_volume.volume, chunks=(self.working_volume.chunksize[0], -1, -1, -1)) # TODO, not already right chunksize? Dont need -1,-1,-1???
        coil_sensitivity_volume_conjugated_dask = da.asarray(coil_sensitivity_volume_conjugated.volume, chunks=(self.working_volume.chunksize[0], -1, -1, -1)) # TODO: same aso one line before?!
        # (4) Bring to right device via map blocks (thus if GPU does not allocate in advance)
        coil_sensitivity_volume_dask, coil_sensitivity_volume_conjugated_dask = GPUTools.dask_map_blocks_many(coil_sensitivity_volume_dask, coil_sensitivity_volume_conjugated_dask, device=compute_on_device)

        # (5) Need to broadcast conjugated array (from (coil, X, Y, Z) --> (coil, time, X, Y, Z))
        coil_sensitivity_volume_conjugated_dask = coil_sensitivity_volume_conjugated_dask[:,None, ...]

        # (6) Now, this (MRSI volume * coil_sensitivity_complex_conj) / |sum(coil_sensitivity, axis=0)|^2. Note: The working volume is here MRSI volume.
        numerator = da.sum(working_volume * coil_sensitivity_volume_conjugated_dask, axis=0)


        # TODO: This line is an issue!
        #coil_sensitivity_volume.volume
        #denominator = (cp.sum(cp.abs(GPUTools.to_device(coil_sensitivity_volume2.volume, device="gpu"))**2, axis=0))

        pass


class ModelOLD:
    # TODO docstring

    def __init__(self,
                 computational_graph_spectral_spatial_model: da.Array = None,
                 block_size_computational_graph_spectral_spatial_model: tuple = None,
                 coil_sensitivity_maps: np.ndarray | cp.ndarray | da.Array = None,
                 block_size_coil_sensitivity_maps: tuple = None,
                 path_cache: str = None,
                 data_type: str = "complex64", persist_computational_graph_spectral_spatial_model=False):


        # only dask array is allowed as input? -> yes at moment!
        # re-chunk only if desired chunks do not match

        self.coil_sensitivity_maps = coil_sensitivity_maps
        if isinstance(coil_sensitivity_maps, da.Array):
            self.coil_sensitivity_maps = coil_sensitivity_maps.rechunk(chunks=block_size_coil_sensitivity_maps)
        elif coil_sensitivity_maps is None:
            Console.printf("warning", "coil_sensitivity_maps is None in sampling Model")
        else:
            Console.printf("info", f"Sampling Model: Converting coil_sensitivity_maps {type(coil_sensitivity_maps)} --> dask Array")
            self.coil_sensitivity_maps = da.asarray(coil_sensitivity_maps, chunks=block_size_coil_sensitivity_maps)

        self.computational_graph_spectral_spatial_model = computational_graph_spectral_spatial_model

        if self.computational_graph_spectral_spatial_model is not None:
            self.computational_graph_spectral_spatial_model.rechunk(chunks=block_size_computational_graph_spectral_spatial_model)

        if self.computational_graph_spectral_spatial_model is not None:
            self.computational_graph_spectral_spatial_model = self.computational_graph_spectral_spatial_model.persist() \
                if persist_computational_graph_spectral_spatial_model \
                else self.computational_graph_spectral_spatial_model

        self.path_cache = path_cache
        self.data_type = data_type

        dask.config.set(temporary_directory=path_cache)

        # self.compute_on_device = compute_on_device
        # self.return_on_device = return_on_device

        # self.volumes_with_coil_sensitivity_maps: list[da.Array] = []
        # self.volumes_with_coil_sensitivity_maps_k_space: list[da.Array] = []

    ###    def _select_devices(self, computation_device: str = 'cpu', return_device: str = 'cpu'):
    ###        possible_devices = {'cpu', 'cuda'}
    ###        if computation_device not in possible_devices or return_device not in possible_devices:
    ###            Console.printf("error", f"Selected devices are {possible_devices, return_device}. However, only {possible_devices} is possible.")
    ###            sys.exit()
    ###
    ###        xp_compute = cp if computation_device == 'cuda' else np
    ###        xp_return = cp if return_device == 'cuda' else np
    ###
    ###        return xp_compute, xp_return


    def _get_xp(self, to_device):
        """
        To choose the appropriate device where something should be computed or returned.
        :param to_device: choose either "cpu" or "cuda" to select the respective device. For cpu numpy is used and for
        cuda cupy is used.
        :return: numpy object (np) or cupy object (cp) as xp
        """
        xp = None

        if to_device == 'cpu':
            xp = np
        elif to_device == 'cuda':
            xp = cp
        else:
            Console.printf("error", f"Only possible to choose 'cpu' or 'cuda', but you set: {to_device}")

        return xp

    def _to_device(self, array: da.Array, device: str = 'cpu'):
        """
        TODO: Bring to tools -> Because general usage possible.
        TODO: Then also use at other places? (E.g. spatial_spectral_simulation.Model.assemble_graph (end of code))

        To bring a dask array to the desired device. 'cuda' and 'cpu' is supported.

        :param array: dask.Array
        :param device: "cpu" or "cuda"
        :return: dask.Array with numpy or cupy backend
        """
        possible_devices = {'cpu', 'cuda'}
        if device not in possible_devices:
            Console.printf("error", f"Selected device is {device}. However, only {possible_devices} is possible.")
            sys.exit()

        if not isinstance(array, da.Array):
            Console.printf("error", f"Can only convert dask array backend 'cpu' <-> 'cuda'. But you gave: {type(array)}")

        if isinstance(array._meta, np.ndarray) and device == 'cpu':
            return array  # do nothing
        elif isinstance(array._meta, cp.ndarray) and device == 'cuda':
            return array  # do nothing
        elif isinstance(array._meta, np.ndarray) and device == 'cuda':
            #return array.map_blocks(cp.asarray, dtype=array.dtype, meta=cp.empty((0,), dtype=array.dtype),)
            print(">>>>>>> Transform numpy array to cupy array! <<<<<<<")
            return array.map_blocks(cp.asarray, dtype='c8', meta=cp.empty((0,), dtype=array.dtype), )
        elif isinstance(array._meta, cp.ndarray) and device == 'cpu':
            #return array.map_blocks(cp.asnumpy, dtype=array.dtype, meta=np.empty((0,), dtype=array.dtype),)
            print(">>>>>>> Transform cupy array to numpy array! <<<<<<<")
            return array.map_blocks(cp.asnumpy, dtype='c8', meta=np.empty((0,), dtype=array.dtype), )
        else:
            Console.printf("error", f"In Sampling: Some error occurred. Cannot do transfer CPU <-> CUDA of type {type(array._meta)}")
            sys.exit()


    def apply_coil_sensitivity_maps(self,
                                compute_on_device: str = 'cpu',
                                return_on_device: str = 'cpu') -> da.Array:
        """
        For applying the coil sensitivity maps to the volume of the spectral spatial model.
        :param compute_on_device: target device (cpu or cuda)
        :param return_on_device: target device (cpu or cuda)
        :return: list of dask arrays
        """

        # 1) Bring both the sampling volume and the coil sensitivity maps volumes to same device
        self.coil_sensitivity_maps = self._to_device(self.coil_sensitivity_maps, device=compute_on_device)
        self.computational_graph_spectral_spatial_model = self._to_device(self.computational_graph_spectral_spatial_model, device=compute_on_device)

        ########################## NOT USED YET ########################################################################
        ####### For only computing values that don#t yield 0
        ####### => But is slower for 1536 FID points => Maybe test with 100_000 FID Points
        ### if compute_on_device == 'cpu':
        ###     xp = np
        ### elif compute_on_device == 'cuda':
        ###     xp = cp
        ### else:
        ###     Console.printf("error", f"Only possible to compute on 'cpu' or on 'cuda', but you set: {compute_on_device}")
        ### computational_graph_spectral_spatial_model_broadcast = da.broadcast_to(
        ###     self.computational_graph_spectral_spatial_model,
        ###     (self.coil_sensitivity_maps[:, None, ...].shape[0],) + self.computational_graph_spectral_spatial_model[None, ...].shape[1:]  # => (coils, time, X, Y, Z)
        ### )
        ###
        ### volume = da.where(
        ###     self.computational_graph_spectral_spatial_model[None, ...] != 0,
        ###     self.computational_graph_spectral_spatial_model[None, ...]      # (1,     time, X, Y, Z)
        ###     * self.coil_sensitivity_maps[:, None, ...],                     # (coils, 1,    X, Y, Z)
        ###     computational_graph_spectral_spatial_model_broadcast
        ###     )
        ################################################################################################################

        # 2) Broadcasting to be able to multiply at once:
        #    coil sensitivity maps shape   : e.g., (32,   X, Y, Z) --> (32, 1,    X, Y, Z)
        #    spectral spatial volume shape : e.g., (1536, X, Y, Z) --> (32, 1536, X, Y, Z)
        volume = (
            self.coil_sensitivity_maps[:, None, ...]                      # (coils, 1,    X, Y, Z)
            * self.computational_graph_spectral_spatial_model[None, ...]  # (1,     time, X, Y, Z)
        )
        # 3) Move the full (coils,time,X,Y,Z) stacked volume back once to target device
        volume = self._to_device(volume, device=return_on_device)
        return volume

    def cartesian_FFT(self,
                      volumes_with_coil_sensitivity_maps: list[da.Array],
                      crop_center_shape: tuple = None,
                      compute_on_device: str = 'cpu',
                      return_on_device: str = 'cpu') -> list[da.Array]:
        """
        For bringing the volume into the cartesian k-space via fast fourier transformation (FFT). Usually used after
        apply the coil sensitivity maps, thus this method deals with a list of dask arrays.

        :param volumes_with_coil_sensitivity_maps: the volume with applied coil sensitivity maps.
        :param crop_center_shape: tuple with shape values for a symmetrical cut around the center of the volume.
        :param compute_on_device: target device (cpu or cuda)
        :param return_on_device: target device (cpu or cuda)
        :return: list of volumes (one for each coil) in cartesian k-space
        """

        if compute_on_device == 'cpu':
            xp = np
            Console.printf("info", f"Cartesian FFT: compute on device '{compute_on_device}' and return on '{return_on_device}'")
        elif compute_on_device == 'cuda':
            xp = cp
        else:
            Console.printf("error", f"Only possible to compute on 'cpu' or on 'cuda', but you set: {compute_on_device}")

        volumes_cartesian_k_space: list[da.Array] = []

        Console.add_lines("Performed FFT for:")
        for i, one_coil_image_domain in enumerate(volumes_with_coil_sensitivity_maps):
            Console.add_lines(f" > Coil {i+1}")
            
            one_coil_image_domain = self._to_device(one_coil_image_domain, device=compute_on_device)

            one_coil_k_space = one_coil_image_domain.map_blocks(xp.fft.fftn, dtype=xp.complex64, axes=(1, 2, 3))
            one_coil_k_space_shifted = one_coil_k_space.map_blocks(xp.fft.fftshift, dtype=xp.complex64, axes=(1, 2, 3))
            one_coil_k_space = one_coil_k_space_shifted

            if crop_center_shape is not None:
                original_shape = one_coil_k_space_shifted.shape
                desired_shape = (original_shape[0], crop_center_shape[0], crop_center_shape[1], crop_center_shape[2])
                start_indices = [(orig - des) // 2 for orig, des in zip(original_shape, desired_shape)]

                end_indices = [start + des for start, des in zip(start_indices, desired_shape)]
                one_coil_k_space_cropped = one_coil_k_space_shifted[
                                           :,
                                           start_indices[1]:end_indices[1],
                                           start_indices[2]:end_indices[2],
                                           start_indices[3]:end_indices[3]]
                one_coil_k_space = one_coil_k_space_cropped

            volumes_cartesian_k_space.append(self._to_device(one_coil_k_space, device=return_on_device))

        Console.printf_collected_lines("success")

        return volumes_cartesian_k_space


    def cartesian_IFFT(self,
                       volumes_cartesian_k_space: list[da.Array],
                       compute_on_device: str = 'cpu',
                       return_on_device: str = 'cpu') -> list[da.Array]:

        if compute_on_device == 'cpu':
            xp = np
            Console.printf("info", f"Cartesian IFFT: compute on device '{compute_on_device}' and return on '{return_on_device}'")
        elif compute_on_device == 'cuda':
            xp = cp
        else:
            Console.printf("error", f"Only possible to compute on 'cpu' or on 'cuda', but you set: {compute_on_device}")

        volumes_image_domain: list[da.Array] = []
        for one_coil_k_space in volumes_cartesian_k_space:
            one_coil_k_space = self._to_device(one_coil_k_space, device=compute_on_device)

            one_coil_image_domain = one_coil_k_space.map_blocks(xp.fft.ifftshift, dtype=xp.complex64, axes=(1, 2, 3))
            one_coil_image_domain = one_coil_image_domain.map_blocks(xp.fft.ifftn, dtype=xp.complex64, axes=(1, 2, 3))
            volumes_image_domain.append(self._to_device(one_coil_image_domain, device=return_on_device))
        return volumes_image_domain

    def apply_gaussian_noise(self,
                             volumes_with_coil_sensitivity_maps: list[da.Array],
                             snr_desired: float,
                             compute_on_device: str = 'cpu',
                             return_on_device: str = 'cpu') -> list[da.Array]:
        # TODO: Need the peak of the signal -> thus compute max value and then std -> muss nicht hier vorher das Ganze in den fourier domain bringen (den FID?)
        # Von jedem Peak individuell? Müsste dann alle FID FFT? (also auch rechunk...)

        volumes_image_domain_noisy: list[da.Array] = []

        for one_coil_volume in volumes_with_coil_sensitivity_maps:
            one_coil_volume = self._to_device(one_coil_volume.rechunk(chunks=(one_coil_volume.shape[0],
                                                                              int(one_coil_volume.shape[1] / 20),
                                                                              int(one_coil_volume.shape[2] / 20),
                                                                              int(one_coil_volume.shape[3] / 20))), device=compute_on_device)

            #### Get the spectra of all FID signals
            ###spectra = da.map_blocks(cp.fft.fft, one_coil_volume, axis=0)
            #### Get the mean FID signa
            ###mean_spectrum = spectra.mean(axis=(1, 2, 3))
            ###mean_spectrum = mean_spectrum.rechunk(chunks=one_coil_volume.shape[0])
            ####mean_spectrum = da.map_blocks(cp.asnumpy, mean_spectrum).compute()
            #### Get max peak in spectrum
            ###max_peak = da.map_blocks(cp.amax, mean_spectrum) #.compute()
            max_peak = cp.array(53.308907 - 60.410015j)
            # Get the desired standard deviation based on the desired snr

            noise_std_desired = max_peak / (2 * snr_desired)
            # Generate noise
            # print(one_coil_volume.shape[0])
            # print(noise_std_desired.compute())
            # input("=================================")
            # noise = cp.random.normal(0+0j, noise_std_desired.compute(), size=one_coil_volume.shape[0])

            noise_real = cp.random.normal(0, cp.abs(noise_std_desired), size=one_coil_volume.shape[0], dtype=cp.float32)
            noise_imag = cp.random.normal(0, cp.abs(noise_std_desired), size=one_coil_volume.shape[0], dtype=cp.float32)
            noise = noise_real + 1j * noise_imag
            noise = noise.astype(cp.complex64)
            noise = noise.reshape(1536, 1, 1, 1)

            # noise = da.asarray(noise, chunks=one_coil_volume.shape[0])
            # noise = da.random.normal(0, noise_std_desired, size=mean_spectrum.shape, chunks=mean_spectrum.compute_chunk_sizes())
            # noise = noise.map_blocks(cp.asarray) #.compute()
            # Add to each spectrum (each voxel)
            one_coil_volume_noisy = one_coil_volume + noise
            one_coil_volume_noisy = one_coil_volume_noisy.rechunk(chunks=one_coil_volume.shape)

            volumes_image_domain_noisy.append(one_coil_volume_noisy)

        return volumes_image_domain_noisy



###    def coil_combination(self,
###                         volumes_with_coil_sensitivity: list[da.Array],
###                         compute_on_device: str = 'cpu',
###                         return_on_device: str = 'cpu') -> da.Array | np.ndarray | cp.ndarray:
###        """
###        Gets a volume of shape (coils, time, X, Y, Z) and combines it to ==> (time, X, Y, Z) via summing up along coils axis.
###
###        :param volumes_with_coil_sensitivity: list of dask arrays with (time, X, Y, Z), thus for each coil
###        :param compute_on_device: "cuda" or "cpu"
###        :param return_on_device: "cuda" or "cpu"
###        :return:
###        """
###        xp = self._get_xp(compute_on_device)
###
###        volume_with_coil_sensitivity = self._to_device(volumes_with_coil_sensitivity, device=compute_on_device)#.sum(axis=0)
###        xp.conjugate(volume_with_coil_sensitivity)*
###
###
###        volume_with_coil_sensitivity = self._to_device(volume_with_coil_sensitivity, device=return_on_device)
###        return volume_with_coil_sensitivity


class CoilSensitivityVolume(Interpolation):

    def __init__(self):
        self.coils: list[str] = None
        self.volume: np.ndarray | np.memmap = None

    def interpolate(self, target_size: tuple, order: int = 3, compute_on_device: str = "gpu", return_on_device: str="cpu", target_gpu: int= 0, verbose= True):
        """
        To interpolate the coil sensitivity maps to a target shape. Note, typically a 3D shape is provided and the coil axis is still preserved during interpolation.
        Thus, target shape (X,Y,Z) dos not include coil dimension; coil sensitivity data shape is typically (coil, X, Y, Z).

        :param target_size: tuple (X,Y,Z)
        :param order: see scipy documentation for interpolation order
        :param compute_on_device: either "cpu" or "gpu"/"cuda"
        :param return_on_device: either "cpu" or "gpu"/"cuda"
        :param target_gpu: if gpu is used for interpolation the index of the device
        :param verbose: If True then print infos about results
        :return: Nothing
        """
        if len(target_size) == 3:
            target_size = (self.volume.shape[0], *target_size)
            Console.printf("info", "Preserving coil dimension during interpolation.")
            #Console.printf("info", f"3D target shape is given. However, preserving coil axis during interpolation, therefore will still get 4D result: {target_size}.")

        self.volume = InterpolationTools.interpolate(array=self.volume,
                                                     target_size=target_size,
                                                     order=order,
                                                     compute_on_device=compute_on_device,
                                                     return_on_device=return_on_device,
                                                     target_gpu=target_gpu,
                                                     verbose=verbose)

    def conjugate(self):
        """
        The complex conjugate of a complex number is obtained by changing the sign of its imaginary part.

        :return: Nothing
        """
        self.volume = np.conjugate(self.volume)
        Console.printf("success", "Complex Conjugated the Coil Sensitivity Volume.")


class Trajectory:
    """
    To calculate and prepare the trajectories for respective parameters, e.g., stored in json files.
    This includes at the moment the cartesian trajectory and the concentric ring trajectory.
    """

    def __init__(self, configurator: file.Configurator, size_x: int, size_y: int):
        """
        To set parameters used by the methods. Since the cartesian and concentric rings trajectory are 'paired' in size,
        one object - programmatically speaking - with shared size makes absolute sense.

        :param configurator: object that holds the information of the paths to the necessary files.
        :param size_x: size of trajectory in x direction (historically called frequency encoding direction)
        :param size_y: size of trajectory in y direction (historically called phase encoding direction)
        """

        self.configurator = configurator
        self.size_x = size_x    # for cartesian and concentric rings trajectory necessary.
        self.size_y = size_y    # for only cartesian trajectory necessary.

        # For introducing units with pint. For just having one pint registry using the class variable from file.Trajectory.
        # Therefore, able to use one registry for different instances of the file.Trajectory.
        self.u = file.Trajectory.get_pint_unit_registry()

    def _get_delta_gradient_moment_x(self) -> pint.Quantity:
        """
        Just a helper function.

        \nFormular:\n ΔG_x = 1/(FOV_x⋅γ/2π). The bigger the field of view (FOV) the smaller the gradient moment, which yields a lower spatial resolution.

        :return: A pint Quantity with a numpy array inside. Thus, the delta gradient moment (ΔGM) in x direction with unit.
        """

        # To be able to use units
        u = self.u

        # Load the required parameters form a defined json file
        parameters = file.Trajectory(configurator=self.configurator).load_cartesian()

        # Get relevant parameters from loaded parameters
        larmor_frequency = parameters["SpectrometerFrequency"]
        field_strength = parameters["MagneticFieldStrength"]
        matrix_size = parameters["MatrixSize"]
        voxel_size = parameters["AcquisitionVoxelSize"]

        # Calculate relevant parameters
        field_strength = field_strength.to(u.mT)  # convert from [T] to [mT] to fit final units
        gyromagnetic_ratio_over_two_pi = larmor_frequency / field_strength
        field_of_view = matrix_size * voxel_size
        field_of_view = field_of_view.to(u.meter)  # convert from [mm] to [m]
        field_of_view_x = field_of_view[0]
        delta_gradient_moment_x = 1 / (field_of_view_x * gyromagnetic_ratio_over_two_pi)

        return delta_gradient_moment_x

    def _get_maximum_radius_x(self):
        """
        Just a helper function.

        Defines the maximum trajectory radius. Therefore, it uses the ΔGM (delta gradient moment) computed from the
        parameters for the cartesian trajectory.

        \nFormular:\n maximum radius in x = kx size / 2, where kx is the 'frequency encoding direction. The x-direction is just historically named in that way.

        :return: A pint Quantity with a numpy array inside. Thus, the maximum radius in x direction with respective units.
        """

        maximum_radius_x = self._get_delta_gradient_moment_x() * self.size_x / 2  # maximum trajectory radius

        return maximum_radius_x

    def get_cartesian_2D(self) -> np.ndarray:
        """
        To compute the encoding field for the 2D cartesian trajectory, thus use the parameters from the respective json file. This parameters will be loaded and used by helper functions that will be used in this method.

        \nThe functionalities of the mentioned helper functions:
        * the delta gradient moment in x direction,
        * the maximum radius in x of the trajectory in k-space.

        :return: Numpy array that represents the 2D cartesian trajectory encoding field.
        """

        # Create the 2D encoding field:
        #  (a) with just indices [0, ... size] in the respective dimension.
        #     e.g., x-dimension:         e.g., y-dimension:
        #      [0, 1, 2, ... x]           [0, 0, 0, ... 0]
        #      [0, 1, 2, ... x]           [1, 1, 1, ... 1]
        #      [0, 1, 2, ... x]           [2, 2, 2, ... 2]
        #      [0, 1, 2, ... x]           [y, y, y, ... y]
        size_x, size_y = self.size_x, self.size_y
        encoding_field = np.indices((size_x, size_y)).astype(dtype=np.float64)

        # (b) now, bring the zeros in the center of the field and thus also check if
        #     dimension shape is odd or even with modulo operator. (Shorthand If Else btw.)
        x_field, y_field = encoding_field[0, :, :], encoding_field[1, :, :]
        encoding_field[0, :, :] = (x_field - np.floor(size_x / 2) + 0.5) if (size_x / 2 % 2 == 0) else (x_field - np.floor(size_x / 2))
        encoding_field[1, :, :] = y_field - np.floor(size_y / 2) + 0.5 if size_y / 2 % 2 == 0 else y_field - np.floor(size_y / 2)

        # (c) create delta gradient moment field finally & normalize trajectory:
        encoding_field = encoding_field * self._get_delta_gradient_moment_x() / (self._get_maximum_radius_x() * 2)

        # Print some information of the cartesian trajectory to the console
        Console.add_lines("Created 2D cartesian trajectory:")
        Console.add_lines(f" -> with shape: {encoding_field.shape}")
        ###Console.add_lines(f" -> from parameters:")
        ###Console.add_lines(f"    -> {'field strength':.<20}: {field_strength}")
        ###Console.add_lines(f"    -> {'larmor frequency':.<20}: {larmor_frequency}")
        ###Console.add_lines(f"    -> {'field of view':.<20}: {field_of_view}")
        Console.printf_collected_lines("success")

        return encoding_field

    def get_concentric_rings(self, plot: bool = False) -> list[np.ndarray]:
        """
        The results are the Gradient Moment (GM) values of the concentric rings trajectory.

        \nThis incorporates:
        * the launch track points,
        * the loop track points,
        * the oversampling of the ADC, given by = gradient_raster_time / dwell_time_of_adc_per_angular_interleaf.
          Based on that, the loop points will be interpolated.

        \nNote the abbreviations in the code:
        * GV... Gradient Values
        * GM... Gradient Moments

        \nFurther to note in the code:
        (!) Since pint is used for introducing units, e.g., variable.magnitude just extracts the value without the unit. Don't confuse with the mathematical magnitude |x|!

        :param plot: If True a plot of launch track points, loop points and the normalized combination will be displayed.
        :return: A list of numpy arrays where each entry contains the values of one concentric ring. These values are representing the normalized gradient moment (GM) values (includes launch track + loop track).
        """

        # For being able to work with units
        u = self.u

        # Load the required parameters form defined json files
        parameters_and_data = file.Trajectory(configurator=self.configurator).load_concentric_rings()

        # Get and prepare (shape, right unit) relevant parameters from loaded parameters
        gradient_raster_time = (parameters_and_data["GradientRasterTime"] * u.ms).to(u.us)
        gradient_raster_time = gradient_raster_time.to(u.us)
        dwell_time_of_adc_per_angular_interleaf = (np.asarray(parameters_and_data["DwellTimeOfADCPerAngularInterleaf"]) * u.ns).to(u.us)
        measured_points = parameters_and_data["MeasuredPoints"]
        measured_points = np.asarray(measured_points)[:, 0, :]  # prepare data: pint to numpy array and also drop one dimension

        # Calculate additional necessary parameters
        launch_track_points = measured_points[:, 0] - 1
        number_of_loop_points = measured_points[:, 1] - measured_points[:, 0]
        oversampling_ADC = gradient_raster_time / dwell_time_of_adc_per_angular_interleaf
        oversampling_ADC = oversampling_ADC.magnitude  # dimensionless, thus just get values

        # The Maximum Gradient Amplitude refers to the peak gradient strength:
        #  -> represents the highest magnetic field gradient that the system can generate during the pulse sequence
        #  -> Further reading: https://mriquestions.com/gradient-specifications.html
        maximum_gradient_amplitude = parameters_and_data["MaximumGradientAmplitude"] * u.mT / u.m  # TODO: right unit?

        # Prepare values and lists
        all_rings_GV = [ring * u.mT / u.m for ring in parameters_and_data["GradientValues"]]  # TODO: right unit?
        all_rings_GM = []  # will be filled later in the for loop

        # Create a figure with subplots (4 rows, 3 columns)
        fig, axs = plt.subplots(3, 3, figsize=(15, 12))

        # Calculate the final gradient moment (GM) values for each ring:
        for i, one_ring_GV in tqdm(enumerate(all_rings_GV), total=len(all_rings_GV)):
            # To get the gradient values of the
            #  (a) launch track
            #  (b) loop track and interpolate based on the oversampling factor.
            #      To have no issue with the boundaries in interpolation, the array gets tripled,
            #      which will be deleted later on (2/3 extra values)
            launch_track_GV = one_ring_GV[:launch_track_points[i]]
            loop_track_GV = one_ring_GV[launch_track_points[i] - 1: (launch_track_points[i] + number_of_loop_points[i])]
            loop_track_GV = np.tile(loop_track_GV, 3)  # Triple array for interpolation as described above.

            # To interpolate from current x values to desired x values
            loop_track_length = len(loop_track_GV)
            x_currently = np.arange(0, loop_track_length)
            x_interpolate = np.arange(0, loop_track_length, 1 / oversampling_ADC[i])
            loop_track_GV_interpolated = CubicSpline(x=x_currently, y=loop_track_GV.magnitude, extrapolate=True)(x_interpolate)

            # Remove extra points used for interpolation, thus take central 1/3 values
            x_range = [len(loop_track_GV_interpolated) // 3,      # from 1/3 of the data
                       len(loop_track_GV_interpolated) // 3 * 2]  # ..to 2/3 of the data
            loop_track_GV_interpolated = loop_track_GV_interpolated[x_range[0]:x_range[1]]

            """
            a) Additional explanations for the calculations in the code below:
            
            - Gradient Values........ Defines the speed and direction of movement in k-space.
                                      The actual position in k-space is given by: k(t)=γ⋅∫G(τ)dτ, with G(τ) as gradient values over time.
                                      
            - Gradient Moment (GM)... Cumulative effect of gradients over time, calculated as ∫G(τ)dτ.
            - Gradient Raster Time... In which Δt you are able to update the gradients and thus also effects how often you update the position in k-space.
            - ADC Oversampling....... For sampling the signal at a higher rate than the minimum Nyquist rate, allowing finer sampling of the signal.


            b) To compute the Gradient Moment (GM) in discrete space:

              ∫ GV(τ) dτ  --->  ∑ GV_i ⋅ Δτ

            - Explanation: The integral (∫) of gradient values GV(τ) over continuous time τ is approximated by a summation (∑) of discrete gradient values GV_i, each multiplied by the interval Δτ.
              - dτ: Represents an infinitesimally small time interval in continuous space, corresponding to the gradient raster time in this context.
              - Δτ: The discrete time interval representing the spacing between sampled points, also set by the gradient raster time.

            (!) Note: When oversampling is applied, Δτ is adjusted by dividing it by the ADC oversampling factor, effectively refining the interval Δτ/(ADC factor).
            """

            loop_track_GM = -cumulative_trapezoid(loop_track_GV_interpolated, initial=0) * gradient_raster_time.magnitude / oversampling_ADC[i]
            launch_track_GM = -trapezoid(launch_track_GV.magnitude) * gradient_raster_time.magnitude

            # Then, applying scaling via maximum gradient amplitude
            loop_track_GM = loop_track_GM * maximum_gradient_amplitude[i].magnitude
            launch_track_GM = launch_track_GM * maximum_gradient_amplitude[i].magnitude

            # Combine gradient values (GM) of launch track und loop track
            combined_GM = launch_track_GM + loop_track_GM

            # Normalise the combined gradient moment (GM) values to the maximum radius in x-direction in k-space
            radius_maximum_x = self._get_maximum_radius_x()
            normalised_GM = combined_GM / (radius_maximum_x * 2)

            # Store all ring data in a list of numpy arrays.
            all_rings_GM.append(normalised_GM)

            # Just for plotting the gradient moment (GM) values of the launch track, loop track and normalized combination
            # (Part 1/2)
            if plot:
                sets_for_plotting = [launch_track_GV.magnitude, loop_track_GM, normalised_GM.magnitude]
                sets_names = ["Launch tracks GMs", "Loop tracks GMs", "Normalised GMs"]
                for u, (set_name, set_values) in enumerate(zip(sets_names, sets_for_plotting)):
                    # Plot real and imaginary parts
                    axs[u, 0].plot(set_values.real, set_values.imag, linestyle='-', linewidth=0.5, marker='.', markersize=1.5)
                    axs[u, 1].plot(set_values.real, label="Real Part", linestyle='-', linewidth=0.5, marker='.', markersize=1.5)
                    axs[u, 2].plot(set_values.imag, label="Imaginary Part", linestyle='-', linewidth=0.5, marker='.', markersize=1.5)

                    axs[u, 0].set_title(f"{set_name}: Real vs Imaginary")
                    axs[u, 1].set_title(f"{set_name}: Real Part")
                    axs[u, 2].set_title(f"{set_name}: Imaginary Part")

        # To show the assembled plot if plot = True
        # (Part 1/2)
        if plot:
            plt.tight_layout()
            plt.show()

        # Return a list of numpy arrays (one per concentric ring), each element containing all the gradient moment (GM) values.
        return all_rings_GM


    def flatten_concentric_rings(self, all_rings_GM_list: list) -> np.ndarray:
        """
        To transform a list containing the gradient moment (GM) values of the respective concentric ring as numpy arrays to one numpy array.

        [np.array(GM_values_ring_1), ..., np.array(GM_values_ring_n)] -> np.array(GM_values_all_rings)

        :param all_rings_GM_list:
        :return: a numpy array containing a vector of gradient moment values (of all trajectories).
        """
        return np.concatenate(all_rings_GM_list, axis=0)


    def get_operator(self, input_trajectory_GM: np.ndarray, output_trajectory_GM: np.ndarray, inverse_FT: bool = False) -> np.ndarray:
        """
        To get the operator E for transformation either (A) from i-space (image-space) to k-space or (B) vice versa.

        For a better understanding of the operator:

        (1) Let the 2D Discrete Fourier Transformation (DFT)
              F(kx, ky) = ΣΣ f(x, y) ⋅ e^(-j2π (kx⋅x/N + ky⋅y/M)), with
                    - N and M as the shape of X and Y,
                    - x and y as image space indices, and
                    - kx and ky as k-space positions

        (2) To bring it to a matrix form, let's define all values kx, ky, x, y in their respective vectors kx_vector,
            ky_vector, x_vector, and y_vector. Then let K be the matrix of frequency vectors [kx_vector, ky_vector] and
            R the matrix containing the spatial coordinate vectors [x_vector, y_vector].

        (3) Then, the expression kx⋅x + ky⋅y can be expressed as below, yielding a combination of each frequency
            component (kx, ky) with each spatial coordinate (x, y).

            K × R^T, where T indicates transposing the matrix to fit the dimensionality.

        (4) Finally, to get the complex exponential matrix E:

            E = e^(-j2π ⋅ (K × R^T))

            Each entry E(i, j) in E represents e^(-j2π ⋅ (kx(i)⋅x(j) + ky(i)⋅y(j))), representing each combination of kx, ky, and x, y.

        =========================================

        Then, for the case (A) to bring the image to k-space:

        (5) To bring an image into k-space via this operator, first flatten the image matrix into a vector:

            f ∈ ℝ^(m×n) → f ∈ ℝ^(m⋅n)

        (6) Finally, obtain the Fourier-transformed image F by multiplication:

            F = E × f

        :param input_trajectory_GM: trajectory of the source space,
        :param output_trajectory_GM: trajectory of the target space
        :param inverse_FT: set True if you would like to go from k-space to i-space (image-space), and False in the opposite direction.
        :return: the operator
        """

        # TODO: Check why first imag and then real??
        # TODO: Maybe already with two columns for real and image in previous method
        output_trajectory_GM = np.column_stack((output_trajectory_GM.imag, output_trajectory_GM.real))
        # TODO: Why is here with * self.size_x multiplied? What is the purpose and not better in previous method or does it depend on the operator? Is this a normalisation, but then why multiplication?
        input_trajectory_GM = (input_trajectory_GM * self.size_x).reshape(2, -1)

        # complex angular frequency coefficient
        # TODO: From i-space to k-space -2 normally, right? But get than opposite like Ali. My issue?
        complex_coefficient = (2 if inverse_FT else -2) * np.pi * 1j

        operator = np.exp(complex_coefficient * output_trajectory_GM.magnitude @ input_trajectory_GM.magnitude)

        return operator

    def density_compensation_concentric_rings(self, trajectory:list):
        number_concentric_rings = len(trajectory)
        radii = np.array([np.array(ring.magnitude[0]) for ring in trajectory])
        # Euclidean distance to get radius (from real and imag)
        radii = np.sqrt(radii.real**2 + radii.imag**2)
        # Get maximum radius
        maximum_radius = np.max(radii)

        print(trajectory[0])

        import sys
        sys.exit()

        Rm = maximum_radius
        R = radii
        N = number_concentric_rings
        annuli = [] #should then contain all annulus
        annulus = None

        for i in range(len(radii)):
            if i == 0:
                annulus = ((R[i]/Rm + R[i+1]/Rm)**2 / 4 - (R[i-1]/Rm + R[i]/Rm)**2 / 4)**1 * np.pi
            elif i < len(radii):
                annulus = (0.25 * np.pi * (R[0]/Rm + R[1]/Rm)**2)
            else:
                annulus = np.pi * ((1.5 * R[N-1]/Rm - 0.5 * R[N-2]/Rm)**2 - (0.5 * R[N-1]/Rm + 0.5 * R[N-2]/Rm)**2)

            annuli.append(annulus)

        print(annuli)

        # TODO: Under construction


        #np.pi * ((radii[i]))


    def sampling_density_voronoi_new(self, values_GM, plot=False):
        """
        Compute Density Compensation Based on Voronoi polygon
        """

        # TODO First image or real ????

        # (1) To center trajectory around it's mean to origin (0,0)
        values_GM = values_GM - np.mean(values_GM, axis=0)

        # (2) Remove duplicated points for Voronoi computation
        unique_values_GM, indices, inverse_indices = np.unique(values_GM, axis=0, return_index=True, return_inverse=True)
        Console.printf("info", f"Number of duplicated points removed: {len(values_GM) - len(unique_values_GM)}")

        # (3) Compute the Voronoi Diagram based cleared points.
        #     The Voronoi diagram divides the given space into (called Voronoi regions).
        #     Each Voronoi region has a center (called seed); here the k-space points are used therefore.
        #
        #     Mathematically, a k-space point x belongs to a Voronoi region R[i] if the distance is minimum to its center K[i].
        #     Further note that K[j] are all centers that are not K[i].
        #
        #                     x ∈ R[i] if ∥x-K[i]∥ ≤ ∥x-K[j]∥   ∀ j≠i
        #
        #     Additional infos: The voronoi diagram consists of:
        #                        * edges ........................ borders between regions
        #                        * vertices ..................... point where edges meet
        #                        * seeds (aka centers or sites).. points that define the regions
        #                        * faces ........................ the regions themselves
        voronoi_diagram = Voronoi(unique_values_GM)

        # To plot the voronoi diagram
        if plot:
            plt.figure()
            voronoi_plot_2d(voronoi_diagram, show_vertices=True, line_colors="blue", line_width=0.5)
            plt.scatter(unique_values_GM[:, 0], unique_values_GM[:, 1], color="red", s=20, marker="x", label="k-space points == Seeds here (blue)")
            plt.gca().set_aspect("equal") # equal distance of axes
            plt.title("Voronoi Diagram of k-Space Points")
            plt.xlabel("k_x")
            plt.ylabel("k_y")
            plt.legend()
            plt.show()

        # (4) Define bounding disk for clipping (based on max radius of values_GM in k-space)
        max_radius = np.max(np.linalg.norm(values_GM, axis=1))
        bounding_disk = Point(0, 0).buffer(max_radius)

        # To collect the areas of the Voronoi Regions
        areas = []
        for region_index in voronoi_diagram.point_region:
            region = voronoi_diagram.regions[region_index]

            # Skip the infinite regions
            if not region or -1 in region:
                areas.append(np.nan)
                continue

            # Create a polygon for the respective region
            polygon_coordinates = [voronoi_diagram.vertices[i] for i in region]
            polygon = Polygon(polygon_coordinates)

            # Clip the polygon to the bounding disk
            clipped_polygon = polygon.intersection(bounding_disk)

            # Calculate area of the valid polygon
            area = clipped_polygon.area if not clipped_polygon.is_empty else np.nan
            areas.append(area)

        # Convert areas to a NumPy array
        areas = np.array(areas)

        # Handle NaN areas by assigning max valid area
        max_area = np.nanmax(areas)
        areas = np.where(np.isnan(areas), max_area, areas)

        # Map areas back to the original points
        density_compensation_weights = areas[inverse_indices]

        # Return the weights, which are just the areas of the Voronoi regions corresponding to each k-space point.
        # Thus, each region can be seen as fractions of the total k-space area.
        return density_compensation_weights


    def transformation(self):
        # TODO: One trajectory to the other one (data)!
        pass


if __name__ == "__main__":
    configurator = Configurator(path_folder="/home/mschuster/projects/Synthetic_MRSI/config/", file_name="paths_25092024.json")

    trajectory = Trajectory(configurator=configurator, size_x=128, size_y=64)
    # trajectory.get_cartesian_2D()

    # Radius is defined inside cartesian trajectory
    # -> given by: DeltaGM * DataSize[0] / 2
    # -> DeltaGM computed in cartesian trajectory:
    #     --> DeltaGM = 10**9 / (FOV*gyromagnetic ratio over 2 pi) = 106.75724962474001
    #     --> DataSize[0] = 128 ==> self.size_x
    cartesian_trajectory_GM = trajectory.get_cartesian_2D()

    concentric_rings_trajectory_GM = trajectory.get_concentric_rings(plot=False)
    trajectory.get_operator(input_trajectory_GM=cartesian_trajectory_GM,
                            output_trajectory_GM=trajectory.flatten_concentric_rings(concentric_rings_trajectory_GM),
                            inverse_FT=False)

    # TODO: Change the required input data of the GM values in sampling_density_voronoi_new handles
    # TODO: Remove sampling_density_voronoi
    #trajectory.density_compensation_concentric_rings(concentric_rings_trajectory_GM)
    data = np.vstack((trajectory.flatten_concentric_rings(concentric_rings_trajectory_GM).real, trajectory.flatten_concentric_rings(concentric_rings_trajectory_GM).imag))
    data = np.transpose(data)

    print(trajectory.sampling_density_voronoi(data.magnitude, plot=False).shape)
    print(trajectory.sampling_density_voronoi_new(data.magnitude, plot=False).shape)

    print(trajectory.sampling_density_voronoi(data.magnitude, plot=False)[0:10])
    print(trajectory.sampling_density_voronoi_new(data.magnitude, plot=True)[0:10])

