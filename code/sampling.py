from tools import Console
import dask.array as da
import numpy as np
import cupy as cp
import dask
import sys

from tqdm import tqdm

import matplotlib.pyplot as plt


class Model:
    # TODO docstring

    def __init__(self,
                 computational_graph_spectral_spatial_model: da.Array = None,
                 block_size_computational_graph_spectral_spatial_model: tuple = None,
                 coil_sensitivity_maps: np.ndarray | cp.ndarray | da.Array = None,
                 block_size_coil_sensitivity_maps: tuple = None,
                 path_cache: str = None,
                 data_type: str = "complex64",
                 persist_computational_graph_spectral_spatial_model: bool = False, # Uses more space, but speeds up computation
                 ):

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

    def _to_device(self, array: da.Array, device: str = 'cpu'):
        possible_devices = {'cpu', 'cuda'}
        if device not in possible_devices:
            Console.printf("error", f"Selected device is {device}. However, only {possible_devices} is possible.")
            sys.exit()

        if isinstance(array._meta, np.ndarray) and device == 'cpu':
            return array  # do nothing
        elif isinstance(array._meta, cp.ndarray) and device == 'cuda':
            return array  # do nothing
        elif isinstance(array._meta, np.ndarray) and device == 'cuda':
            return array.map_blocks(cp.asarray)
        elif isinstance(array._meta, cp.ndarray) and device == 'cpu':
            return array.map_blocks(cp.asnumpy)

    def apply_coil_sensitivity_maps(self,
                                    compute_on_device: str = 'cpu',
                                    return_on_device: str = 'cpu') -> list[da.Array]:

        coil_sensitivity_maps = self._to_device(self.coil_sensitivity_maps, device=compute_on_device)

        volumes_with_coil_sensitivity_maps: list[da.Array] = []

        for i in range(coil_sensitivity_maps.shape[0]):
            volumes_with_coil_sensitivity_maps.append(
                self._to_device(self.computational_graph_spectral_spatial_model[:, :, :, :] * self.coil_sensitivity_maps[i, :, :, :], device=return_on_device))
        return volumes_with_coil_sensitivity_maps

    def cartesian_FFT(self,
                      volumes_with_coil_sensitivity_maps: list[da.Array],
                      crop_center_shape: tuple = None,
                      compute_on_device: str = 'cpu',
                      return_on_device: str = 'cpu') -> list[da.Array]:

        volumes_cartesian_k_space: list[da.Array] = []
        for one_coil_image_domain in volumes_with_coil_sensitivity_maps:
            one_coil_image_domain = self._to_device(one_coil_image_domain, device=compute_on_device)

            one_coil_k_space = one_coil_image_domain.map_blocks(cp.fft.fftn, dtype=cp.complex64, axes=(1, 2, 3))
            one_coil_k_space_shifted = one_coil_k_space.map_blocks(cp.fft.fftshift, dtype=cp.complex64, axes=(1, 2, 3))
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

        return volumes_cartesian_k_space

    def cartesian_IFFT(self,
                       volumes_cartesian_k_space: list[da.Array],
                       compute_on_device: str = 'cpu',
                       return_on_device: str = 'cpu') -> list[da.Array]:

        volumes_image_domain: list[da.Array] = []
        for one_coil_k_space in volumes_cartesian_k_space:
            one_coil_k_space = self._to_device(one_coil_k_space, device=compute_on_device)

            one_coil_image_domain = one_coil_k_space.map_blocks(cp.fft.ifftshift, dtype=cp.complex64, axes=(1, 2, 3))
            one_coil_image_domain = one_coil_image_domain.map_blocks(cp.fft.ifftn, dtype=cp.complex64, axes=(1, 2, 3))
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
                                                                              int(one_coil_volume.shape[1]/20),
                                                                              int(one_coil_volume.shape[2]/20),
                                                                              int(one_coil_volume.shape[3]/20))), device=compute_on_device)

            #### Get the spectra of all FID signals
            ###spectra = da.map_blocks(cp.fft.fft, one_coil_volume, axis=0)
            #### Get the mean FID signa
            ###mean_spectrum = spectra.mean(axis=(1, 2, 3))
            ###mean_spectrum = mean_spectrum.rechunk(chunks=one_coil_volume.shape[0])
            ####mean_spectrum = da.map_blocks(cp.asnumpy, mean_spectrum).compute()
            #### Get max peak in spectrum
            ###max_peak = da.map_blocks(cp.amax, mean_spectrum) #.compute()
            max_peak = cp.array(53.308907-60.410015j)
            # Get the desired standard deviation based on the desired snr

            noise_std_desired = max_peak / (2 * snr_desired)
            # Generate noise
            #print(one_coil_volume.shape[0])
            #print(noise_std_desired.compute())
            #input("=================================")
            #noise = cp.random.normal(0+0j, noise_std_desired.compute(), size=one_coil_volume.shape[0])

            noise_real = cp.random.normal(0, cp.abs(noise_std_desired), size=one_coil_volume.shape[0], dtype=cp.float32)
            noise_imag = cp.random.normal(0, cp.abs(noise_std_desired), size=one_coil_volume.shape[0], dtype=cp.float32)
            noise = noise_real + 1j * noise_imag
            noise = noise.astype(cp.complex64)
            noise = noise.reshape(1536, 1, 1, 1)

            #noise = da.asarray(noise, chunks=one_coil_volume.shape[0])
            #noise = da.random.normal(0, noise_std_desired, size=mean_spectrum.shape, chunks=mean_spectrum.compute_chunk_sizes())
            #noise = noise.map_blocks(cp.asarray) #.compute()
            # Add to each spectrum (each voxel)
            one_coil_volume_noisy = one_coil_volume + noise
            one_coil_volume_noisy = one_coil_volume_noisy.rechunk(chunks=one_coil_volume.shape)

            volumes_image_domain_noisy.append(one_coil_volume_noisy)

        return volumes_image_domain_noisy




    def coil_combination(self,
                         volumes_with_coil_sensitivity_maps: list[da.Array],
                         compute_each_coil: bool = True,
                         compute_on_device: str = 'cpu',
                         return_on_device: str = 'cpu') -> da.Array | np.ndarray | cp.ndarray:

        shape = volumes_with_coil_sensitivity_maps[0].shape
        dtype = volumes_with_coil_sensitivity_maps[0].dtype

        cumulative_sum_coils = np.zeros(shape=shape, dtype=dtype) if compute_on_device else cp.zeros(shape=shape, dtype=dtype)

        for i, one_coil_volume in tqdm(enumerate(volumes_with_coil_sensitivity_maps), total=len(volumes_with_coil_sensitivity_maps), desc="coil combination"):
            Console.printf("info", f"Start to include coil sensitivity map:{i} / {len(volumes_with_coil_sensitivity_maps)}")
            one_coil_volume = self._to_device(one_coil_volume, device=compute_on_device)

            if compute_each_coil:
                cumulative_sum_coils += self._to_device(one_coil_volume, device=return_on_device).compute()
                print(f"compute and sum up coil {i}")
            else:
                cumulative_sum_coils += one_coil_volume

            print(cumulative_sum_coils.shape) # TODO: Remove!!!

        if compute_each_coil:
            if isinstance(cumulative_sum_coils, np.ndarray) and return_on_device == 'cpu':
                return cumulative_sum_coils  # do nothing
            elif isinstance(cumulative_sum_coils, cp.ndarray) and return_on_device == 'cuda':
                return cumulative_sum_coils  # do nothing
            elif isinstance(cumulative_sum_coils, np.ndarray) and return_on_device == 'cuda':
                return cp.asarray(cumulative_sum_coils)
            elif isinstance(cumulative_sum_coils, cp.ndarray) and return_on_device == 'cpu':
                return cp.asnumpy(cumulative_sum_coils)
        else:
            return self._to_device(cumulative_sum_coils, device=return_on_device)

        return cumulative_sum_coils

###class Model:
###    def __init__(self, spectral_spatial_model: SpectralSpatialModel, data_type: str = "complex64", compute_on_device: str = "cpu", return_on_device: str = "cpu"):
###
###        # Input only dask graph or also numpy, cupy array?
###        self.spectral_spatial_model: SpectralSpatialModel = spectral_spatial_model
###        self.computational_graph = self.spectral_spatial_model.assemble_graph()
###        self.compute_on_device = compute_on_device
###        self.return_on_device = return_on_device
###
###        if self.compute_on_device == "cuda":
###            self.xp = cp  # use cupy
###        elif self.compute_on_device == "cpu":
###            self.xp = np  # use numpy
###
###    def cartesian_FFT_graph(self):
###        computational_graph = self.computational_graph.map_blocks(self.xp.fft.fftn, dtype=self.xp.complex64, axes=(1, 2, 3))
###
###        if self.compute_on_device == "cuda" and self.return_on_device == "cpu":
###            computational_graph = self.computational_graph.map_blocks(cp.asnumpy)
###        elif self.compute_on_device == "cpu" and self.return_on_device == "cuda":
###            computational_graph = self.computational_graph.map_blocks(cp.asarray)
###
###        return computational_graph
