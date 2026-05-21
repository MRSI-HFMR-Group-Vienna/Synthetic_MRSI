from bokeh.core.property.container import Array
from numba.np.arrayobj import array_shape
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from resources import Configurator
from prettyconsole import Console
import dask.array as da
from tqdm import tqdm
import numpy as np
import cupy as cp
import pint
import dask
from file import Trajectory as FileTrajectory
import sys
from tools import InterpolationTools, DaskTools, GPUTools, FourierTools, ArrayTools
from interface import Interpolation
import os
from pathlib import Path
import copy



class CoilSensitivityVolume(Interpolation):

    def __init__(self):
        self.coils: list[str] = None
        self.volume: np.ndarray | np.memmap = None

    def interpolate(self, target_size: tuple, order: int = 3, compute_on_device: str = "gpu", return_on_device: str="cpu", target_gpu: int= None, verbose= True):
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

        if target_gpu is None:
            Console.printf("warning", f"Interpolation Coil Sensitivity Volume on default GPU since no desired GPU was specified!")

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
        xp = ArrayTools.get_backend(self.volume)
        self.volume = xp.conjugate(self.volume)
        Console.printf("success", "Complex Conjugated the Coil Sensitivity Volume.")


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
        if not isinstance(coil_sensitivity_volume, CoilSensitivityVolume):
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

    def coil_combination(self, compute_on_device: str, return_on_device: str):
        """
        This performs a coil combination with equation:

        result = Σ S_i * conj(C_i) / Σ |C_i|^2, where "S" is the already created MRSI volume, incorporated the coil sensitivity
                                                volume, and the "C" with the coil volumes. S_i is the respective coil volume of
                                                (time, X, Y, Z) and C_i the respective coil sensitivity map (X, Y, Z) of one coil.

                Already created MRSI volume: ... S_i ∈ ℂ^(coils x time x X x Y x Z)
                Coil volume: ................... C_i ∈ ℂ^(coils x X x Y x Z)

        Note: This is applied in the image space and therefore needs a volume in the image space of (..., X,Y,Z) and which also
              has before applied the coil sensitivity maps to the spectral spatial volume. Otherwise, it may make no sense.

        :param compute_on_device: either 'cpu' or 'gpu'/'cuda'/'cuda'
        :param return_on_device: either 'cpu' or 'gpu'/'cuda'/'cuda'
        :return: dask Array
        """

        # (0) Check if working volume is already cerated and if the number of axes is correct
        #     a) Check if working volume is already created.
        if self.working_volume is None:
            Console.printf("error", "Working volume is None. Have you applied 'apply_coil_sensitivity' before?")
            return
        #     b) Check if working volume has already desired number of axes. This indirectly checks if coil sensitivity maps
        #     were already applied.
        elif len(self.working_volume.shape) != 5:
            Console.printf("error", f"Working volume shape is not 5. It is instead: {self.working_volume.shape}. Cannot perform coil combination!")
            return

        # (1) Interpolate to target size (of other volume). No task yet required!
        working_volume = self.working_volume
        coil_sensitivity_volume_shape_before = self.coil_sensitivity_volume.volume.shape
        #     The following: compute_on_device, return_on_device => both compute_on_device input argument since not lst operation in this method!
        coil_sensitivity_volume = copy.deepcopy(self.coil_sensitivity_volume)
        #     Interpolat on GPU if compute on device is gpu/cuda and return on cpu to avoid gpu allocation issues
        coil_sensitivity_volume.interpolate_volume(compute_on_device=compute_on_device, return_on_device="cpu", target_size=working_volume.shape[2:], target_gpu=self.target_gpu_smaller_tasks, verbose=False)
        Console.printf("success", f"Interpolated coil sensitivity volume: {coil_sensitivity_volume_shape_before} => {coil_sensitivity_volume.volume.shape}")

        # (2) Need complex conjugated coil sensitivity volume
        coil_sensitivity_volume_conjugated = copy.deepcopy(coil_sensitivity_volume)
        coil_sensitivity_volume_conjugated.conjugate()

        # (3) Create dask arrays (+ good junks regarding performance)
        coil_sensitivity_volume_dask = da.asarray(coil_sensitivity_volume.volume, chunks=(self.working_volume.chunksize[0], -1, -1, -1)) # TODO: can make no issue?
        coil_sensitivity_volume_conjugated_dask = da.asarray(coil_sensitivity_volume_conjugated.volume, chunks=(self.working_volume.chunksize[0], -1, -1, -1)) # TODO: can make no issue?

        # (4) Bring to right device via map blocks (thus if GPU does not allocate in advance)
        coil_sensitivity_volume_dask, coil_sensitivity_volume_conjugated_dask = GPUTools.dask_map_blocks_many(coil_sensitivity_volume_dask, coil_sensitivity_volume_conjugated_dask, device=compute_on_device)

        # (5) Need to broadcast conjugated array (from (coil, X, Y, Z) --> (coil, time, X, Y, Z))
        coil_sensitivity_volume_conjugated_dask = coil_sensitivity_volume_conjugated_dask[:,None, ...]

        # (6) Now, this (MRSI volume * coil_sensitivity_complex_conj) / |sum(coil_sensitivity, axis=0)|^2. Note: The working volume is here MRSI volume.
        #    a) Schedule numerator via dask (heavy computation part)
        numerator = da.sum(working_volume * coil_sensitivity_volume_conjugated_dask, axis=0)

        # (7) Compute denominator in the CPU and then map via dask to GPU if desired
        coil_sensitivity_volume_cpu = GPUTools.to_device(coil_sensitivity_volume.volume, device="cpu")
        denominator_cpu = (np.sum(np.abs(coil_sensitivity_volume_cpu) ** 2, axis=0))
        ArrayTools.count_zeros(denominator_cpu)
        ArrayTools.check_nan(denominator_cpu)
        #       Create dask array of result: whole array should be one chunk for performance reasons, and push to gpu
        denominator = da.from_array(denominator_cpu, chunks=(-1, -1, -1))
        denominator = GPUTools.dask_map_blocks(denominator, device=compute_on_device)

        # (8) Perform division
        result = numerator / denominator

        # (9) Return on desired device
        result = GPUTools.dask_map_blocks(result, device=return_on_device)

        return result


class Trajectory:
    """
    To calculate and prepare the trajectories for respective parameters, e.g., stored in json files.
    This includes at the moment the cartesian trajectory and the concentric ring trajectory.
    """

    def __init__(self, configurator: Configurator, size_x: int, size_y: int):
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

        # For introducing units with pint. For just having one pint registry using the class variable from FileTrajectory.
        # Therefore, able to use one registry for different instances of the FileTrajectory.
        self.u = FileTrajectory.get_pint_unit_registry()

    def _get_delta_gradient_moment_x(self) -> pint.Quantity:
        """
        Just a helper function.

        \nFormular:\n ΔG_x = 1/(FOV_x⋅γ/2π). The bigger the field of view (FOV) the smaller the gradient moment, which yields a lower spatial resolution.

        :return: A pint Quantity with a numpy array inside. Thus, the delta gradient moment (ΔGM) in x direction with unit.
        """

        # To be able to use units
        u = self.u

        # Load the required parameters form a defined json file
        parameters = FileTrajectory(configurator=self.configurator).load_cartesian()

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
        parameters_and_data = FileTrajectory(configurator=self.configurator).load_concentric_rings()

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
                    # PlotInterface real and imaginary parts
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