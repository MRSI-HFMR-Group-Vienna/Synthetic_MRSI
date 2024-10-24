from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from file import Configurator
from tools import Console
import dask.array as da
from tqdm import tqdm
import numpy as np
import cupy as cp
import pint
import dask
import file
import sys




class Model:
    # TODO docstring

    def __init__(self,
                 computational_graph_spectral_spatial_model: da.Array = None,
                 block_size_computational_graph_spectral_spatial_model: tuple = None,
                 coil_sensitivity_maps: np.ndarray | cp.ndarray | da.Array = None,
                 block_size_coil_sensitivity_maps: tuple = None,
                 path_cache: str = None,
                 data_type: str = "complex64",
                 persist_computational_graph_spectral_spatial_model: bool = False,  # Uses more space, but speeds up computation
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
        """
        TODO: Bring to tools -> Because general usage possible.
        TODO: Then also use at other places? (E.g. spatial_spectral_simulation.Model.assemble_graph (end of code))

        To bring a dask array to the desired device. 'cuda' and 'cpu' is supported.

        :param array:
        :param device:
        :return:
        """
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
        """
        For applying the coil sensitivity maps to the volume of the spectral spatial model.

        :param compute_on_device: target device (cpu or cuda)
        :param return_on_device: target device (cpu or cuda)
        :return: list of dask arrays
        """

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
        """
        For bringing the volume into the cartesian k-space via fast fourier transformation (FFT). Usually used after
        apply the coil sensitivity maps, thus this method deals with a list of dask arrays.

        :param volumes_with_coil_sensitivity_maps: the volume with applied coil sensitivity maps.
        :param crop_center_shape: tuple with shape values for a symmetrical cut around the center of the volume.
        :param compute_on_device: target device (cpu or cuda)
        :param return_on_device: target device (cpu or cuda)
        :return: list of volumes (one for each coil) in cartesian k-space
        """

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

    def non_cartesian_FT(self):
        " TODO TODO TODO TODO TODO"
        pass
        # load necessary data from json file
        import json
        with open('data.json', 'r') as file:
            data = json.load(file)

        # data_shape =

    # def non_cartesian_FT
    # -> this function calculates/uses and operator and multiplies them for getting the volume in non cartesian k-space
    # -> maybe: to cartesian k-space
    # -> maybe: to concentric ring k-sapce? or just to non_cartesian_k_space
    #                                               --> because in and out trajectory is defined! (thus general naming like non_cartesian_k_space/non_cartesian FT?)
    # ----> need to read the (crt) concentric ring trajectory from json file
    # ----> need to calculate the cartesian trajectory (based on parameters -> where?)

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

            print(cumulative_sum_coils.shape)  # TODO: Remove!!!

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

    def _get_delta_gradient_moment_x(self) -> pint.Quantity:
        """
        Just a helper function.

        \nFormular:\n ΔG_x = 1/(FOV_x⋅γ/2π). The bigger the field of view (FOV) the smaller the gradient moment, which yields a lower spatial resolution.

        :return: A pint Quantity with a numpy array inside. Thus, the delta gradient moment (ΔGM) in x direction with unit.
        """

        # To be able to use units
        u = pint.UnitRegistry()

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
        encoding_field = np.indices((size_x, size_y))
        # (b) now, bring the zeros in the center of the field and thus also check if
        #     dimension shape is odd or even with modulo operator. (Shorthand If Else btw.)
        x_field, y_field = encoding_field[0, :, :], encoding_field[0, :, :]
        encoding_field[0, :, :] = x_field - np.floor(size_x / 2) + 0.5 if size_x / 2 % 2 == 0 else x_field - np.floor(size_x / 2)
        encoding_field[1, :, :] = y_field - np.floor(size_y / 2) + 0.5 if size_y / 2 % 2 == 0 else y_field - np.floor(size_y / 2)
        # (c) create delta gradient moment field finally & normalize trajectory:
        encoding_field = encoding_field * self._get_delta_gradient_moment_x() / (self._get_maximum_radius_x() * 2)

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

        :param plot: If True a plot of launch track points, loop points and the normalized combination will be displayed.
        :return: A list of numpy arrays. Each numpy array represents the normalized gradient moment (GM) values of one concentric ring trajectory (includes launch track + loop track).
        """

        # For beeing able to work with units
        u = pint.UnitRegistry()

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
            loop_track_GV_interpolated = CubicSpline(x=x_currently, y=loop_track_GV, extrapolate=False)(x_interpolate)

            # Remove extra points used for interpolation, thus take central 1/3 values
            x_range = [len(loop_track_GV_interpolated) // 3,  # from 1/3 of the data
                       len(loop_track_GV_interpolated) // 3 * 2]  # ..to 2/3 of the data
            loop_track_GV_interpolated = loop_track_GV_interpolated[x_range[0]:x_range[1]]

            """
            Some additional explanations for the calculations in the code below:
            
            Gradient values........ Defines the speed and the direction of the movement in k-space.
                                    The real position is given by: k(t)=γ*∫G(τ)dτ
            
            Gradient Moment (GM)... Cumulative effect of gradients over time = ∫G(τ)dτ, with G as gradient values
            
            Gradient Raster Time... In which Δt you are able to update the gradients and thus also effects
                                    how often you update the position in k-space.
            
            ADC oversampling....... How finely you sample the signal relative to the Nyquist criterion.
            
            Note: 'scaling factor' is just a random name, introduced, since it 'scales' the GM values.
            """

            # Just factory that modify the Gradient Moment (GM) values
            scaling_factor_launch_track = maximum_gradient_amplitude[i].magnitude * gradient_raster_time.magnitude
            scaling_factor_loop_track = maximum_gradient_amplitude[i].magnitude * gradient_raster_time.magnitude / oversampling_ADC[i]
            # To get gradient moment values GM = ∫GV(τ)dτ, but also product with a respective scaling factor
            loop_track_GM = -cumulative_trapezoid(loop_track_GV_interpolated, initial=0) * scaling_factor_loop_track
            launch_track_GM = -trapezoid(launch_track_GV) * scaling_factor_launch_track

            # TODO: combined in the right way?
            # Combine gradient values (GM) of launch track und loop track
            combined_GM = launch_track_GM + loop_track_GM

            # Normalise the combined gradient moment (GM) values to the maximum radius in x-direction in k-space
            radius_maximum_x = self._get_maximum_radius_x()
            normalised_GM = combined_GM / (radius_maximum_x * 2)

            # Store all ring data in a list of numpy arrays
            all_rings_GM.append(normalised_GM)

            # Just for plotting the gradient moment (GM) values of the launch track, loop track and normalized combination
            # (Part 1/2)
            if plot:
                sets_for_plotting = [launch_track_GV, loop_track_GM, normalised_GM]
                sets_names = ["Launch tracks GMs", "Loop tracks GMs", "Normalised GMs"]
                for i, (set_name, set_values) in enumerate(zip(sets_names, sets_for_plotting)):
                    # Plot real and imaginary parts
                    axs[i, 0].plot(set_values.real, set_values.imag, linestyle='-', linewidth=0.5, marker='.', markersize=1.5)
                    axs[i, 1].plot(set_values.real, label="Real Part", linestyle='-', linewidth=0.5, marker='.', markersize=1.5)
                    axs[i, 2].plot(set_values.imag, label="Imaginary Part", linestyle='-', linewidth=0.5, marker='.', markersize=1.5)

                    axs[i, 0].set_title(f"{set_name}: Real vs Imaginary")
                    axs[i, 1].set_title(f"{set_name}: Real Part")
                    axs[i, 2].set_title(f"{set_name}: Imaginary Part")

        # To show the assembled plot if plot = True
        # (Part 1/2)
        if plot:
            plt.tight_layout()
            plt.show()

        # Return the list of numpy arrays, containing the gradient moment (GM) values for the concentric ring trajectory (CRT).
        return all_rings_GM


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


if __name__ == "__main__":
    configurator = Configurator(path_folder="/home/mschuster/projects/Synthetic_MRSI/config/", file_name="paths_25092024.json")

    trajectory = Trajectory(configurator=configurator, size_x=128, size_y=64)
    # trajectory.get_cartesian_2D()

    # Radius is defined inside cartesian trajectory
    # -> given by: DeltaGM * DataSize[0] / 2
    # -> DeltaGM computed in cartesian trajectory:
    #     --> DeltaGM = 10**9 / (FOV*gyromagnetic ratio over 2 pi) = 106.75724962474001
    #     --> DataSize[0] = 128 ==> self.size_x
    trajectory.get_concentric_rings(plot=False)
