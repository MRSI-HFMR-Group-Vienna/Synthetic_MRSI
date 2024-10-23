from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import CubicSpline
from file import Configurator
from tools import Console
import dask.array as da
import numpy as np
import cupy as cp
import pint
import dask
import file
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
    def __init__(self, configurator: file.Configurator, size_x: int, size_y: int):
        self.configurator = configurator
        self.size_x = size_x
        self.size_y = size_y

    def get_cartesian_2D(self) -> None:
        """
        To simulate other relevant parameters after loading the parameters from the json file.

        TODO: check if units are correct of "encoding_field"
        TODO: self.size_x, self.size_y has no units? (was nFreqEnc and nPhasEnc)

        :return: None
        """

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
        maximum_radius_x = delta_gradient_moment_x * size_x / 2  # maximum trajectory radius
        encoding_field = encoding_field * delta_gradient_moment_x / (maximum_radius_x * 2)

        Console.add_lines("Created 2D cartesian trajectory:")
        Console.add_lines(f" -> with shape: {encoding_field.shape}")
        Console.add_lines(f" -> from parameters:")
        Console.add_lines(f"    -> {'field strength':.<20}: {field_strength}")
        Console.add_lines(f"    -> {'larmor frequency':.<20}: {larmor_frequency}")
        Console.add_lines(f"    -> {'field of view':.<20}: {field_of_view}")
        Console.printf_collected_lines("success")

        return encoding_field


    def get_concentric_rings(self, radius_maximum: float, plot:bool = False):
        # TODO: Girf not included, but maybe it shouldn't be here


        #-------------------------
        # GV ... gradient values
        # GM ... gradient moment
        #-------------------------

        u = pint.UnitRegistry()

        # Load the required parameters form defined json files
        parameters_and_data = file.Trajectory(configurator=self.configurator).load_concentric_rings()

        # Get and prepare relevant parameters from loaded parameters
        gradient_raster_time = (parameters_and_data["GradientRasterTime"] * u.ms).to(u.us)
        gradient_raster_time = gradient_raster_time.to(u.us)
        dwell_time_of_adc_per_angular_interleaf = (np.asarray(parameters_and_data["DwellTimeOfADCPerAngularInterleaf"]) * u.ns).to(u.us)
        measured_points = parameters_and_data["MeasuredPoints"]
        measured_points = np.asarray(measured_points)[:, 0, :]  # prepare data: pint to numpy array and also drop one dimension


        # Maximum Gradient Amplitude refers to the peak gradient strength
        #  -> represents the highest magnetic field gradient that the system can generate during the pulse sequence
        #  -> Further reading: https://mriquestions.com/gradient-specifications.html
        maximum_gradient_amplitude = parameters_and_data["MaximumGradientAmplitude"] * u.mT / u.m # TODO: right unit?

        all_rings_GV = [ring*u.mT/u.m for ring in parameters_and_data["GradientValues"]]  # TODO: right unit?


        # Calculate additional necessary parameters
        launch_track_points = measured_points[:, 0] - 1
        number_of_loop_points = measured_points[:, 1] - measured_points[:, 0]
        oversampling_ADC = gradient_raster_time / dwell_time_of_adc_per_angular_interleaf
        oversampling_ADC = oversampling_ADC.magnitude  # dimensionless, thus just get values

        #gradient_raster_time = gradient_raster_time.magnitude # just extract values without the unit for further operations

        # Since the Gradient Moment (GM) is given by GM=∫G(t)dt

        for i, one_ring_GV in enumerate(all_rings_GV):
            """
            GV... Gradient Values
            """

            ###
            #Interpolation of the GM values
            ###
            trajectory_GV = one_ring_GV[launch_track_points[i] - 1: (launch_track_points[i] + number_of_loop_points[i])]
            trajectory_GV = np.tile(trajectory_GV, 3)

            # To interpolate from current x values to desired x values
            trajectory_length = len(trajectory_GV)
            x_currently = np.arange(0, trajectory_length)
            x_interpolate = np.arange(0, trajectory_length, 1 / oversampling_ADC[i])
            trajectory_GV_interpolated = CubicSpline(x=x_currently, y=trajectory_GV, extrapolate=False)(x_interpolate)

            # Remove extra points used for interpolation, thus take central 1/3 values
            x_range = [len(trajectory_GV_interpolated) // 3,     # from 1/3 of the data
                       len(trajectory_GV_interpolated) // 3 * 2] # ..to 2/3 of the data
            trajectory_GV_interpolated = trajectory_GV_interpolated[x_range[0]:x_range[1]]


            #TODO: DID MOT USE INTERPOLATED VALUES!!!!

            # to how finely the MRI system is able to encode spatial frequencies
            # -> Maximum Gradient Amplitude dictates how fast you can move through k-space
            # -> Gradient Raster Time dictates how often you update the position in k-space.
            # -> ADC Oversampling adjusts how finely you sample the signal relative to the Nyquist criterion.
            #maximum_gradient_amplitude * gradient_raster_time / oversampling_ADC

            # Maximum Gradient Amplitude: "speed" through the k-space
            # Gradient Raster Time: Update frequency of position in k-space
            # ADC Oversampling: how finely the signal gets sampled in relation to the Nyquist criterion

            #kummulativen effeckt über die zeit
            # gesamteffeckt von gradienten in k-raum -> gesamtakkumulierte position?


            #  = ∫G(t)*G_max*Δt/OSR
            # Let G be the gradient, G_max the maximal gradient amplitude, Δt as gradient raster time and OSR the oversampling rate.

            # Gradient values........ Defines the speed and the direction of the movement in k-space.
            #                         The real position is given by: k(t)=γ*∫G(τ)dτ
            #
            # Gradient Moment (GM)... Cumulative effect of gradients over time = ∫G(τ)dτ
            #
            # Gradient Raster Time... In which Δt you are able to update the gradients and thus also effects
            #                         how often you update the position in k-space.
            #
            # ADC oversampling....... How finely you sample the signal relative to the Nyquist criterion.

            scaling_factor_loop_track = maximum_gradient_amplitude[i].magnitude * gradient_raster_time.magnitude / oversampling_ADC[i]
            loop_track_GM = -cumulative_trapezoid(trajectory_GV_interpolated, initial=0) * scaling_factor_loop_track

            launch_track_GV = one_ring_GV[:launch_track_points[i]]
            scaling_factor_launch_track = maximum_gradient_amplitude[i].magnitude * gradient_raster_time.magnitude
            launch_track_GM = -trapezoid(launch_track_GV) * scaling_factor_launch_track # gradient_raster_time.magnitude to get only value without unit, otherwise warning message. Nothing else!


            final_GM = launch_track_GM + loop_track_GM

            radius_maximum = self.size_x *

        print(len(all_rings_GV))
        input("- - - -")



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
    trajectory.get_concentric_rings()
