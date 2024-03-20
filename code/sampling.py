from spectral_spatial_simulation import Model as SpectralSpatialModel
from torch.utils.data import Dataset, DataLoader
from tools import ConfiguratorGPU
from file import Console
from tqdm import tqdm
import numpy as np
import tools
import torch
import sys

import os


# Class FourierTransform

class SubVolumeDataset(Dataset):
    """
    For getting sub-volumes from the main volume. It inherits the PyTorch Dataset, thus it can be used together with
    the PyTorch dataloader. The sub volumes do not need to be isotropic!
    Example usage 1: Iterate over each sub volume
    Example usage 2: Get sub-volume by index (index_x, index_y, index_z) in main volume
    Example usage 2: Get batch of sub-volumes (e.g., [3, 50, 50, 50, 30]). Useful, if put on GPU and limited space is available there.
                     Thus, put batch on GPU, process it and then put back to CPU.
    """

    def __init__(self, volume: np.ndarray | np.memmap, sub_volume_shape: tuple):
        self.volume = volume
        self.sub_volume_shape = sub_volume_shape
        self.indices = self._initialize_indices()

    def get_by_index(self, i_x: int, i_y: int, i_z: int) -> torch.Tensor:
        """
        Get sub-volume by indices in the main volume. Thus, if the main volume, e.g., is divided into eight sub-volumes, maximum
        indices are (7,7,7).

        :param i_x: sub-volume index in the main volume in x direction
        :param i_y: sub-volume index in the main volume in y direction
        :param i_z: sub-volume index in the main volume in z direction

        :return: sub-volume as PyTorch Tensor
        """
        sx, sy, sz = self.sub_volume_shape
        sub_volume = self.volume[i_x:i_x + sx, i_y:i_y + sy, i_z:i_z + sz, :]
        sub_volume = torch.from_numpy(sub_volume).detach()
        return sub_volume

    def _initialize_indices(self):
        """
        Find starting indices for sub-volumes in whole volume. Then, append it to a list containing all possible starting indices of
        the sub volumes.
        """

        # (1) Get target shape of sub volumes.
        sx, sy, sz = self.sub_volume_shape
        unfolded = self.volume.unfold(0, sx, sx).unfold(1, sy, sy).unfold(2, sz,
                                                                          sz)  # creates shape: [sub_x_index, sub_y_index, sub_z_index, t_vector, sub_x_volume, sub_y_volume, sub_z_volume]
        # e.g., [0,0,0,:,:,:,:] -> get x,y,z with t of index 0 (=index of sub volume)

        # Get starting indices of sub-volumes (e.g., x=0,y=0,z=50). Create it for each sub-volume and append to a list as tuple.
        indices = []

        u = 0
        # Get each possible starting index for each sub-volume in the whole volume (e.g., x=0,y=0,z=50)
        for i_x in range(unfolded.shape[0]):
            for i_y in range(unfolded.shape[1]):
                for i_z in range(unfolded.shape[2]):
                    # print(f"indices new = {x,y,z}")
                    indices.append((i_x * sx, i_y * sy, i_z * sz))

        return indices

    def __len__(self):
        """
        To get the number of sub-volumes.
        """
        return len(self.indices)

    def __getitem__(self, index):
        """
        Cut sub volumes out of the whole volume using starting indices and size of sub volumes.
        """
        i_x, i_y, i_z = self.indices[index]
        sx, sy, sz = self.sub_volume_shape
        sub_volume = self.volume[i_x:i_x + sx, i_y:i_y + sy, i_z:i_z + sz, :]  # start and end index to cut out sub-volume of whole volume
        sub_volume = torch.from_numpy(sub_volume).detach()
        return sub_volume


class VolumeForFFTDataset(Dataset):
    """
    TODO: Does the Dataset automatically reduces the shape?
    """

    def __init__(self, data: np.ndarray | np.memmap, device: str):
        self.data = data
        self.device = device
        # self.transform = transform

    def __len__(self):
        return self.data.shape[4]  # Length of the time 't' dimension

    def __getitem__(self, idx):
        batch = self.data[:, :, :, 0, idx]  # Get batch along the 'i' dimension
        batch = batch.copy()  # detach it from the memmap after it is in ram -> speeds it up?
        # batch = np.squeeze(batch, axis=3)   #
        # batch = self.transform(batch)      # Transform numpy array to torch tensor!
        batch = torch.from_numpy(batch).detach()  # detach also disables gradient tracking!
        batch = batch.to(self.device)  # Push batch to cuda
        return batch, idx  # Return batch and corresponding indices


class Model:
    """
    TODO: Until now only contains functionalities to do cartesian FT and to get sub volume of transformed main volume.
    """

    def __init__(self, spectral_model, sub_volume_shape: tuple = None):  # TODO class Type
        self.spectral_spatial_model: SpectralSpatialModel = spectral_model
        self.whole_volume: np.memmap = None
        self.sub_volume_iterator = None
        self.path_cache: str = spectral_model.path_cache # TODO use configurator instead!!!!!!!!
        self.sub_volume_shape: tuple = sub_volume_shape

    # build() -> put next two functions inside, also build make sub-volume?
    # cartesian_FT
    # non-cartesian_FT

    # get_sub_volume(i_x,i_y,i_z)
    # get_sub_volume_iterative()

    def build(self):
        raise NotImplementedError("This method is not yet implemented")

    def create_sub_volume_iterative(self) -> None:
        """
        To divide the main volume in sub volumes. Each sub volume has an index in the main volume.
        Also, these sub-volumes are iterable then and represent the main volume.
        However, it is stored in an extra variable.

        :return: Nothing
        """

        self.sub_volume_iterator = SubVolumeDataset(volume=self.whole_volume, sub_volume_shape=self.sub_volume_shape)

    def get_sub_volume(self, i_x: int, i_y: int, i_z: int) -> torch.Tensor:
        """
        To get sub-volume by index. Each sub volume has an index in the main volume.

        :param i_x: index of sub-volume in the main volume along x-axis
        :param i_y: index of sub-volume in the main volume along y-axis
        :param i_z: index of sub-volume in the main volume along z-axis
        :return: 4D sub-volume of desired index with shape
        """

        # create sub volume first if not exists
        if self.sub_volume_shape is None:
            self.create_sub_volume_iterative()

        return self.sub_volume_iterator.get_by_index(i_x, i_y, i_z)

    def cartesian_FT(self, file_name_cache: str, auto_gpu: bool = True, custom_batch_size: int = None):
        """
        Cartesian Fourier Transformation of the 4D volume. Thus, for each volume in time a FT is performed
        (for each 'time point' in [:,:,:,'time point']) and then the data is merged again.
        For example, let the shape be (112, 128, 80, 1536), then for the 1536 time points the volume (112, 128, 80)
                     is FT separate and then merged -> Yielding a target volume of (112, 128, 80, 1536) again.

        :param file_name_cache: name of the cache file on disk where target volume (numpy memmap) should be stored
        :param auto_gpu: select if the Fourier Transformation (FT) should be automatically performed on GPU if available
        :param custom_batch_size: custom shape of each batch processed. If not set, then automatic batch size is selected
        by the method (for GPU computation depending on available space, and for CPU simply one).
        :return:
        """

        # TODO: Also implement to load if file_name_cache is available in cache path!
        Console.printf_section("Sampling -> FFT")

        # Get the new shape of the target volume ([x,y,z,1,t] -> [x,y,z,t])
        v = self.spectral_spatial_model.volume
        volume_shape = v.shape[0], v.shape[1], v.shape[2], v.shape[4]

        # Create the memmap file for the target volume
        path_cache_file = os.path.join(self.path_cache, f"{file_name_cache}_sampling_volume.npy")
        self.whole_volume = np.memmap(path_cache_file, dtype=self.spectral_spatial_model.volume.dtype, mode='w+', shape=volume_shape)
        Console.printf("info", f"Create new cache volume with shape {volume_shape} for 'sampling' in path: {path_cache_file}")

        # OPTION 1: Use the GPU
        if torch.cuda.is_available() and auto_gpu:

            # (a) Estimate space required to put whole tensor on GPU (bytes)
            #     Also: Check if required space is available and select the least busy GPU!
            space_required_bytes = tools.SpaceEstimator.for_numpy(data_shape=self.spectral_spatial_model.volume.shape,
                                                                  data_type=self.spectral_spatial_model.volume.dtype,
                                                                  unit="byte")
            configurator_gpu = ConfiguratorGPU(required_space_gpu=space_required_bytes)
            configurator_gpu.print_available_gpus()
            configurator_gpu.select_least_busy_gpu()

            # (b) Create custom dataset, it transforms the numpy memmap to a torch tensor
            #     Also, calculate max batch size that fits on GPU.
            dataset = VolumeForFFTDataset(data=self.spectral_spatial_model.volume, device="cuda")
            number_of_batches = np.ceil(configurator_gpu.required_space_gpu / configurator_gpu.free_space_selected_gpu)
            size_of_batches = int(np.floor(len(dataset) / number_of_batches))

            # (c) Create the Dataloader with the Custom Dataset. Based on the batch size, the DataLoader creates a
            #     number n batches.
            size_of_batches = size_of_batches if custom_batch_size is None else custom_batch_size
            Console.add_lines(f"Performing cartesian FFT on CPU")
            if custom_batch_size is not None:
                Console.add_lines(f"Manual user selected batch size: {custom_batch_size}")
            else:
                Console.add_lines(f"Automatically set batch size (based on free GPU space): {size_of_batches}")
            Console.printf_collected_lines("info")
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=size_of_batches,
                                    shuffle=False,
                                    drop_last=False)

        # OPTION 2: Use the CPU
        else:
            dataset = VolumeForFFTDataset(data=self.spectral_spatial_model.volume, device="cpu")
            size_of_batches = len(dataset) if custom_batch_size is None else custom_batch_size
            if custom_batch_size is not None:
                Console.add_lines(f"Manual chosen batch size by user: {custom_batch_size}")
            else:
                Console.add_lines(f"Automatically set batch size: {size_of_batches}")
            Console.printf_collected_lines("info")
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=size_of_batches,
                                    shuffle=False,
                                    drop_last=False)

        # The following fits for GPU and CPU:
        indent = " " * 22
        # (1) Iterate through each batch. If GPU is chosen, then the respective batch is put on the GPU (see VolumeForFFTDataset)
        for i, (batch, indices) in tqdm(enumerate(dataloader), total=len(dataloader), desc=indent + "3D FFT of batches (each t in [:,:,:,t])", position=0):
            # change order of batch from [t,x,y,z] back to [x,y,z,t] because DataLoader changes it automatically!
            batch = batch.permute(1, 2, 3, 0)
            batch_size = batch.size()[-1]

            # (2) Iterate through each time point of the whole volume [:,:,:,t] and do FFT (the fast one ;)).
            for t in range(batch_size):
                batch[:, :, :, t] = torch.fft.fftn(batch[:, :, :, t])

            # (3) Ensure that the volume is back on the CPU when assigning it to the target volume on disk (the memmap)
            #     it has no effect, if the task was already performed on the CPU!
            self.whole_volume[:, :, :, indices] = batch.cpu().numpy()

            # (4) Flush the changes to disk (memmap cache file)
            self.whole_volume.flush()

            # (5) Also ensure that the cache on the GPU is released. has no effect if it was performed on the CPU!
            torch.cuda.empty_cache()

    def non_cartesian_FT(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    # def non_cartesian

    # Requirement:
    # -> Cartesian FFT
    # -> Non cartesian FFT
