import sys

from spectral_spatial_simulation import Model as SpectralSpatialModel
import matplotlib.pyplot as plt
from file import Console
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import transforms
import torch.multiprocessing as mp
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
        return sub_volume

    def _initialize_indices(self):
        """
        Find starting indices for sub-volumes in whole volume. Then, append it to a list containing all possible starting indices of
        the sub volumes.
        """

        # (1) Get target shape of sub volumes.
        sx, sy, sz = self.sub_volume_shape
        unfolded = self.volume.unfold(0, sx, sx).unfold(1, sy, sy).unfold(2, sz, sz)  # creates shape: [sub_x_index, sub_y_index, sub_z_index, t_vector, sub_x_volume, sub_y_volume, sub_z_volume]
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


class ConfiguratorGPU():
    def __init__(self, required_space_gpu: float):

        torch.cuda.empty_cache()
        # self.dataset: torch.Tensor = dataset
        self.available_gpus: int = torch.cuda.device_count()
        self.selected_gpu: int = None
        self.required_space_gpu = required_space_gpu

        # self.required_space_gpu: torch.Tensor = dataset.element_size() * dataset.numel()

        # print(self.required_space_gpu / (1024**2))
        # input("-----")
        # self.required_space_gpu: torch.Tensor = torch.prod(dataset.shape) * torch.dtype(dataset).itemsize  # in bytes?
        self.free_space_selected_gpu: torch.Tensor = None

    def select_least_busy_gpu(self):
        # based on percentage free
        free_space_devices = []
        for device in range(self.available_gpus):
            space_total_cuda = torch.cuda.mem_get_info(device)[1]
            space_free_cuda = torch.cuda.mem_get_info(device)[0]
            percentage_free_space = space_free_cuda / space_total_cuda
            free_space_devices.append(percentage_free_space)

        self.selected_gpu = free_space_devices.index(max(free_space_devices))

        torch.cuda.set_device(self.selected_gpu)
        Console.printf("info", f"Selected GPU {self.selected_gpu} -> most free space at moment!")

        self.free_space_selected_gpu = torch.cuda.mem_get_info(self.selected_gpu)[0]  # free memory on GPU [bytes]

    def print_available_gpus(self):
        Console.add_lines("Available GPU(s) and free space on it:")

        for device in range(self.available_gpus):
            space_total_cuda = torch.cuda.mem_get_info(device)[1]
            space_free_cuda = torch.cuda.mem_get_info(device)[0]
            percentage_free_space = space_free_cuda / space_total_cuda
            Console.add_lines(f" GPU {device} [MB]: {space_free_cuda / (1024 ** 2)} / {space_total_cuda / (1024 ** 2)} ({round(percentage_free_space, 2) * 100}%)")

        Console.printf_collected_lines("info")

    def select_gpu(self, gpu_index: int = 0):
        torch.cuda.set_device(gpu_index)
        self.selected_gpu = gpu_index
        Console.printf("info", f"Selected GPU {gpu_index} -> manually selected by the user!")

    def enough_space_available(self) -> bool:

        if self.selected_gpu is not None:
            self.free_space_selected_gpu = torch.cuda.mem_get_info(self.selected_gpu)[0]  # free memory on GPU [bytes]

            if self.required_space_gpu <= self.free_space_selected_gpu:
                Console.printf("info",
                               f"Possible to put whole tensor of size {self.required_space_gpu / (1024 ** 2)} [MB] on GPU. Available: {self.free_space_selected_gpu / (1024 ** 2)} [MB]")
            else:
                Console.printf("info",
                               f"Not possible to put whole tensor of size {self.required_space_gpu / (1024 ** 2)} [MB] on GPU. Available: {self.free_space_selected_gpu / (1024 ** 2)} [MB]")

        else:
            Console.printf("error", f"No GPU is selected. Number of available GPUs: {self.available_gpus}")
            sys.exit()

        return self.required_space_gpu <= self.free_space_selected_gpu


#  + size Dataset
#  + available GPUs
#  + space_free_cuda
#  + space_required_cuda
#
#
# -> set least busy GPU
# -> print available GPUs
# -> select GPU
# -> enough_space_available

##def set_least_busy_GPU():
##    num_devices = torch.cuda.device_count()
##
##    free_space_devices = []
##    Console.add_lines("Free space on:")
##    for device in range(num_devices):
##        torch.cuda.mem_get_info(device)
##
##        space_total_cuda = torch.cuda.mem_get_info(device)[1]
##        space_free_cuda = torch.cuda.mem_get_info(device)[0]
##        percentage_free_space = space_free_cuda / space_total_cuda
##        free_space_devices.append(percentage_free_space)
##        Console.add_lines(f" GPU {device} [MB]: {space_free_cuda / (1024 ** 2)} / {space_total_cuda / (1024 ** 2)}")
##
##    Console.printf_collected_lines("info")
##    index_most_free_space = free_space_devices.index(max(free_space_devices))
##
##    print(f"On GPU {index_most_free_space} most free space at moment!")
##    torch.cuda.set_device(index_most_free_space)


## def check_required_space_GPU(data: torch.Tensor) -> bool:
##     # torch.cuda.empty_cache()
##     space_required_mb_cuda = np.prod(data.shape) * np.dtype(data.dtype).itemsize * 1 / (1024 ** 2)
##     space_free_cuda = torch.cuda.mem_get_info()[0] * 1 / (1024 ** 2)  # free memory on GPU [MB]
##     space_total_cuda = torch.cuda.mem_get_info()[1] * 1 / (1024 ** 2)  # total memory on GPU [MB]
##
##     if space_required_mb_cuda <= space_free_cuda:
##         Console.printf("info", f"Possible to put whole tensor of size '{space_required_mb_cuda}' on GPU. Available: {space_free_cuda} ")
##     else:
##         Console.printf("info", f"Not possible to put whole tensor of size '{space_required_mb_cuda}' on GPU. Available: {space_free_cuda}")
##
##     return space_required_mb_cuda <= space_free_cuda

#class Model:
#    def __init__(self, model): # TODO class Type
#        self.spectral_spatial_model = model
#        self.volume: np.memmap = None
#        self.sub_volume_iterator

    # build() -> arguments: cartesian, non-cartesian
    # cartesian_FT
    # non-cartesian_FT

    # get_sub_volume(i_x,i_y,i_z)
    # get_sub_volume_iterator()


def cartesian_FT(model: SpectralSpatialModel, path_cache: str, file_name_cache: str, auto_gpu: bool = True, custom_batch_size: int = None):
    Console.printf_section("Sampling -> FFT")

    # Get the new shape of the target volume ([x,y,z,1,t] -> [x,y,z,t])
    v = model.volume
    volume_shape = v.shape[0], v.shape[1], v.shape[2], v.shape[4]

    # Create the memmap file for the target volume
    path_cache_file = os.path.join(path_cache, f"{file_name_cache}_sampling_volume.npy")
    volume_cache = np.memmap(path_cache_file, dtype=model.volume.dtype, mode='w+', shape=volume_shape)
    Console.printf("info", f"Create new cache volume with shape {volume_shape} for 'sampling' in path: {path_cache_file}")

    # OPTION 1: Use the GPU
    if torch.cuda.is_available() and auto_gpu:

        # (a) Estimate space required to put whole tensor on GPU (bytes)
        #     Also: Check if required space is available and select the least busy GPU!
        space_required_bytes = np.prod(model.volume.shape) * np.dtype(model.volume.dtype).itemsize
        configurator_gpu = ConfiguratorGPU(required_space_gpu=space_required_bytes)
        configurator_gpu.print_available_gpus()
        configurator_gpu.select_least_busy_gpu()

        # (b) Create custom dataset, it transforms the numpy memmap to a torch tensor
        #     Also, calculate max batch size that fits on GPU.
        dataset = VolumeForFFTDataset(data=model.volume, device="cuda")
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
        dataset = VolumeForFFTDataset(data=model.volume, device="cpu")
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
        volume_cache[:, :, :, indices] = batch.cpu()

        # (4) Also ensure that the cache on the GPU is released. has no effect if it was performed on the CPU!
        torch.cuda.empty_cache()


def non_cartesian_FT():
    pass
# def non_cartesian

# Requirement:
# -> Cartesian FFT
# -> Non cartesian FFT
