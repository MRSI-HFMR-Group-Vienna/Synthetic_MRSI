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

class CustomDataset(Dataset):
    """
    Does the Dataset automaticall reduces the shape?
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

        Console.printf_collected_lines("info")
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


def cartesian_FT(model: SpectralSpatialModel, path_cache: str, file_name_cache: str, auto_gpu: bool = True, custom_batch_size: int = 0):
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
        dataset = CustomDataset(data=model.volume, device="cuda")
        number_of_batches = np.ceil(configurator_gpu.required_space_gpu / configurator_gpu.free_space_selected_gpu)
        size_of_batches = int(np.floor(len(dataset) / number_of_batches))

        # (c) Create the Dataloader with the Custom Dataset. Based on the batch size, the DataLoader creates a
        #     number n batches.
        dataloader = DataLoader(dataset=dataset,
                                batch_size=size_of_batches,
                                shuffle=False,
                                drop_last=False)

    # OPTION 2: Use the CPU
    else:
        dataset = CustomDataset(data=model.volume, device="cpu")
        dataloader = DataLoader(dataset=dataset,
                                batch_size=len(dataset),
                                shuffle=False,
                                drop_last=False)

    # The following fits for GPU and CPU:
    indent = " " * 22
    # (1) Iterate through each batch. If GPU is chosen, then the respective batch is put on the GPU (see CustomDataset)
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
