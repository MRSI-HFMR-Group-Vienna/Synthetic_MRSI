from printer import Console
import dask.array as da
import numpy as np
import torch
import sys

import dask


class SubVolumeDataset:

    def __init__(self, main_volume: np.ndarray | np.memmap | da.core.Array, blocks_shape_xyz: tuple): #, main_volume_shape: tuple = None):

        # Case 1: If the main volume is already a task array, asarray() is called. The number of block will just
        #         be one. Necessary for creating sub-volumes recursively in higher classes without the need to
        #         call compute()
        # Case 2: If the main volume is an numpy array, it will transform to a dask array and also create chunks
        #         (=thus sub-blocks).
        self.blocks_shape_xyz = blocks_shape_xyz

        if main_volume is None:
            Console.printf("error", f"Cannot create a SubVolumeDataset since the volume is None. Abort program!")
            sys.exit()
        elif isinstance(main_volume, da.core.Array):
            self.dask_array: da.core.Array = main_volume #da.asarray(main_volume)
        else:
            self.dask_array: da.core.Array = da.from_array(main_volume, chunks=self.blocks_shape_xyz)

        # Create the blocks and flatten it to 1D array, thus get kind of a list of all blocks!
        # Using standard implementation: "C-style", it refers to the memory layout convention used by the C programming language.
        self.blocks = self.dask_array.blocks.ravel()
        self.number_of_blocks = len(self.blocks)
        self.block_iterator: int = 0  # initial value, start at block 0

    # TODO: necessary? def get_block_by_indices(self, x: int, y: int, z: int): Thus, by indices in main volume!

    def get_block_by_number(self, block_number: int):
        """
        How the next block is selected by index:
        By increasing the index, the next lock in z direction is selected until the end is reached. Then, we go to the next
        line of blocks in y-direction and repeat the previous procedure. After we reached the end of the y- and x-direction,
        (we iterated through one "x-y-plane"), we go one step in the x-direction. Then, the previous procedure is repeated.

        The axes defining the coordinates of the sub-volume (chunks) in the main volume are defined as following:

               origin
                 +----------> (+z)
                /|
              /  |
            /    |
        (+x)     |
                 v (+y)
        """

        return self.blocks[block_number]

    def get_global_index_in_sub_volume(self, main_volume_shape: tuple, block_number: int, indices_sub_volume: tuple) -> tuple: # TODO: main_volume_shape can alo be defined here!
        """
        Using the indices in the sub-volume (to access the desired value) to get the corresponding indices in the main volume.
        Necessary to conclude at which place the sub-volume, but much more the values in the sub-volume have been originally
        in the main volume! Let's call the indices in the sub-volume local indices and the corresponding indices in the main
        volume global indices. Thus, the function is -> Sub volume X (=block X) indices --> Main volume indices

        The axes of the sub volume are defined as following:

               origin
                 +----------> (+z)
                /|
              /  |
            /    |
        (+x)     |
                 v (+y)
        """

        # (1) Extract the indices to the associated axes
        in_block_x, in_block_y, in_block_z = indices_sub_volume

        # (2) Get the size of the block in the respective dimension.
        #     * The chunk size does not need to be isotropic!
        #     * Handling remainder blocks: thus, use 'ceil' because an incomplete (=remainder block) still counts as 1 block!
        (number_of_blocks_x,
         number_of_blocks_y,
         number_of_blocks_z) = np.ceil(np.array(main_volume_shape) / np.array(self.blocks_shape_xyz)).astype(int)

        #(number_of_blocks_x,
        # number_of_blocks_y,
        # number_of_blocks_z) = np.ceil(np.array(self.dask_array.shape) / np.array(self.dask_array.chunksize)).astype(int)

        #print(f"number of blocks in x: {number_of_blocks_x} | y: {number_of_blocks_y} | z: {number_of_blocks_z}")

        # (3) Defines the steps of shift (indices of each block in the collection of all blocks).
        #     The block size is not yet taken into account.
        number_of_shifts_x = block_number // (number_of_blocks_y * number_of_blocks_z)  # // -> Floor division
        number_of_shifts_y = (block_number // number_of_blocks_z) % number_of_blocks_y  #
        number_of_shifts_z = block_number % number_of_blocks_z

        print(f"number of shifts x: {number_of_shifts_x}  y: {number_of_shifts_y}  z: {number_of_shifts_z}")

        # (4) Defines the shift size. Now, take the block size into account.
        block_size_x = self.blocks_shape_xyz[0] #dask_array.chunksize[0]
        block_size_y = self.blocks_shape_xyz[1] #dask_array.chunksize[1]
        block_size_z = self.blocks_shape_xyz[2] #dask_array.chunksize[2]

        # (5) Calculate absolute coordinates from block in the main (whole) volume!
        absolute_x = number_of_shifts_x * block_size_x + in_block_x
        absolute_y = number_of_shifts_y * block_size_y + in_block_y
        absolute_z = number_of_shifts_z * block_size_z + in_block_z

        # Return the global indices in the main volume. Thus, where the value indexed in the sub-volume X has been
        # before in the global volume.
        return absolute_x, absolute_y, absolute_z

    def compute(self):
        """
        Short-cut of dask inbuilt function. It is a special case and only valid if only one block is generated. This use case is give
        if a subset is created in MetabolicPropertyMap. Thus, not necessary to call .blocks[0].compute(). Instead directly .compute()
        can be called.
        """
        if len(self.blocks) is 1:
            return self.blocks[0].compute()
        else:
            Console.printf("error", f"Only possible to call short-cut 'compute' is length of blocks is 1. However, it is {len(self.blocks)}")

    def __iter__(self):
        """
        Return the object itself as the iterator
        """
        return self

    def __next__(self):
        """
        To implement it in a for each loop, for example, or to use next() on the object!
        """
        if self.block_iterator >= self.number_of_blocks:
            raise StopIteration  # Stop iterator when the last block was reached!
        else:
            self.block_iterator += 1
            return self.blocks[self.block_iterator - 1]

    def __mul__(self, other):
        """
        To be able to multiply two volumes.
        """
        dask_array = self.dask_array * other.dask_array
        return SubVolumeDataset(main_volume=dask_array, blocks_shape_xyz=self.blocks_shape_xyz)

    def __len__(self):
        """
        To get the number of sub-volumes.
        """
        return self.number_of_blocks

    def __getitem__(self, block_number):
        """
        To get the block by the index. Note the defined coordinate system above to understand how index blocks!
        """
        return self.blocks[block_number]


# class SubVolumeDataset:
#    """
#    For getting sub-volumes from the main volume. The sub volumes do not need to be isotropic!
#    Example usage 1: Iterate over each sub volume
#    Example usage 2: Get sub-volume by index (index_x, index_y, index_z) in main volume
#    Example usage 2: Get batch of sub-volumes (e.g., [3, 50, 50, 50, 30]). Useful, if put on GPU and limited space is available there.
#                     Thus, put batch on GPU, process it and then put back to CPU.
#    """
#
#    def __init__(self, volume: np.ndarray | np.memmap, sub_volume_shape: tuple):
#        self.volume = volume
#        self.sub_volume_shape = sub_volume_shape
#        self.indices = self._initialize_indices()
#
#    def get_by_index(self, i_x: int, i_y: int, i_z: int) -> np.ndarray:
#        """
#        Get sub-volume by indices in the main volume. Thus, if the main volume, e.g., is divided into eight sub-volumes, maximum
#        indices are (7,7,7).
#
#        :param i_x: sub-volume index in the main volume in x direction
#        :param i_y: sub-volume index in the main volume in y direction
#        :param i_z: sub-volume index in the main volume in z direction
#
#        :return: sub-volume as PyTorch Tensor
#        """
#        sx, sy, sz = self.sub_volume_shape
#        sub_volume = self.volume[i_x:i_x + sx, i_y:i_y + sy, i_z:i_z + sz, :]
#        # sub_volume = torch.from_numpy(sub_volume).detach()
#        return sub_volume
#
#    def _initialize_indices(self):
#        """
#        Find starting indices for sub-volumes in whole volume. Then, append it to a list containing all possible starting indices of
#        the sub volumes.
#        """
#
#        # (1) Get target shape of sub volumes.
#        sx, sy, sz = self.sub_volume_shape
#        unfolded = self.volume.unfold(0, sx, sx).unfold(1, sy, sy).unfold(2, sz,
#                                                                          sz)  # creates shape: [sub_x_index, sub_y_index, sub_z_index, t_vector, sub_x_volume, sub_y_volume, sub_z_volume]
#        # e.g., [0,0,0,:,:,:,:] -> get x,y,z with t of index 0 (=index of sub volume)
#
#        # Get starting indices of sub-volumes (e.g., x=0,y=0,z=50). Create it for each sub-volume and append to a list as tuple.
#        indices = []
#
#        u = 0
#        # Get each possible starting index for each sub-volume in the whole volume (e.g., x=0,y=0,z=50)
#        for i_x in range(unfolded.shape[0]):
#            for i_y in range(unfolded.shape[1]):
#                for i_z in range(unfolded.shape[2]):
#                    # print(f"indices new = {x,y,z}")
#                    indices.append((i_x * sx, i_y * sy, i_z * sz))
#
#        return indices
#
#    def __len__(self):
#        """
#        To get the number of sub-volumes.
#        """
#        return len(self.indices)
#
#    def __getitem__(self, index):
#        """
#        Cut sub volumes out of the whole volume using starting indices and the size of sub volumes.
#        """
#        i_x, i_y, i_z = self.indices[index]
#        sx, sy, sz = self.sub_volume_shape
#        sub_volume = self.volume[i_x:i_x + sx, i_y:i_y + sy, i_z:i_z + sz, :]  # start and end index to cut out sub-volume of whole volume
#        sub_volume = torch.from_numpy(sub_volume).detach()
#        return sub_volume


# class Map():
#    """
#    Usage for creating maps. For example B0 map, B1+ map, B1- map, FIF scaling map and so on.
#    It has the option to perform the following to combine two instances:
#     => let instance 1 be B0 map, and instance 2 be FID scaling map,
#        then instance 1 * instance 2 performs a point-wise multiplication
#    """
#
#    def __init__(self):
#        self.volume: np.ndarray
#        self.name: str
#        self.unit: str
#        pass
#
#    def __mul__(self, other):
#        # TODO: Check if size of self, other fits!
#        pass
#
#    def __iter__(self):
#        pass
#
#    def __next__(self):
#        pass


class ConfiguratorGPU():
    """
    TODO!!! Describe it
    """

    def __init__(self, required_space_gpu: np.ndarray):

        torch.cuda.empty_cache()
        self.available_gpus: int = torch.cuda.device_count()
        self.selected_gpu: int = None
        self.required_space_gpu = required_space_gpu
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

        # TODO: Check return type!
        return self.required_space_gpu <= self.free_space_selected_gpu


# put estimate space here

class SpaceEstimator:
    """
    To calculate the required space on the disk for a numpy array of a given shape and data type. Usefully, if a big array
    should be created and to check easily if it does not exceed a certain required space.
    """

    @staticmethod
    def for_numpy(data_shape: tuple, data_type: np.dtype, unit: str = "MB") -> np.ndarray:
        """
        For estimating the required space of a numpy array with a desired shape and data type.

        :param data_shape: desired shape of the numpy array
        :param data_type: desired data type of the numpy array (e.g., np.int64)
        :param unit: desired unit the numpy array
        :return: required space on disk as a numpy array (with defined unit: [bytes], [KB], [MB], [GB]). Standard unit is [MB].
        """

        # np.prod...  To get total numpy array size
        # itemsize... bytes required by each element in array
        space_required_bytes = np.prod(data_shape) * data_type.itemsize

        if unit is "byte":
            return space_required_bytes

        elif unit is "KB":
            return space_required_bytes * 1 / 1024

        elif unit is "MB":
            return space_required_bytes * 1 / (1024 ** 2)

        elif unit is "GB":
            return space_required_bytes * 1 / (1024 ** 3)

    @staticmethod
    def for_torch(torch_array: torch.Tensor, unit: str = "MB"):
        """
        TODO: Implement it.
        """
        raise NotImplementedError("This method is not yet implemented")
