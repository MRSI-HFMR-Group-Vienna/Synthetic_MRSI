import default

from printer import Console
from spectral import FID
from display import plot_FIDs
from tqdm import tqdm
import numpy as np
import file
import sys
import os


class Model:
    # TODO
    def __int__(self):
        self.mask: dict = {"metabolites": np.array([]),
                           "lipids": np.array([])}

        self.map: dict = {"metabolites": dict,
                          "lipids": dict}

    def __init__(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def create_masks(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def add_metabolites_mask(self, mask: np.ndarray) -> None:
        """
        Add a metabolite mask to the Model. It should be binary.

        :param mask: numpy array containing the binary values of the mask.
        :return: Nothing
        """
        self.mask["metabolites"] = mask

    def add_lipids_mask(self, mask: np.ndarray) -> None:
        """
        Add a lipids mask to the Model. It should be binary.

        :param mask: numpy array containing the binary values of the mask.
        :return: Nothing
        """
        self.mask["lipids"] = mask

    def add_metabolite_map(self, name: str, metabolite_map: np.ndarray):
        """
        Add a metabolite map by name to the model. It contains the

        :param name:
        :param metabolite_map:
        :return:
        """
        self.map["metabolites"][name] = map
        Console.printf("success", f"Added map of metabolite {metabolite_map} to the model.")

        # TODO
        # IDEA: Add to a dictionary with key and value
        raise NotImplementedError("This method is not yet implemented")

    def add_T1_image(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def add_subject_variability(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def add_pathological_alterations(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")


class MetabolicAtlas:
    # TODO
    def __init__(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def transform_to_T1(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def load(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")


class Simulator:
    # TODO
    def __init__(self, path_cache: str):
        if os.path.exists(path_cache):
            self.path_cache = path_cache
        else:
            Console.printf("error", f"Terminating the program. Path does not exist: {path_cache}")
            sys.exit()

    def mrsi_data(self,
                       mask: np.memmap | np.ndarray,
                       concentration: np.ndarray,
                       values_to_add: np.ndarray,
                       values_to_add_name: str | list[str] = None,
                       data_type: np.dtype = np.complex64):
        # TODO: Should be FIDs or Spectra?
        # TODO: values_to_add -> can be an array containing a scalar for each voxel, or also possible: a vector (e.g., FID)
        # TODO:   --> rows should be the dimension (metabolite1, metabolite2,...) and columns the data vector or e.g., metabolite1
        # TODO:   --> does not consider the time vector
        Console.printf_section("MRSI spatial simulation")

        # (a) Set path where memmap file should be created and create memmap
        path_cache_file = self.path_cache + '/simulated_metabolite_map.npy'
        Console.printf("info", f"Using cache path: {path_cache_file}")

        # (b) Get required shape depending on the dimensionality
        if values_to_add.ndim == 0:
            Console.printf("error", f"The values to add has dimension {0}. Terminate program!")
            sys.exit()
        elif values_to_add.ndim == 1:  # For one dimension
            shape = (mask.shape[0],
                     mask.shape[1],
                     mask.shape[2],
                     1,
                     values_to_add.shape[0])
        elif values_to_add.ndim == 2:  # For two dimensions
            shape = (mask.shape[0],
                     mask.shape[1],
                     mask.shape[2],
                     values_to_add.shape[0],
                     values_to_add.shape[1])
        else:
            Console.printf("error", f"The values to add has dimension {values_to_add.ndim}. Not yet supported. Terminate program!")
            sys.exit()

        Console.printf("info", f"Using mask shape: {mask.shape}")

        # (c) Set path for storing the memmap on hard drive
        metabolite_map = np.memmap(path_cache_file, dtype=data_type, mode='w+', shape=shape)

        # (d) Estimate space required on hard drive
        # metabolite_map.size returns the number of elements in the memmap array.
        # metabolite_map.itemsize returns the size (in bytes) of each element
        space_required_mb = metabolite_map.size * metabolite_map.itemsize * 1 / (1024 * 1024)

        # (e) Ask the user for agreement of the required space
        # Check if the simulation does not exceed a certain size
        # Console.check_condition(space_required_mb < 30000, ask_continue=True)
        Console.ask_user(f"Estimated required space [MB]: {space_required_mb}")

        # (f) Find indices where mask is not 0
        indices = np.where(mask != 0)

        # (g) Check if just one vector or multiple vectors are added to one voxel
        #     -> If just one vector:     dimension (1, vector or scalar) is added
        #     -> if two or more vectors: dimension (number_of_vectors, vectors) are added
        #
        #     This is for maintaining the following shape: (x, y, z, v, s) with x,y,z as the
        #     voxel position and v as the number of vector and s the signal
        if values_to_add.ndim == 1:
            iterator_signals = range(1)
            values_to_add = np.expand_dims(values_to_add, axis=0)  # to get shape e.g., (1, 1536) instead of only 1536
            #print(values_to_add.shape)
        else:
            iterator_signals = range(len(values_to_add))

        #Console.start_timer()

        # (h) Assign desired values to the voxels
        for v in iterator_signals:
            for y, x, z in tqdm(zip(*indices), total=len(indices[0]), desc="Assigning values to volume"):
                metabolite_map[x, y, z, v, :] = ((mask[x, y, z] * concentration[x, y, z] * values_to_add[v])
                                                 .astype(data_type))

        Console.printf("success", f"Created volume of shape : {metabolite_map.shape}")

        #Console.stop_timer()

        return metabolite_map

        #print(metabolite_map.shape)


if __name__ == "__main__":
    pass