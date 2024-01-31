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

    def metabolite_map(self, #TODO -> mrsi_data -> add_dimensions
                       mask: np.memmap | np.ndarray,
                       concentration: np.ndarray,
                       values_to_add: np.ndarray,
                       values_to_add_name: str | list[str] = None,
                       data_type: np.dtype = np.complex64):
        # TODO: Should be FIDs or Spectra?
        # TODO: values_to_add -> can be an array containing a scalar for each voxel, or also possible: a vector (e.g., FID)
        # TODO:   --> rows should be the dimension (metabolite1, metabolite2,...) and columns the data vector or e.g., metabolite1
        # TODO:   --> does not consider the time vector

        # Set path where memmap file should be created and create memmap
        path_cache_file = self.path_cache + '/simulated_metabolite_map.npy'

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

        metabolite_map = np.memmap(path_cache_file, dtype=data_type, mode='w+', shape=shape)  # TODO: Do I also have to define the shape first?!!!

        # metabolite_map.size returns the number of elements in the memmap array.
        # metabolite_map.itemsize returns the size (in bytes) of each element
        space_required_mb = metabolite_map.size * metabolite_map.itemsize * 1 / (1024 * 1024)

        # Check if the simulation does not exceed a certain size
        # Console.check_condition(space_required_mb < 30000, ask_continue=True)
        Console.ask_user(f"Estimated required space [MB]: {space_required_mb}")

        # Assign desired values to each pixel (can be a signal or just a scalar)
        indices = np.where(mask != 0)

        # TODO TODO TODO TODO ===> memmap x,y,z, !!!!, : -> instead of 0 change for respective metabolite?

        # The v could be a value or also a vector of values
        if values_to_add.ndim == 1:
            iterator_signals = range(1)
            values_to_add = np.expand_dims(values_to_add, axis=0)  # to get shape e.g., (1, 1536) instead of only 1536
            print(values_to_add.shape)
        else:
            iterator_signals = range(len(values_to_add))

        Console.start_timer()

        for v in iterator_signals:
            for y, x, z in tqdm(zip(*indices), total=len(indices[0]), desc="Assigning values to volume"):
                metabolite_map[x, y, z, v, :] = ((mask[x, y, z] * concentration[x, y, z] * values_to_add[v])
                                                 .astype(data_type))

        Console.stop_timer()

        print(metabolite_map.shape)


if __name__ == "__main__":
    configurator = file.Configurator(path_folder="/home/mschuster/projects/SimulationMRSI/config",
                                     file_name="config_04012024.json")
    configurator.load()
    configurator.print_formatted()

    metabolic_mask = file.Mask.load(configurator=configurator,
                                    mask_name="metabolites")

    # Create random distribution with values in the range [0.0, 1.0]
    random_concentration = np.random.uniform(low=0.0, high=1.0, size=metabolic_mask.data.shape)

    # Load FID of metabolites
    metabolites = file.FIDs(configurator=configurator,
                            concentrations=np.asarray([1, 1, 4, 10, 1, 2.5, 1, 6, 12, 4, 1]),
                            t2_values=np.asarray([170, 131, 121, 105, 170, 105, 131, 170, 170, 121, 131]) / 1000)
    metabolites.load(fid_name="metabolites",
                     signal_data_type=np.complex64)

    # Extract from the FID objects the signal data (np.ndarray), merge it to a new numpy array object.
    fids_list: list = []

    time_vector = metabolites.fids[0].time
    [fids_list.append(fid.signal) for fid in metabolites.fids]
    metabolites_fid_data = np.asarray(fids_list)

    # Plotting all FIDs
    fids_dict: dict = {}
    for fid in metabolites.fids:
        fids_dict[fid.name] = fid.signal
    plot_FIDs(amplitude=fids_dict, time=time_vector, save_to_file=True)

    # Simulate map including desired FIDs according to mask with random scaling
    path_cache = configurator.data["path"]["cache"]
    simulator = Simulator(path_cache=path_cache)
    simulator.metabolite_map(mask=metabolic_mask.data,
                             concentration=random_concentration,
                             values_to_add=metabolites_fid_data[0],  # TODO: Try to add all!!!,
                             data_type=np.complex64)