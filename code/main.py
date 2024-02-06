import default
from spectral import FID
import numpy as np
import spatial
import file


def zen_of_python():
    """
    The zen of python. Or just 'import this' - 42!

    :return: just wisdom
    """
    print("The Zen of Python:"
          "\n\n Beautiful is better than ugly."
          "\n Explicit is better than implicit."
          "\n Simple is better than complex."
          "\n Complex is better than complicated."
          "\n Flat is better than nested."
          "\n Sparse is better than dense."
          "\n Readability counts."
          "\n Special cases aren't special enough to break the rules."
          "\n Although practicality beats purity."
          "\n Errors should never pass silently."
          "\n Unless explicitly silenced."
          "\n In the face of ambiguity, refuse the temptation to guess."
          "\n There should be one-- and preferably only one --obvious way to do it."
          "\n Although that way may not be obvious at first unless you're Dutch."
          "\n Now is better than never."
          "\n Although never is often better than right now."
          "\n If the implementation is hard to explain, it's a bad idea."
          "\n If the implementation is easy to explain, it may be a good idea."
          "\n Namespaces are one honking great idea -- let's do more of those!"
          "\n\n -Tim Peters"
          )



if __name__ == '__main__':
    fid1 = FID(signal=np.array([1, 1, 1, 1, 1]), time=np.array([1, 2, 3, 4, 5]), name="Creatine")
    print("FID 1 name:")
    print(fid1.name)
    print("FID 1 signal and time shape:")
    print(fid1.signal.shape)
    print(fid1.time.shape)
    print("FID 1 signal:")
    print(fid1.signal)

    fid2 = FID(signal=np.array([2, 1, 2, 1, 2]), time=np.array([1, 2, 3, 4, 5]), name="Creatine")
    print("FID 2 name:")
    print(fid2.name)
    print("FID 2 signal and time shape:")
    print(fid2.signal.shape)
    print(fid2.time.shape)
    print("FID 2 signal:")
    print(fid2.signal)


    fid3 = fid1+fid2
    print("FID 3 name:")
    print(fid3.name)
    print("FID 3 signal and time shape:")
    print(fid3.signal.shape)
    print(fid3.time.shape)
    print("FID 3 signal:")
    print(fid3.signal)

    fid4 = fid1+fid3


    #print(fid3)
    #print(fid3.signal)
    #print(fid3.time)
    print("FID 1 name:")
    print(fid1.name)
    print("FID 2 name:")
    print(fid2.name)
    print("FID 3 name:")
    print(fid3.name)
    print("FID 4 name:")
    print(fid4.name)
    print("FID 4 signal:")
    print(fid4.signal)

    print("First column of FID 4:")
    print(fid4.signal[:,0])

    input()


    #zen_of_python()

    # Load defined paths in the configurator
    configurator = file.Configurator(path_folder="/home/mschuster/projects/Synthetic_MRSI/code/config/",
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

    ## Plotting all FIDs
    #fids_dict: dict = {}
    #for fid in metabolites.fids:
    #    fids_dict[fid.name] = fid.signal
    #plot_FIDs(amplitude=fids_dict, time=time_vector, save_to_file=True)

    # Simulate map including desired FIDs according to mask with random scaling
    load_cache = False
    mrsi_data = None
    if not load_cache:
        path_cache = configurator.data["path"]["cache"]
        simulator = spatial.Simulator(path_cache=path_cache)

        mrsi_data = simulator.mrsi_data(mask=metabolic_mask.data,
                            concentration=random_concentration,
                            values_to_add=metabolites_fid_data[0:2],  # TODO: Try to add all!!!,
                            data_type=np.complex64)
    else:
        mrsi_data = np.memmap('/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/mschuster/SimulationMRSI/cache//simulated_metabolite_map.npy', dtype='complex64', mode='r')







