import default
from spectral_spatial_simulation import FID
import numpy as np
import spectral_spatial_simulation
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
    fid1 = FID(signal=np.array([1, 1, 1, 1, 1]), time=np.array([1, 2, 3, 4, 5]), name=["CreatineX"])
    print("FID 1 name:")
    print(fid1.name)
    print("FID 1 signal and time shape:")
    #print(fid1.signal.shape)
    #print(fid1.time.shape)
    #print("FID 1 signal:")
    #print(fid1.signal)

    fid2 = FID(signal=np.array([2, 1, 2, 1, 2]), time=np.array([1, 2, 3, 4, 5]), name=["NAA"])
    print("FID 2 name:")
    print(fid2.name)
    #print("FID 2 signal and time shape:")
    #print(fid2.signal.shape)
    #print(fid2.time.shape)
    #print("FID 2 signal:")
    #print(fid2.signal)


    fid3 = fid1+fid2
    print("FID 3 name:")
    print(fid3.name)
    #print("FID 3 signal and time shape:")
    #print(fid3.signal.shape)
    #print(fid3.time.shape)
    #print("FID 3 signal:")
    #print(fid3.signal)

    fid4 = fid1+fid3 # str + list
    print("FID 4 name:")
    print(fid4.name)

    fid5 = fid3+fid1
    print("FID 5 name: (should be: ['CreatineX', 'NAA', CreatineX])")
    print(fid5.name)

    fid6 = fid4+fid5
    print("FID 6 name: ['CreatineX', 'CreatineX', 'NAA'] + ['CreatineX', 'NAA', 'CreatineX']")
    print(fid6.name)
    print(fid6.signal)



    input("-----------------")
    #zen_of_python()

    # Load defined paths in the configurator
    configurator = file.Configurator(path_folder="/home/mschuster/projects/Synthetic_MRSI/code/config/",
                                     file_name="config_04012024.json")
    configurator.load()
    configurator.print_formatted()

    metabolic_mask = file.Mask.load(configurator=configurator,
                                    mask_name="metabolites")


    # Create random distribution with values in the range [0.0, 1.0]
    random_scaling = np.random.uniform(low=0.0, high=1.0, size=metabolic_mask.data.shape)

    # Load FID of metabolites
    metabolites = file.FID(configurator=configurator,
                           concentrations=np.asarray([1, 1, 4, 10, 1, 2.5, 1, 6, 12, 4, 1]),
                           t2_values=np.asarray([170, 131, 121, 105, 170, 105, 131, 170, 170, 121, 131]) / 1000)
    metabolites.load(fid_name="metabolites",
                     signal_data_type=np.complex64)


    fid = metabolites.loaded_fid
    # Extract from the FID objects the signal data (np.ndarray), merge it to a new numpy array object.
    #fids_list: list = []

    #time_vector = metabolites.loaded_fid[0].time
    #[fids_list.append(fid.signal) for fid in metabolites.loaded_fid]
    #metabolites_fid_data = np.asarray(fids_list)

    ## Plotting all FID
    #fids_dict: dict = {}
    #for fid in metabolites.loaded_fid:
    #    fids_dict[fid.name] = fid.signal
    #plot_FIDs(amplitude=fids_dict, time=time_vector, save_to_file=True)

    print(metabolites.loaded_fid.get_partially_FID([1,5]))

    input("------------")
    # Simulate volume with desired FID
    path_cache = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/mschuster/SimulationMRSI/cache/'
    spectral_model = spectral_spatial_simulation.Model(path_cache=path_cache)
    spectral_model.add_fid(metabolites.loaded_fid)
    spectral_model.add_mask(metabolic_mask.data)
    spectral_model.add_fid_scaling_map(random_scaling)
    spectral_model.build()

