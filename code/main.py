import default
from spectral_spatial_simulation import FID
import numpy as np
import spectral_spatial_simulation
import file
from printer import Console
from display import plot_FID, plot_FIDs
import sampling


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

    #zen_of_python()

    # Load defined paths in the configurator
    Console.printf_section("Load and prepare data")
    configurator = file.Configurator(path_folder="/home/mschuster/projects/Synthetic_MRSI/code/config/",
                                     file_name="config_25022024.json")
                                     #file_name="config_04012024.json")
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


    loaded_fid = metabolites.loaded_fid
    print(loaded_fid)
    #plot_FID(signal=fid.signal[0], time=fid.time)
    #print(metabolites.loaded_fid.get_partially_FID([1,5]))

    #plot_FID(signal=loaded_fid.signal, time=loaded_fid.time)


    # Simulate volume with desired FID
    #path_cache = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/mschuster/SimulationMRSI/cache/'


    path_cache = "/home/mschuster/projects/Synthetic_MRSI/cache/" # TODO: use configurator, or at least config file for it!!!! DOnt define it manually!
    spectral_model = spectral_spatial_simulation.Model(path_cache=path_cache, file_name_cache="25022024")
    #spectral_model.add_fid(metabolites.loaded_fid)
    #spectral_model.add_mask(metabolic_mask.data)
    #spectral_model.add_fid_scaling_map(random_scaling)
    #spectral_model.build()


    loaded_fid.sum_all_signals()

    spectral_model.add_fid(loaded_fid)
    spectral_model.add_mask(metabolic_mask.data)
    spectral_model.add_fid_scaling_map(random_scaling) # add ScalingMap (for each metabolite)
    spectral_model.build()

    sampling.cartesian_FT(spectral_model, auto_gpu=True, path_cache=path_cache, file_name_cache="25022024", custom_batch_size=200)
