import sys

import default
from spectral_spatial_simulation import FID
import numpy as np
import spectral_spatial_simulation
import file
from printer import Console
from display import plot_FID, plot_FIDs
import sampling
import pint
from spatial_metabolic_distribution import MetabolicPropertyMap
from tools import SubVolumeDataset
import dask


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
    # zen_of_python()

    # Initialize the UnitRegistry
    u = pint.UnitRegistry()

    # print(np.array([1, 2, 3, 4]) * u.kg / u.ms)
    # input("---------------")

    # Load defined paths in the configurator
    Console.printf_section("Load and prepare data")
    configurator = file.Configurator(path_folder="/home/mschuster/projects/Synthetic_MRSI/code/config/",
                                     file_name="config_25022024.json")
    # file_name="config_04012024.json")

    configurator.load()

    configurator.load()
    configurator.print_formatted()

    metabolic_mask = file.Mask.load(configurator=configurator,
                                    mask_name="metabolites")

    # set the unit with pint
    # metabolic_mask.data *= u.dimensionless

    # Create a metabolic property map for one metabolite
    concentration_random_map = np.random.uniform(low=0.0, high=1.0, size=metabolic_mask.data.shape)
    t2_random_map = np.random.uniform(low=0.0, high=1.0, size=metabolic_mask.data.shape)
    t1_random_map = np.random.uniform(low=0.0, high=1.0, size=metabolic_mask.data.shape)
    main_volume_shape = np.random.uniform(low=0.0, high=1.0, size=metabolic_mask.data.shape)
    sub_volume_shape = np.random.uniform(low=0.0, high=1.0, size=metabolic_mask.data.shape)

    metabolic_property_map_1 = MetabolicPropertyMap(chemical_compound_name="NAA",
                                                    concentration=concentration_random_map,
                                                    t1=t1_random_map,
                                                    t2=t2_random_map,
                                                    blocks_shape_xyz=(20, 22, 24))

    metabolic_property_map_2 = MetabolicPropertyMap(chemical_compound_name="Creatine",
                                                    concentration=concentration_random_map,
                                                    t1=t1_random_map,
                                                    t2=t2_random_map,
                                                    blocks_shape_xyz=(20, 22, 24))

    #metabolic_property_map = metabolic_property_map_1 * metabolic_property_map_2

    # TODO: Do not delete!
    # To get all the chunks (dask.array) in a list
    # metabolic_property_map.t2

    Console.start_timer()

######print(metabolic_property_map_1[5]._t1.get_global_index_in_sub_volume(block_number=5, indices_sub_volume=(2, 2, 3)))
######print(metabolic_property_map_1[5]._t1.blocks[0][2,2,3].compute())
######print(t1_random_map[2,2,3]) # x -> z; y -> y; z -> x

######input("=====================**")

    print(metabolic_property_map_1.chemical_compound_name)
    for i, m in enumerate(metabolic_property_map_1):

        block_number = i
        x,y,z = m.t1.get_global_index_in_sub_volume(main_volume_shape=(112, 128, 80), block_number=i, indices_sub_volume=(0,0,0))
        print(m.t1.blocks[0][0,0,0].compute()) # block 0, just one block here when using __iter__
        print(t1_random_map[x,y,z]) # coordinates in big block!


    #objects = dask.compute(*metabolic_property_maps)
    Console.stop_timer()

    #Console.start_timer()
    #for i, mmap in enumerate(metabolicPropertyMap):
    #    #print(f"{i}: {type(mmap.t2.blocks[0].compute())}")
    #    delayed_result = mmap.t2.compute()
    #    print(i)
    #
    #Console.stop_timer()

    #print(len(metabolicPropertyMap))
    #metabolicPropertyMap[0]

    #subVolumeDataset = SubVolumeDataset(main_volume=concentration_random_map, sub_volume_shape=(20, 20, 20))

    #print(subVolumeDataset[5])

    sys.exit()
    input("~~~~~~~~~~~~+STOP+~~~~~~~~~~~~")

    # Create random distribution with values in the range [0.0, 1.0]

    # Load FID of metabolites
    metabolites = file.FID(configurator=configurator)  # ,
    # concentrations=np.asarray([1, 1, 4, 10, 1, 2.5, 1, 6, 12, 4, 1]),
    # t2_values=np.asarray([170, 131, 121, 105, 170, 105, 131, 170, 170, 121, 131]) / 1000)
    metabolites.load(fid_name="metabolites",
                     signal_data_type=np.complex64)

    loaded_fid = metabolites.loaded_fid
    # set the unit with pint
    # loaded_fid.time *= u.ms
    # loaded_fid.signal *= u.dimensionless  # TODO

    print(loaded_fid)

    # plot_FID(signal=fid.signal[0], time=fid.time)
    # print(metabolites.loaded_fid.get_partially_FID([1,5]))

    # plot_FID(signal=loaded_fid.signal, time=loaded_fid.time)

    # Simulate volume with desired FID
    # path_cache = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/mschuster/SimulationMRSI/cache/'

    # path_cache = "/home/mschuster/projects/Synthetic_MRSI/cache/" # TODO: use configurator, or at least config file for it!!!! DOnt define it manually!
    configurator.load()  # to load data from json file
    path_cache = configurator.data['path']['cache']  # TODO: maybe input configurator to spectral and sampling model?
    spectral_model = spectral_spatial_simulation.Model(path_cache=path_cache, file_name_cache="25022024")
    # spectral_model.add_fid(metabolites.loaded_fid)
    # spectral_model.add_mask(metabolic_mask.data)
    # spectral_model.add_fid_scaling_map(random_scaling)
    # spectral_model.build()

    # loaded_fid.sum_all_signals()

    random_scaling_maps_list = []
    for name in loaded_fid.name:
        random_scaling = np.random.uniform(low=0.0, high=1.0, size=metabolic_mask.data.shape)
        scaling_map = MetabolicScalingMap(name=name, map=random_scaling, unit=None)
        random_scaling_maps_list.append(scaling_map)

    spectral_model.add_fid(loaded_fid)
    spectral_model.add_mask(metabolic_mask.data)
    spectral_model.add_fid_scaling_maps(fid_scaling_maps=random_scaling_maps_list)  # add ScalingMap (for each metabolite)
    spectral_model.build()

    sampling_model = sampling.Model(spectral_model=spectral_model, sub_volume_shape=(14, 16, 10))  # for main volume 112, 128, 80
    sampling_model.cartesian_FT(file_name_cache="25022024")
    sampling_model.create_sub_volume_iterative()
    # TODO
    # sampling.cartesian_FT(spectral_model, auto_gpu=True, path_cache=path_cache, file_name_cache="25022024", custom_batch_size=200)
