import os.path

import torch

import default

from spatial_metabolic_distribution import MetabolicPropertyMap
from dask.distributed import Client, LocalCluster
from distributed.diagnostics import MemorySampler
from dask.diagnostics import ProgressBar
from sklearn.utils import gen_batches
import spectral_spatial_simulation
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from printer import Console
import numpy as np
import dask
import pint
import file
import sys

import seaborn as sns
import pandas as pd

from pycallgraph import Config
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput


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


def main_entry():


    # zen_of_python()

    # Initialize the UnitRegistry
    u = pint.UnitRegistry()

    # Load defined paths in the configurator
    Console.printf_section("Load and prepare data")
    configurator = file.Configurator(path_folder="/home/mschuster/projects/Synthetic_MRSI/code/config/",
                                     file_name="config_30042024.json")
    configurator.load()
    configurator.print_formatted()

    # Load metabolic mask
    metabolic_mask = file.Mask.load(configurator=configurator,
                                    mask_name="metabolites")

    # Load the FIDs
    metabolites = file.FID(configurator=configurator)
    metabolites.load(fid_name="metabolites",
                     signal_data_type=np.complex64)

    loaded_fid = metabolites.loaded_fid  # contains the signal of 11 chemical compounds

    #########################################################################################################################
    ### Using Glu map and interpolate them
    ## Only Cr+PCr = Creatine+Phosphocreatine possible
    # NOTE: _moi means sub-part
    # 0. Acetyl + Aspartyl_moi(NAAG) ----->  shape: (1536,) == > NAAG
    # 1. Choline_moi(GPC) -------------->    shape: (1536,) == > GPC + PCh                      ----> Ignore
    # 2. Creatine(Cr) ----------------->     shape: (1536,) == > CR + PCr
    # 3. Glutamate(Glu) --------------->     shape: (1536,) == > Glu
    # 4. Glutamate_moi(NAAG) ----------->    shape: (1536,) == > NAAG
    # 5. Glutamine_noNH2(Gln) --------->     shape: (1536,) == > Gln
    # 6. Glycerol_moi(GPC) ------------->    shape: (1536,) == > GPC + PCh      (not yet used)  ----> 1+6 (Glycerol_moi and Choline_moi are parts of the same molecule: GPC)
    # 7. MyoInositol(m - Ins) ----------->   shape: (1536,) == > Ins
    # 8. NAcetylAspartate(NAA) -------->     shape: (1536,) == > NAA
    # 9. Phosphocreatine(PCr) --------->     shape: (1536,) == > CR + PCr                       ----> put together /2
    # 10. PhosphorylCholine_new1(PC) - -->   shape: (1536,) == > --> GPC+PCH ((1+6+10)/2 FID)


    # TODO: Create function with N arguments to interpolate, each is a metabolic property map. Or a list of metabolic property maps: better --> and get as output list or tuple
    # NAAG (0), not 4    --> NAAG
    # NOT USED!

    # not
    # Cr+PCr  --> Cr+PCr      --> MetMap_Cr+PCr_con_map_TargetRes_HiRes.nii
    # NOT USED! Usage is below.
    fid_and_concentration = []
    fid_prepared = spectral_spatial_simulation.FID()

    # Glu     --> Glu         --> MetMap_Glu_con_map_TargetRes_HiRes.nii
    path_to_one_map = os.path.join(configurator.data["path"]["maps"]["metabolites"], "MetMap_Glu_con_map_TargetRes_HiRes.nii")
    metabolic_map = file.NeuroImage(path=path_to_one_map).load_nii().data
    zoom_factor_metabolic_map = np.divide(metabolic_mask.shape, metabolic_map.shape)
    metabolic_map = zoom(input=metabolic_map, zoom=zoom_factor_metabolic_map, order=3)
    fid = loaded_fid.get_signal_by_name("Glutamate (Glu)")
    fid.name = fid.get_name_abbreviation()
    fid_prepared += fid
    fid_and_concentration.append([fid, metabolic_map])

    # Gln     --> Gln         --> MetMap_Gln_con_map_TargetRes_HiRes.nii
    path_to_one_map = os.path.join(configurator.data["path"]["maps"]["metabolites"], "MetMap_Gln_con_map_TargetRes_HiRes.nii")
    metabolic_map = file.NeuroImage(path=path_to_one_map).load_nii().data
    zoom_factor_metabolic_map = np.divide(metabolic_mask.shape, metabolic_map.shape)
    metabolic_map = zoom(input=metabolic_map, zoom=zoom_factor_metabolic_map, order=3)
    fid = loaded_fid.get_signal_by_name("Glutamine_noNH2 (Gln)")
    fid.name = fid.get_name_abbreviation()
    fid_prepared += fid
    fid_and_concentration.append([fid, metabolic_map])

    #### GPC+PCh --> GPC (1+6)   --> MetMap_GPC+PCh_con_map_TargetRes_HiRes.nii
    ###path_to_one_map = os.path.join(configurator.data["path"]["maps"]["metabolites"], "MetMap_GPC+PCh_con_map_TargetRes_HiRes.nii")
    ###metabolic_map = file.NeuroImage(path=path_to_one_map).load_nii().data
    ###fid1 = loaded_fid.get_signal_by_name("Choline_moi(GPC)")
    ###fid2 = loaded_fid.get_signal_by_name("Glycerol_moi(GPC)")
    ###signal = fid1.signal+fid2.signal
    ###name = [fid1.get_name_abbreviation()[0] + "+" + fid2.get_name_abbreviation()[0]]
    ###fid = spectral_spatial_simulation.FID(signal=signal, name=name, time=fid1.time)
    ###fid_prepared += fid

    # Ins     --> Ins         --> MetMap_Ins_con_map_TargetRes_HiRes.nii
    path_to_one_map = os.path.join(configurator.data["path"]["maps"]["metabolites"], "MetMap_Ins_con_map_TargetRes_HiRes.nii")
    metabolic_map = file.NeuroImage(path=path_to_one_map).load_nii().data
    zoom_factor_metabolic_map = np.divide(metabolic_mask.shape, metabolic_map.shape)
    metabolic_map = zoom(input=metabolic_map, zoom=zoom_factor_metabolic_map, order=3)
    fid = loaded_fid.get_signal_by_name("MyoInositol (m-Ins)")
    fid.name = fid.get_name_abbreviation()
    fid_prepared += fid
    fid_and_concentration.append([fid, metabolic_map])

    # NAA     --> NAA         --> MetMap_NAA_con_map_TargetRes_HiRes.nii
    path_to_one_map = os.path.join(configurator.data["path"]["maps"]["metabolites"], "MetMap_NAA_con_map_TargetRes_HiRes.nii")
    metabolic_map = file.NeuroImage(path=path_to_one_map).load_nii().data
    zoom_factor_metabolic_map = np.divide(metabolic_mask.shape, metabolic_map.shape)
    metabolic_map = zoom(input=metabolic_map, zoom=zoom_factor_metabolic_map, order=3)
    fid = loaded_fid.get_signal_by_name("NAcetylAspartate (NAA)")
    fid.name = fid.get_name_abbreviation()
    fid_prepared += fid
    fid_and_concentration.append([fid, metabolic_map])

    # Cr+PCr  --> Cr+PCr/2    --> MetMap_Cr+PCr_con_map_TargetRes_HiRes.nii
    path_to_one_map = os.path.join(configurator.data["path"]["maps"]["metabolites"], "MetMap_Cr+PCr_con_map_TargetRes_HiRes.nii")
    metabolic_map = file.NeuroImage(path=path_to_one_map).load_nii().data
    zoom_factor_metabolic_map = np.divide(metabolic_mask.shape, metabolic_map.shape)
    metabolic_map = zoom(input=metabolic_map, zoom=zoom_factor_metabolic_map, order=3)
    fid1 = loaded_fid.get_signal_by_name("Creatine (Cr)")
    fid2 = loaded_fid.get_signal_by_name("Phosphocreatine (PCr)")
    signal = (fid1.signal+fid2.signal)/2
    name = [fid1.get_name_abbreviation()[0] + "+" + fid2.get_name_abbreviation()[0]]
    fid = spectral_spatial_simulation.FID(signal=signal, name=name, time=fid1.time)
    fid_prepared += fid
    fid_and_concentration.append([fid, metabolic_map])

    # GPC+PCH --> (1+6+10)/2  --> MetMap_GPC+PCh_con_map_TargetRes_HiRes.nii
    path_to_one_map = os.path.join(configurator.data["path"]["maps"]["metabolites"], "MetMap_GPC+PCh_con_map_TargetRes_HiRes.nii")
    metabolic_map = file.NeuroImage(path=path_to_one_map).load_nii().data
    zoom_factor_metabolic_map = np.divide(metabolic_mask.shape, metabolic_map.shape)
    metabolic_map = zoom(input=metabolic_map, zoom=zoom_factor_metabolic_map, order=3)
    fid1 = loaded_fid.get_signal_by_name("Choline_moi(GPC)")
    fid2 = loaded_fid.get_signal_by_name("Glycerol_moi(GPC)")
    fid3 = loaded_fid.get_signal_by_name("PhosphorylCholine_new1 (PC)")
    signal = (fid1.signal+fid2.signal+fid3.signal)/3
    name = fid1.get_name_abbreviation()
    fid = spectral_spatial_simulation.FID(signal=signal, name=name, time=fid1.time)
    fid_prepared += fid
    fid_and_concentration.append([fid, metabolic_map])

    Console.printf("success", f"Interpolated all concentration maps to shape: {metabolic_mask.shape}")

    #### Load and interpolate Glu map and T1 map
    ###path_to_one_map = os.path.join(configurator.data["path"]["maps"]["metabolites"], "MetMap_Glu_con_map_TargetRes_HiRes.nii")
    ###loaded_metabolic_map = file.NeuroImage(path=path_to_one_map).load_nii().data
    ###zoom_factor_one_metabolic_map = np.divide(metabolic_mask.shape, loaded_metabolic_map.shape)
    ###loaded_metabolic_map_interpolated = zoom(input=loaded_metabolic_map, zoom=zoom_factor_one_metabolic_map, order=3)
    ###Console.printf("success", f"Interpolated Glu Map: from {loaded_metabolic_map.shape} to --> {loaded_metabolic_map_interpolated.shape}")

    path_to_one_map = os.path.join(configurator.data["path"]["maps"]["metabolites"], "T1_TargetRes_HiRes.nii")
    loaded_T1_map = file.NeuroImage(path=path_to_one_map).load_nii().data
    zoom_factor_one_T1_map = np.divide(metabolic_mask.shape, loaded_T1_map.shape)
    loaded_T1_map_interpolated = zoom(input=loaded_T1_map, zoom=zoom_factor_one_T1_map, order=3)
    Console.printf("success", f"Interpolated T1 Map: from {loaded_T1_map.shape} to --> {loaded_T1_map_interpolated.shape}")

    Console.printf("success", f"Interpolated one T1 map to shape: {metabolic_mask.shape}")

    ####################################
    # Configuration:
    block_size = (int(1), int(112), int(128), int(80))
    ####################################

    # Create MetabolicPropertyMap (theoretically for each metabolite; however, only Glu used so far)
    metabolic_property_map_dict = {}
    for fid, concentration_map in fid_and_concentration:
        #concentration_map = loaded_metabolic_map_interpolated
        t1_map = loaded_T1_map_interpolated * 1e-3  # Assuming it is in [ms]
        t2_random_map = np.random.uniform(low=55 * 1e-3, high=70 * 1e-3, size=metabolic_mask.data.shape)

        metabolic_property_map = MetabolicPropertyMap(chemical_compound_name=fid.name[0],
                                                      block_size=(block_size[1], block_size[2], block_size[3]), # 10,10,10
                                                      t1=t1_map,
                                                      t1_unit=u.ms,
                                                      t2=t2_random_map,
                                                      t2_unit=u.ms,
                                                      concentration=concentration_map,
                                                      concentration_unit=u.mmol)

        metabolic_property_map_dict[metabolic_property_map.chemical_compound_name] = metabolic_property_map

    # Create spectral spatial model
    simulation = spectral_spatial_simulation.Model(path_cache='/home/mschuster/projects/Synthetic_MRSI/cache/dask_tmp',
                                                   block_size=block_size, # TODO: was 1536x10x10x10
                                                   TE=0.0013,
                                                   TR=0.6,
                                                   alpha=45)
                                                    # TODO: Also state size of one block after created model!

    # Add components to the Model
    #Console.ask_user("Create computational graph?")
    simulation.add_metabolic_property_maps(metabolic_property_map_dict)  # all maps
    simulation.add_fid(fid_prepared)  # all fid signals
    simulation.add_mask(metabolic_mask.data)

    computational_graph = simulation.assemble_graph()
    computational_graph.dask.visualize(filename='visualisation/dask_graph_high_level.png')

    # Start client & and start dasboard, since computation is below
    memory_limit_per_worker = "20GB"

    cluster = LocalCluster(n_workers=5, threads_per_worker=8, memory_limit=memory_limit_per_worker)  # --> 5,8 fro cross sectional view;  --> 4,5 for whole 1536
    client = Client(cluster)
    dashboard_url = client.dashboard_link
    print(dashboard_url)

    # Display one slice per axis of time point 100
    #Console.start_timer()
    #fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    # Plotting data on each subplot using imshow
    all_blocks = computational_graph.blocks.ravel()

    ###print(type(all_blocks))
    ###print(type(all_blocks[0]))
    ###print(len(all_blocks))
    ###print(all_blocks[0].shape)
    ###print("==================================")
    ###
    ###Console.start_timer()
    ###with ms.sample("collection 1"), ProgressBar():
    ###    for i, block in enumerate(all_blocks):
    ###        block.compute()
    ###        print(f"computed block no {i}/ {len(all_blocks)}")
    ###        print(f"block shape: {block.shape}")
    ###Console.stop_timer()

    Console.start_timer()
    print("====================================== Compute chunks in parallel:")
    blocks = 500
    batches_slice = list(gen_batches(len(all_blocks), blocks))
    print(f"Number of batches {len(batches_slice)}, each {blocks} blocks")
    ms = MemorySampler()
    with ms.sample("collection 1"):
        for i, s in enumerate(batches_slice):
            print(f"compute batch {i}")
            dask.compute(all_blocks[s])

    Console.stop_timer()
    ms.plot(align=True)
    plt.show()



if __name__ == '__main__':
    config = Config(max_depth=4)
    graphviz = GraphvizOutput(output_file='pycallgraph.png')
    with PyCallGraph(output=GraphvizOutput(), config=config):
        main_entry()