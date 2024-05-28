import os.path

import torch

import default

from spatial_metabolic_distribution import MetabolicPropertyMap, MetabolicPropertyMapsAssembler
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
import cupy

from tqdm import tqdm

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

    # Initialize the UnitRegistry
    u = pint.UnitRegistry()

    # Load defined paths in the configurator
    Console.printf_section("Load the data")
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

    # Contains the signal of 11 chemical compounds
    loaded_fid = metabolites.loaded_fid

    # Load and prepare the concentration maps
    loaded_concentration_maps = file.Maps(configurator=configurator, map_type_name="metabolites")
    working_name_and_file_name = {"Glu": "MetMap_Glu_con_map_TargetRes_HiRes.nii",
                                  "Gln": "MetMap_Gln_con_map_TargetRes_HiRes.nii",
                                  "Ins": "MetMap_Ins_con_map_TargetRes_HiRes.nii",
                                  "NAA": "MetMap_NAA_con_map_TargetRes_HiRes.nii",
                                  "Cr+PCr": "MetMap_Cr+PCr_con_map_TargetRes_HiRes.nii",
                                  "GPC+PCh": "MetMap_GPC+PCh_con_map_TargetRes_HiRes.nii",
                                  }

    loaded_concentration_maps.load(working_name_and_file_name=working_name_and_file_name)
    loaded_concentration_maps.interpolate_to_target_size(target_size=metabolic_mask.shape) # TODO only nii supported at the beginning! Also other formats?


    using_fid_signals = ["Glutamate (Glu)",
                         "Glutamine_noNH2 (Gln)",
                         "MyoInositol (m-Ins)",
                         "NAcetylAspartate (NAA)",
                         "Creatine (Cr)",
                         "Phosphocreatine (PCr)",
                         "Choline_moi(GPC)",
                         "Glycerol_moi(GPC)",
                         "PhosphorylCholine_new1 (PC)"
                         ]

    fid = loaded_fid.get_partly_fid(using_fid_signals)

    fid.merge_signals(names=["Creatine (Cr)", "Phosphocreatine (PCr)"], new_name="Creatine (Cr)+Phosphocreatine (PCr)", divisor=2)
    fid.merge_signals(names=["Choline_moi(GPC)", "Glycerol_moi(GPC)", "PhosphorylCholine_new1 (PC)"], new_name="Choline_moi(GPC)+Glycerol_moi(GPC)", divisor=2)
    fid.name = fid.get_name_abbreviation()
    print(fid)

    # TODO TODO TODO TODO TODO
    assembler = MetabolicPropertyMapsAssembler(fid=fid,
                                               concentration_maps=loaded_concentration_maps,
                                               T1_maps=None, # TODO make Maps in Spatial!
                                               T2_maps=None, # TODO make Maps in Spatial!
                                               concentration_unit=u.mmol,
                                               T1_unit=u.ms,
                                               T2_unit=u.ms)




    # TODO TODO TODO: assemble maps and FID together to MetabolicPropertyMaps



    fid_prepared = spectral_spatial_simulation.FID()
    fid_and_concentration = []


    # Maps class?
    # as input the configurator?
    # then only pass list of names??? --> and abbrevation --> and match option=>does do interpolation and also does do matching regarding names??
    #                            --> match FID and all maps!!!!!!! --> it should not dependent on path

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
    signal = (fid1.signal+fid2.signal+fid3.signal)/2
    name = fid1.get_name_abbreviation()
    fid = spectral_spatial_simulation.FID(signal=signal, name=name, time=fid1.time)
    fid_prepared += fid
    fid_and_concentration.append([fid, metabolic_map])

    Console.printf("success", f"Interpolated all concentration maps to shape: {metabolic_mask.shape}")




    path_to_one_map = os.path.join(configurator.data["path"]["maps"]["metabolites"], "T1_TargetRes_HiRes.nii")
    loaded_T1_map = file.NeuroImage(path=path_to_one_map).load_nii().data
    zoom_factor_one_T1_map = np.divide(metabolic_mask.shape, loaded_T1_map.shape)
    loaded_T1_map_interpolated = zoom(input=loaded_T1_map, zoom=zoom_factor_one_T1_map, order=3)
    Console.printf("success", f"Interpolated T1 Map: from {loaded_T1_map.shape} to --> {loaded_T1_map_interpolated.shape}")

    Console.printf("success", f"Interpolated one T1 map to shape: {metabolic_mask.shape}")

    ####################################
    # Configuration:
    block_size = (int(5), int(112), int(128), int(80))
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
                                                   alpha=45,
                                                   data_type="complex64",
                                                   compute_on_device="cuda",
                                                   return_on_device="cuda")
                                                    # TODO: Also state size of one block after created model!


    # ############ JUST FOR TEST PURPOSES #################
    ###fid_prepared_2 = spectral_spatial_simulation.FID()
    ###fid_length = 100_000
    ###for fid in fid_prepared:
    ###    # Generate 100000 real and imaginary parts
    ###    real_parts = np.random.rand(fid_length).astype(np.float32)
    ###    imaginary_parts = np.random.rand(fid_length).astype(np.float32)
    ###    # Combine into complex64
    ###    complex_numbers = real_parts + 1j * imaginary_parts
    ###    complex_numbers = complex_numbers.astype(np.complex64)
    ###    fid.signal = complex_numbers
    ###    fid.time = np.arange(0,fid_length)
    ###    fid_prepared_2 += fid
    # ######################################################

    # Add components to the Model
    #Console.ask_user("Create computational graph?")
    simulation.add_metabolic_property_maps(metabolic_property_map_dict)  # all maps
    simulation.add_fid(fid_prepared)  # all fid signals  # TODO: change back to fid_prepared
    simulation.add_mask(metabolic_mask.data)

    simulation.model_summary()

    computational_graph = simulation.assemble_graph()
    computational_graph = computational_graph.map_blocks(np.fft.fftn, dtype=cupy.complex64, axes=(1,2,3)) # TODO: Change to cupy
    #computational_graph = computational_graph.map_blocks(cupy.asnumpy)


    computational_graph.dask.visualize(filename='visualisation/dask_graph_high_level.png')

    # Start client & and start dasboard, since computation is below
    memory_limit_per_worker = "20GB"

    #cluster = LocalCluster(n_workers=5, threads_per_worker=8, memory_limit=memory_limit_per_worker)  # --> 5,8 fro cross sectional view;  --> 4,5 for whole 1536
    from dask_cuda import LocalCUDACluster
    # https://docs.rapids.ai/api/dask-cuda/nightly/spilling/
    cluster = LocalCUDACluster(n_workers=4, device_memory_limit="30GB", jit_unspill=True)
    client = Client(cluster)
    dashboard_url = client.dashboard_link
    print(dashboard_url)

    # Display one slice per axis of time point 100
    #Console.start_timer()
    #fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    # Plotting data on each subplot using imshow

    ### OPTIMISE GRAPH
    ###from dask.optimization import cull, inline, fuse
    ###dsk = computational_graph.__dask_graph__()
    ###dsk, dependencies = cull(dsk, computational_graph.__dask_keys__())
    ###dsk = inline(dsk)
    ###dsk, dependencies = fuse(dsk)
    ####sk = optimize(dsk, keys=computational_graph.__dask_keys__())
    ###computational_graph = dsk
    ###computational_graph.dask.visualize(filename='visualisation/dask_graph_high_level_optimised.png')

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

    #Console.start_timer()
    print("====================================== Compute chunks in parallel:")
    blocks = 200 # TODO: was 500 with FID point 1535 OR 10_000
    batches_slice = list(gen_batches(len(all_blocks), blocks))
    print(f"Number of batches {len(batches_slice)}, each {blocks} blocks")
    ms = MemorySampler()
    Console.start_timer()
    with ms.sample("collection 1"):
        for i, s in tqdm(enumerate(batches_slice), total=len(batches_slice)):
            ####print(f"compute batch {i}") ###
            #Console.start_timer()
            blocks = dask.compute(all_blocks[s])[0]
            #Console.stop_timer()
            ###blocks_squeezed = [np.squeeze(block, axis=0) for block in blocks]
            ###result = np.stack(blocks_squeezed, axis=0)
            ###print(result.shape)

    Console.stop_timer()
    ms.plot(align=True)
    plt.show()


if __name__ == '__main__':
    config = Config(max_depth=4)
    graphviz = GraphvizOutput(output_file='pycallgraph.png')
    with PyCallGraph(output=GraphvizOutput(), config=config):
        main_entry()