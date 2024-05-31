import os.path

import torch

import default
from spatial_metabolic_distribution import Maps, MetabolicPropertyMapsAssembler, MetabolicPropertyMap
from spectral_spatial_simulation import Model as SpectralSpatialModel
from distributed.diagnostics import MemorySampler
from spectral_spatial_simulation import FID
from dask.diagnostics import ProgressBar
from sklearn.utils import gen_batches
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from printer import Console
import numpy as np
import cupy as cp
import sampling
import tools
import dask
import pint
import file
import sys

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
    Console.printf_section("Load FID and Mask")
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

    Console.printf_section("Load and prepare the Maps")
    # Load and prepare the concentration maps
    loaded_concentration_maps = file.Maps(configurator=configurator, map_type_name="metabolites")
    working_name_and_file_name = {"Glu": "MetMap_Glu_con_map_TargetRes_HiRes.nii",
                                  "Gln": "MetMap_Gln_con_map_TargetRes_HiRes.nii",
                                  "m-Ins": "MetMap_Ins_con_map_TargetRes_HiRes.nii",
                                  "NAA": "MetMap_NAA_con_map_TargetRes_HiRes.nii",
                                  "Cr+PCr": "MetMap_Cr+PCr_con_map_TargetRes_HiRes.nii",
                                  "GPC+PCh": "MetMap_GPC+PCh_con_map_TargetRes_HiRes.nii",
                                  }
    loaded_concentration_maps.load(working_name_and_file_name=working_name_and_file_name)

    # Assign loaded maps from module file to maps object from the spatial metabolic distribution & Interpolate it
    concentration_maps = Maps(loaded_concentration_maps.loaded_maps)
    concentration_maps.interpolate_to_target_size(target_size=metabolic_mask.shape, order=3, target_device='cuda', target_gpu=1)

    # Load T1 map (TODO: only one for the moment, but need one for each metabolite?)
    path_to_one_map = os.path.join(configurator.data["path"]["maps"]["metabolites"], "T1_TargetRes_HiRes.nii")
    loaded_T1_map = file.NeuroImage(path=path_to_one_map).load_nii().data * 1e-3

    working_name_and_map = {"Glu": loaded_T1_map,
                            "Gln": loaded_T1_map,
                            "m-Ins": loaded_T1_map,
                            "NAA": loaded_T1_map,
                            "Cr+PCr": loaded_T1_map,
                            "GPC+PCh": loaded_T1_map,
                            }
    t1_maps = Maps(maps=working_name_and_map)
    t1_maps.interpolate_to_target_size(target_size=metabolic_mask.shape, order=3, target_device='cuda', target_gpu=1)

    # Create T2 map (TODO: only random values for the moment!)
    working_name_and_map = {"Glu": np.random.uniform(low=55 * 1e-3, high=70 * 1e-3, size=metabolic_mask.data.shape),
                            "Gln": np.random.uniform(low=55 * 1e-3, high=70 * 1e-3, size=metabolic_mask.data.shape),
                            "m-Ins": np.random.uniform(low=55 * 1e-3, high=70 * 1e-3, size=metabolic_mask.data.shape),
                            "NAA": np.random.uniform(low=55 * 1e-3, high=70 * 1e-3, size=metabolic_mask.data.shape),
                            "Cr+PCr": np.random.uniform(low=55 * 1e-3, high=70 * 1e-3, size=metabolic_mask.data.shape),
                            "GPC+PCh": np.random.uniform(low=55 * 1e-3, high=70 * 1e-3, size=metabolic_mask.data.shape),
                            }
    t2_maps = Maps(maps=working_name_and_map)
    t2_maps.interpolate_to_target_size(target_size=metabolic_mask.shape, order=3, target_device='cuda', target_gpu=1)


    # Get partially FID containign only the desired signals
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

    Console.printf_section("Use desired FID signals and merge some of them")
    fid = loaded_fid.get_partly_fid(using_fid_signals)

    # Merge signals of the FID in order to match the Maps
    fid.merge_signals(names=["Creatine (Cr)", "Phosphocreatine (PCr)"], new_name="Creatine (Cr)+Phosphocreatine (PCr)", divisor=2)
    fid.merge_signals(names=["Choline_moi(GPC)", "Glycerol_moi(GPC)", "PhosphorylCholine_new1 (PC)"], new_name="GPC+PCh", divisor=2)
    fid.name = fid.get_name_abbreviation()
    print(fid)

    Console.printf_section("Assemble FID and Maps")
    block_size = (int(1), int(112), int(128), int(80))
    assembler = MetabolicPropertyMapsAssembler(block_size=(block_size[1], block_size[2], block_size[3]),
                                               concentration_maps=concentration_maps,
                                               t1_maps=t1_maps,
                                               t2_maps=t2_maps,
                                               concentration_unit=u.mmol,
                                               t1_unit=u.ms,
                                               t2_unit=u.ms)
    metabolic_property_map_dict = assembler.assemble()

    Console.printf_section("Create Spectral-Spatial-Model")
    # Create spectral spatial model
    spectral_spatial_model = SpectralSpatialModel(path_cache='/home/mschuster/projects/Synthetic_MRSI/cache/dask_tmp',
                                                  block_size=block_size,  # TODO: was 1536x10x10x10
                                                  TE=0.0013,
                                                  TR=0.6,
                                                  alpha=45,
                                                  data_type="complex64",
                                                  compute_on_device="cuda",
                                                  return_on_device="cpu")
    # TODO: Also state size of one block after created model!

    spectral_spatial_model.add_metabolic_property_maps(metabolic_property_map_dict)  # all maps
    spectral_spatial_model.add_fid(fid)  # all fid signals  # TODO: change back to fid_prepared
    spectral_spatial_model.add_mask(metabolic_mask.data)

    spectral_spatial_model.model_summary()

    sampling_model = sampling.Model(spectral_spatial_model=spectral_spatial_model, compute_on_device="cuda", return_on_device="cpu")
    computational_graph = sampling_model.cartesian_FFT_graph()


    cluster = tools.MyLocalCluster()
    #cluster.start_cpu(number_workers=5, threads_per_worker=8, memory_limit_per_worker="8GB")
    cluster.start_cuda(device_numbers=[1,2], device_memory_limit="30GB")


    all_blocks = computational_graph.blocks.ravel()


    Console.printf("info", "Compute chunks in parallel:")
    blocks = 1536 # 1536 # TODO: was 500 with FID point 1536 OR 10_000
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