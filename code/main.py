import os.path

import cupy
import torch

import default
from spatial_metabolic_distribution import Maps, MetabolicPropertyMapsAssembler, MetabolicPropertyMap
from spectral_spatial_simulation import Model as SpectralSpatialModel
from distributed.diagnostics import MemorySampler
from cupyx.scipy.ndimage import zoom as zoom_gpu
from spectral_spatial_simulation import FID
from dask.diagnostics import ProgressBar
from sklearn.utils import gen_batches
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from tools import CustomArray, CustomArray2
from printer import Console
import dask.array as da
import xarray as xr
import numpy as np
import cupy as cp
import sampling
import tools
import h5py
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

from cupyx.profiler import benchmark


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
    target_gpu_smaller_tasks = 2

    # Initialize the UnitRegistry
    u = pint.UnitRegistry()

    # Load defined paths in the configurator
    Console.printf_section("Load FID and Mask")
    configurator = file.Configurator(path_folder="/home/mschuster/projects/Synthetic_MRSI/code/config/",
                                     file_name="config_12062024.json")
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
    loaded_concentration_maps.load_files_from_folder(working_name_and_file_name=working_name_and_file_name)

    # Assign loaded maps from module file to maps object from the spatial metabolic distribution & Interpolate it
    concentration_maps = Maps(loaded_concentration_maps.loaded_maps)
    concentration_maps.interpolate_to_target_size(target_size=metabolic_mask.shape, order=3, target_device='cuda', target_gpu=target_gpu_smaller_tasks)

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
    t1_maps.interpolate_to_target_size(target_size=metabolic_mask.shape, order=3, target_device='cuda', target_gpu=target_gpu_smaller_tasks)

    # Create T2 map (TODO: only random values for the moment!)
    working_name_and_map = {"Glu": np.random.uniform(low=55 * 1e-3, high=70 * 1e-3, size=metabolic_mask.data.shape),
                            "Gln": np.random.uniform(low=55 * 1e-3, high=70 * 1e-3, size=metabolic_mask.data.shape),
                            "m-Ins": np.random.uniform(low=55 * 1e-3, high=70 * 1e-3, size=metabolic_mask.data.shape),
                            "NAA": np.random.uniform(low=55 * 1e-3, high=70 * 1e-3, size=metabolic_mask.data.shape),
                            "Cr+PCr": np.random.uniform(low=55 * 1e-3, high=70 * 1e-3, size=metabolic_mask.data.shape),
                            "GPC+PCh": np.random.uniform(low=55 * 1e-3, high=70 * 1e-3, size=metabolic_mask.data.shape),
                            }
    t2_maps = Maps(maps=working_name_and_map)
    t2_maps.interpolate_to_target_size(target_size=metabolic_mask.shape, order=3, target_device='cuda', target_gpu=target_gpu_smaller_tasks)

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

    # TODO
    #fid.signal = np.random.rand(6, 100_000) + 1j * np.random.rand(6, 100_000)
    #fid.time = np.arange(0, 100_000)
    Console.printf("warning", "Replaced loaded simulated FID signals of length 1536 to random values of length 100000")

    Console.printf_section("Assemble FID and Maps")
    block_size = (int(20), int(112), int(128), int(80))
    # block_size = (int(200), int(60), int(60), int(20))
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
                                                  return_on_device="cuda")
    # TODO: Also state size of one block after created model!

    spectral_spatial_model.add_metabolic_property_maps(metabolic_property_map_dict)  # all maps
    spectral_spatial_model.add_fid(fid)  # all fid signals  # TODO: change back to fid_prepared
    spectral_spatial_model.add_mask(metabolic_mask.data)

    spectral_spatial_model.model_summary()
    computational_graph = spectral_spatial_model.assemble_graph()

    cluster = tools.MyLocalCluster()
    cluster.start_cuda(device_numbers=[1,2,3], device_memory_limit="30GB", use_rmm_cupy_allocator=True)

    # ########################################################################################
    # FID + volume before
    computational_for_cross_sectional_plot = computational_graph
    #cross_section_1_before = computational_for_cross_sectional_plot[20, 40, :, :].compute()
    #cross_section_2_before = computational_for_cross_sectional_plot[20, :, 40, :].compute()
    #cross_section_3_before = computational_for_cross_sectional_plot[20, :, :, 40].compute()

    computational_graph_one_fid = computational_graph.rechunk(chunks=(1536,10,10,10)) # TODO: do fft
    #one_fid_before = computational_graph_one_fid[:, 40, 40, 40].compute()
    # ########################################################################################

    # Implement the following workflow:
    #  1) load the coil sensitivity maps
    Console.printf("info","Start loading the coil sensitivity maps")
    path_coil_sensitivity_maps = configurator.data["path"]["maps"]["coil_sensitivity"]
    coil_sensitivity_maps = h5py.File(path_coil_sensitivity_maps, "r")
    Console.printf("info", "Start converting the coil sensitivity maps from h5py to numpy + assemble parts to complex")


    with cp.cuda.Device(target_gpu_smaller_tasks):
        coil_sensitivity_maps_real = cp.asarray(coil_sensitivity_maps["real"][:])
        coil_sensitivity_maps_imag = cp.asarray(coil_sensitivity_maps["imag"][:])
        coil_sensitivity_maps_complex = coil_sensitivity_maps_real + 1j * coil_sensitivity_maps_imag
        target_size = (32, metabolic_mask.shape[0], metabolic_mask.shape[1], metabolic_mask.shape[2])
        Console.printf("info", "Interpolate to target size")
        zoom_factor = np.divide(target_size, coil_sensitivity_maps_complex.shape)
        coil_sensitivity_maps_complex_interpolated = zoom_gpu(input=coil_sensitivity_maps_complex, zoom=zoom_factor, order=3)


    # 1b) bring to cpu
    #computational_graph = computational_graph.map_blocks(cp.asnumpy) # Very important to put on CPU after computing!
    # 1c) compute block wise
    #compute_blocks(computational_graph=computational_graph, n_blocks=20, coil_sensitivity_maps=coil_sensitivity_maps)

    #  2) multiply each map (32?) with the spectral spatial model
    #cumulative_sum_coils = dask.array.zeros((1536, 64, 64, 39), dtype=np.complex64) # TODO: OLD
    #with cp.cuda.Device(target_gpu_smaller_tasks):
    cumulative_sum_coils = np.zeros(shape=(1536, 64, 64, 40), dtype=np.complex64)

    ####Console.printf("info", "Start GPU cluster for computation")
    ####cluster = tools.MyLocalCluster()
    ####cluster.start_cuda(device_numbers=[2], device_memory_limit="30GB", use_rmm_cupy_allocator=False)

    number_coil_sensitivity_maps = coil_sensitivity_maps_complex_interpolated.shape[0]
    for i in range(number_coil_sensitivity_maps):
        # Multiply the whole graph times one coil sensitivity map for each block
        Console.printf("info", "Multiply the whole graph times one coil sensitivity map for each block")
        one_coil_image_domain = computational_graph[:, :, :, :] * coil_sensitivity_maps_complex_interpolated[i, :, :, :]
        #  3) fft -> get high-resolution k-space (32?), make not parallel TODO: compute on GPU
        Console.printf("info", "FFT along axes 1,2,3")
        one_coil_k_space = one_coil_image_domain.map_blocks(cp.fft.fftn, dtype=cp.complex64, axes=(1, 2, 3))
        #  4) Crop the k-space of each 32 to 64x64x39
        Console.printf("info", "Crop k-space to (1536, 64, 64, 40)")
        # center x -> [(shape.x/2)-(64/2), (shape.x/2)+(64/2)]
        # center y -> [(shape.x/2)-(64/2), (shape.x/2)+(64/2)]
        # center z -> [(shape.z/2)-19), (shape.z/2)+20)]
        original_shape = (1536, 112, 128, 80)
        desired_shape = (1536, 64, 64, 40)
        start_indices = [(orig - des) // 2 for orig, des in zip(original_shape, desired_shape)]
        end_indices = [start + des for start, des in zip(start_indices, desired_shape)]
        one_coil_k_space_cropped = one_coil_k_space[
                                   :,
                                   start_indices[1]:end_indices[1],
                                   start_indices[2]:end_indices[2],
                                   start_indices[3]:end_indices[3]
                                   ]
        #  5) Add gaussian noise to cropped k-space of each 32 coils (individually? -> guess yes, because is measurement noise). Make function with SNR as argument
        # TODO TODO TODO TODO
        one_coil_k_space_cropped_noisy = one_coil_k_space_cropped

        # 6) Bring back to image domain
        one_coil_image_domain_cropped_noisy = one_coil_k_space_cropped_noisy.map_blocks(cp.fft.ifftn, dtype=cp.complex64, axes=(1, 2, 3))

        # 7) Coil combination (I guess just to add them together? sum?) TODO =>just sum them up?
        one_coil_image_domain_cropped_noisy = one_coil_image_domain_cropped_noisy.map_blocks(cp.asnumpy) # very important!!!

        cumulative_sum_coils += one_coil_image_domain_cropped_noisy.compute()
        Console.printf("success", f"Computed for coil {i}")

    cumulative_sum_coils = da.from_array(cumulative_sum_coils, chunks="auto")


    #### For cross-sectional plot
    cross_section_1_before = np.abs(cp.asnumpy(computational_for_cross_sectional_plot[20, 40, :, :].compute()))
    cross_section_2_before = np.abs(cp.asnumpy(computational_for_cross_sectional_plot[20, :, 40, :].compute()))
    cross_section_3_before = np.abs(cp.asnumpy(computational_for_cross_sectional_plot[20, :, :, 40].compute()))

    cross_section_1_after = np.abs(cumulative_sum_coils[20, int(40/2), :, :].compute())
    cross_section_2_after = np.abs(cumulative_sum_coils[20, :, int(40/2), :].compute())
    cross_section_3_after = np.abs(cumulative_sum_coils[20, :, :, int(40/2)].compute())

    #### For fid
    start_x = int((112-64)/2)
    start_y = int((128-64)/2)
    start_z = int((80-40)/2)
    end_x = start_x + 64
    end_y = start_y + 64
    end_z = start_z + 40

    example_x = 30
    example_y = 35
    example_z = 12

    one_fid_before = computational_graph_one_fid[:, start_x+example_x, start_y+example_y, start_z+example_z].compute()
    one_fid_before_spectrum = np.abs(np.fft.fft(one_fid_before))
    computational_graph_one_fid = computational_graph.rechunk(chunks=(1536,10,10,10))
    one_fid_after = computational_graph_one_fid[:, example_x, example_y, example_z].compute()
    one_fid_after_spectrum = np.abs(np.fft.fft(one_fid_after))

    ####################plots########################
    # Create a figure with a 2x3 grid for the first plot
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Plot the images on the grid
    axs[0, 0].imshow(cross_section_1_before, cmap='gray')
    axs[0, 0].set_title('Cross Section 1 Before')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(cross_section_1_after, cmap='gray')
    axs[0, 1].set_title('Cross Section 1 After')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(cross_section_2_before, cmap='gray')
    axs[1, 0].set_title('Cross Section 2 Before')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(cross_section_2_after, cmap='gray')
    axs[1, 1].set_title('Cross Section 2 After')
    axs[1, 1].axis('off')

    axs[2, 0].imshow(cross_section_3_before, cmap='gray')
    axs[2, 0].set_title('Cross Section 3 Before')
    axs[2, 0].axis('off')

    axs[2, 1].imshow(cross_section_3_after, cmap='gray')
    axs[2, 1].set_title('Cross Section 3 After')
    axs[2, 1].axis('off')

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the 2D signals
    axs[0].imshow(cp.asnumpy(one_fid_before_spectrum), cmap='viridis', aspect='auto')
    axs[0].set_title('One FID Before')
    axs[0].axis('off')

    axs[1].imshow(one_fid_after_spectrum, cmap='viridis', aspect='auto')
    axs[1].set_title('One FID After')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

    input("-----------------")

    # TODO: Then create again dask array!?
    #input("------------------")
    #print(cumulative_sum_coils)


    #Console.printf("info", "Start CPU cluster and compute k-space of one coil")
    #cluster = tools.MyLocalCluster()
    #cluster.start_cuda(device_numbers=[1, 2, 3], device_memory_limit="30GB", use_rmm_cupy_allocator=False)
    #cluster.start_cpu(number_workers=5, threads_per_worker=8, memory_limit_per_worker="8GB")
    #cumulative_sum_coils = cumulative_sum_coils.map_blocks(cp.asnumpy)

    #cluster = tools.MyLocalCluster()
    #cluster.start_cpu(number_workers=5, threads_per_worker=8, memory_limit_per_worker="30GB")
    #cluster.start_cuda(device_numbers=[1,2,3], device_memory_limit="30GB", use_rmm_cupy_allocator=True)


    cumulative_sum_coils.blocks.ravel()[0].compute()

    input("?????????????????????????????????????")



    #  8) plot cross-sectional view
    #      --> high resolution image before    | fid (e.g., 1-50 --> value max peak)
    #      --> low resolution image afterwards | fid (e.g., 1-50 --> value max peak)

    ####### BEFORE FFT PART #######

    sampling_model = sampling.Model(spectral_spatial_model=spectral_spatial_model, compute_on_device="cuda", return_on_device="cpu")
    computational_graph = sampling_model.cartesian_FFT_graph()

    # computational_graph = xr.DataArray(computational_graph)
    computational_graph = CustomArray2(computational_graph)

    print(computational_graph)
    print(computational_graph.blocks[0])
    print(computational_graph.blocks[1])
    print(computational_graph.blocks[2])
    print(computational_graph.blocks[400])
    input("==================")

    cluster = tools.MyLocalCluster()
    # cluster.start_cpu(number_workers=5, threads_per_worker=8, memory_limit_per_worker="8GB")
    cluster.start_cuda(device_numbers=[1, 2], device_memory_limit="30GB", use_rmm_cupy_allocator=True)

    # print(benchmark(compute_blocks, (computational_graph,), n_repeat=5, devices=(1,2)))
    # print(computational_graph)
    compute_blocks(computational_graph)


# TODO: Maybe compute blocks here and then apply the remaining steps ;) --> and then again with a conversion into dask array ;)
def compute_blocks(computational_graph, n_blocks=20, coil_sensitivity_maps=None):
    all_blocks = computational_graph.blocks.ravel()

    Console.printf("info", "Compute chunks in parallel:")
    blocks = n_blocks  # 1536 # TODO: was 500 with FID point 1536 OR 10_000
    batches_slice = list(gen_batches(len(all_blocks), blocks))
    print(f"Number of batches {len(batches_slice)}, each {blocks} blocks")
    ms = MemorySampler()
    Console.start_timer()
    with ms.sample("collection 1"):
        for i, s in tqdm(enumerate(batches_slice), total=len(batches_slice)):
            ####print(f"compute batch {i}") ###
            # Console.start_timer()
            dask.compute(all_blocks[s])[0]
            ###for block in blocks:
            ###    for fid_index in len(block.shape[0]):
            ###        block[fid_index,:,:,:] * coil_sensitivity_maps_complex_interpolated[i,:,:,:]

            # TODO: for loop and apply * coil_sensitivity_maps_complex_interpolated[i,:,:,:]


            # Console.stop_timer()
            ###blocks_squeezed = [np.squeeze(block, axis=0) for block in blocks]
            ###result = np.stack(blocks_squeezed, axis=0)
            ###print(result.shape)

    Console.stop_timer()

    # ms.plot(align=True)
    # plt.show()


if __name__ == '__main__':
    config = Config(max_depth=4)
    graphviz = GraphvizOutput(output_file='pycallgraph.png')
    with PyCallGraph(output=GraphvizOutput(), config=config):
        main_entry()
