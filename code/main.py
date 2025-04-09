import sys

from spatial_metabolic_distribution import Maps, MetabolicPropertyMapsAssembler
from spectral_spatial_simulation import Model as SpectralSpatialModel
from sampling import Model as SamplingModel
from display import plot_FID
from printer import Console
import dask.array as da
import numpy as np
import cupy as cp
import os.path
import tools
import dask
import pint
import file

import h5py

from tqdm import tqdm


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
    cm = tools.CitationManager("../docs/references.bib")

    if True:  # TODO change: use cache here:

        Console.start_timer()
        target_gpu_smaller_tasks = 1
        target_gpus_big_tasks = [1]

        # Initialize the UnitRegistry
        u = pint.UnitRegistry()

        # Load defined paths in the configurator
        Console.printf_section("Load FID and Mask")
        configurator = file.Configurator(path_folder="/home/mschuster/projects/Synthetic_MRSI/config/",
                                         file_name="paths_14032025.json")
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

        # Load and prepare the concentration maps ################################################################ START
        # [X] Real data / computation is used
        # [ ] Placeholder data / computation is used
        # [ ] Something is missing here
        # TODO: Nothing yet
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
        concentration_maps.interpolate_to_target_size(target_size=metabolic_mask.shape,
                                                      order=3,
                                                      target_device='cuda',
                                                      target_gpu=target_gpu_smaller_tasks)
        ############################################################################################################ END


        # Load and prepare the T1 maps ########################################################################### START
        # [ ] Real data / computation is used
        # [ ] Placeholder data / computation is used
        # [X] Something is missing here
        # TODO: Same T1 map for each metabolite used
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
        t1_maps.interpolate_to_target_size(target_size=metabolic_mask.shape,
                                           order=3,
                                           target_device='cuda',
                                           target_gpu=target_gpu_smaller_tasks)
        ############################################################################################################ END


        # Load and prepare the T1 maps ########################################################################### START
        # [ ] Real data / computation is used
        # [X] Placeholder data / computation is used
        # [X] Something is missing here
        # TODO: Only placeholder values for T2. Need also for each metabolite map?

        # Using T2* values from:
        cm.cite("peters2007t2") # Paper offers Grey Matter (GM), White matter (WM), Caudate, Putamen T2* values in [ms];
                                # However, using naive approach for first step: (GM+WM)/2 and also the scattering of values
                                # (33.2+26.8)/2 +/- (1.3+1.2)/2 ==> 30 +/- 1.25

        mu=30
        sigma=1.25

        working_name_and_map = {"Glu":      np.random.normal(loc=mu, scale=sigma, size=metabolic_mask.data.shape),
                                "Gln":      np.random.normal(loc=mu, scale=sigma, size=metabolic_mask.data.shape),
                                "m-Ins":    np.random.normal(loc=mu, scale=sigma, size=metabolic_mask.data.shape),
                                "NAA":      np.random.normal(loc=mu, scale=sigma, size=metabolic_mask.data.shape),
                                "Cr+PCr":   np.random.normal(loc=mu, scale=sigma, size=metabolic_mask.data.shape),
                                "GPC+PCh":  np.random.normal(loc=mu, scale=sigma, size=metabolic_mask.data.shape),
                                }

        t2_maps = Maps(maps=working_name_and_map)
        t2_maps.interpolate_to_target_size(target_size=metabolic_mask.shape,
                                           order=3,
                                           target_device='cuda',
                                           target_gpu=target_gpu_smaller_tasks)
        ############################################################################################################ END


        # Get a subset of all signals of teh FID and merge some FID signals ###################################### START
        # [X] Real data / computation is used
        # [ ] Placeholder data / computation is used
        # [ ] Something is missing here
        # TODO: Nothing yet
        # Get partially FID containing only the desired signals
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
        fid.merge_signals(names=["Creatine (Cr)", "Phosphocreatine (PCr)"],
                          new_name="Creatine (Cr)+Phosphocreatine (PCr)",
                          divisor=2)
        fid.merge_signals(names=["Choline_moi(GPC)", "Glycerol_moi(GPC)", "PhosphorylCholine_new1 (PC)"],
                          new_name="GPC+PCh",
                          divisor=2)
        fid.name = fid.get_name_abbreviation()
        ############################################################################################################ END


        ##### TODO
        ####fid.signal = np.random.rand(6, 100_000) + 1j * np.random.rand(6, 100_000)
        ####fid.signal = fid.signal.astype(np.complex64)
        ####fid.time = np.arange(0, 100_000)
        ####Console.printf("warning", "Replaced loaded simulated FID signals of length 1536 to random values of length 100000")

        Console.printf_section("Assemble FID and Maps")
        block_size = (int(20), int(112), int(128), int(80)) # TODO: was best for 1536
        #block_size = (int(5), int(112 / 2), int(128 / 2), int(80 / 2)) # TODO: worked almost with 100_000 FID points
        # TODO: Print number of blocks used for the computation and how much memory it eats up per block


        # Get a subset of all signals of the FID and merge some FID signals ###################################### START
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
                                                      compute_on_device="cpu",
                                                      return_on_device="cpu") # TODO TODO TODO TODO

        # TODO: Also state size of one block after created model!
        spectral_spatial_model.add_metabolic_property_maps(metabolic_property_map_dict)  # all maps
        spectral_spatial_model.add_fid(fid)  # all fid signals  # TODO: change back to fid_prepared
        spectral_spatial_model.add_mask(metabolic_mask.data)

        spectral_spatial_model.model_summary()
        computational_graph = spectral_spatial_model.assemble_graph()

        cluster = tools.MyLocalCluster()
        #cluster.start_cuda(device_numbers=target_gpus_big_tasks, device_memory_limit="20GB", use_rmm_cupy_allocator=True)
        cluster.start_cpu(number_workers=2, threads_per_worker=10, memory_limit_per_worker="60GB")

        # Load and interpolate the coil sensitivity maps
        coil_sensitivity_maps_loader = file.CoilSensitivityMaps(configurator=configurator)
        coil_sensitivity_maps_loader.load_h5py(keys=["imag", "real"], dtype=np.complex64)
        target_size = (32, metabolic_mask.shape[0], metabolic_mask.shape[1], metabolic_mask.shape[2])
        coil_sensitivity_maps = coil_sensitivity_maps_loader.interpolate(target_size=target_size, order=3, compute_on_device='cuda', gpu_index=target_gpu_smaller_tasks, return_on_device='cpu') # TODO TODO TODO TODO


        sampling_model = SamplingModel(computational_graph_spectral_spatial_model=computational_graph,
                                       block_size_computational_graph_spectral_spatial_model=(10, 112, 128, 80),
                                       coil_sensitivity_maps=coil_sensitivity_maps,
                                       block_size_coil_sensitivity_maps=(10, 112, 128, 80),
                                       path_cache='/home/mschuster/projects/Synthetic_MRSI/cache/dask_tmp', # TODO: has no effect!
                                       persist_computational_graph_spectral_spatial_model=False
                                       )

        computational_graphs_list = sampling_model.apply_coil_sensitivity_maps(compute_on_device='cpu', # cuda work with 100_000
                                                                               return_on_device='cpu')   # cpu work with 100_000
        computational_graphs_list = sampling_model.cartesian_FFT(volumes_with_coil_sensitivity_maps=computational_graphs_list,
                                                                 crop_center_shape=(64, 64, 40),
                                                                 compute_on_device='cpu', # cuda work with 100_000
                                                                 return_on_device='cpu')   # cpu work with 100_000
        computational_graphs_list = sampling_model.cartesian_IFFT(volumes_cartesian_k_space=computational_graphs_list,
                                                                  compute_on_device='cpu',  # cuda work with 100_000
                                                                  return_on_device='cpu')    # cpu work with 100_000

        import time
        start_time = time.time()
        print("Storing data as HDF5 array with gzip compression...")
        with h5py.File('coils_cropped_cartesian_compressed.h5', 'w') as f:
            for i, dask_array in enumerate(computational_graphs_list):
                print(f"Computing and storing for coil {i} / {len(computational_graphs_list)}")
                # Create a dataset for each computed Dask array
                #f.create_dataset(f'coil_{i}', data=cp.asnumpy(dask_array.compute()), shape=dask_array.shape, chunks=dask_array.chunksize, compression="gzip", compression_opts=9)
                #f.create_dataset(f'coil_{i}', data=cp.asnumpy(dask_array.compute()), shape=dask_array.shape, chunks=dask_array.chunksize)
                f.create_dataset(f'coil_{i}',
                                 data=dask_array.compute(),
                                 shape=dask_array.shape,
                                 chunks=dask_array.chunksize)

        write_time_hdf5 = time.time() - start_time
        print(write_time_hdf5/60)

        input("--------!-------")

    ####################################################################################################################################
    sampling_model = SamplingModel(path_cache='/home/mschuster/projects/Synthetic_MRSI/cache/dask_tmp')

    target_gpu_smaller_tasks = 1
    target_gpus_big_tasks = [1]
    cluster = tools.MyLocalCluster()
    cluster.start_cuda(device_numbers=target_gpus_big_tasks, device_memory_limit="20GB", use_rmm_cupy_allocator=True)

    Console.start_timer()
    with h5py.File('coils_cropped_cartesian_compressed.h5', 'r') as f:
        # Create a list to hold the Dask arrays
        computational_graphs_list = []
        for i in tqdm(range(len(f.keys())), total=len(f.keys()), desc="Load h5 data from disk"): # Iterate over the datasets in the file
            dset = f[f'coil_{i}'][:]

            block_size = (int(20*8), int(64), int(64), int(40)) # TODO: Save also in H5-file!
            dask_array = da.from_array(dset, chunks=block_size)
            computational_graphs_list.append(dask_array)

    Console.stop_timer()

    Console.start_timer()
    computational_graphs_list = sampling_model.apply_gaussian_noise(volumes_with_coil_sensitivity_maps=computational_graphs_list,
                                                                    snr_desired=50,
                                                                    compute_on_device='cuda',
                                                                    return_on_device='cuda')

    volume = sampling_model.coil_combination(volumes_with_coil_sensitivity_maps=computational_graphs_list,
                                             compute_each_coil=True,   # use cpu if True (for compute_on_device, thus dask.array.compute() for each coil)
                                             compute_on_device='cuda', # cpu work with 100_000
                                             return_on_device='cpu')   # cpu work with 100_000



    with h5py.File('coils_cropped_combined_cartesian.h5', 'r') as f:
        f.create_dataset(f'data', data=volume, shape=volume.shape)


    # TODO: Integrate maybe in FID class
    dwelltime = fid.time[0] - fid.time[1]
    vecSize = len(fid.time)
    LarmorFreq = 297223042
    CenterAroundPPM = 4.65

    # Calculate bandwidth frequency
    bandwidth_frequency = 1 / dwelltime
    # Calculate step frequency
    step_frequency = bandwidth_frequency / vecSize
    # Generate frequency vector
    freq_vector = (np.ceil(vecSize / 2) - 1 - np.arange(vecSize)) * step_frequency
    # Calculate chemical shift vector
    chemshift_vector = 10 ** 6 * freq_vector / LarmorFreq + CenterAroundPPM

    plot_FID(signal=np.fft.fftshift(np.fft.fft(volume[:, 10, 10, 10])), time=chemshift_vector)

    Console.stop_timer()

    input("----------")




if __name__ == '__main__':
    dask.config.set(**{'array.slicing.split_large_chunks': False})
    main_entry()
    #config = Config(max_depth=4)
    #graphviz = GraphvizOutput(output_file='pycallgraph.png')
    #with PyCallGraph(output=GraphvizOutput(), config=config):
    #    main_entry()
