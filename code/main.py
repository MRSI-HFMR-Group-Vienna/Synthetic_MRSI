import os.path

import torch

import default

from spatial_metabolic_distribution import MetabolicPropertyMap
from dask.distributed import Client, LocalCluster
from distributed.diagnostics import MemorySampler
from dask.diagnostics import ProgressBar
import spectral_spatial_simulation
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from printer import Console
import numpy as np
import pint
import file
import sys

import seaborn as sns
import matplotlib.pyplot as plt
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
    # Create an empty DataFrame
    df = pd.read_csv('performance/data.txt', delimiter='\t', header=0)

    # Filter out rows with time_sec == -1
    filtered_df = df[df['time_sec'] != -1]
    # Plotting with time_sec on the y-axis
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='num_worker', y='num_threads', hue='time_sec', palette='viridis',
                    size='time_sec', sizes=(5, 500), legend='brief')
    plt.title('Number of Workers vs. Number of Threads')
    plt.xlabel('Number of Workers')
    plt.ylabel('Number of Threads')
    plt.legend(title='Time (sec)')
    plt.show()
    ####input("END!")
    Console.ask_user("Continue?")


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

    #### Load B0 and B1 maps & Interpolate them based on the mask shape
    ###loaded_B0 = file.NeuroImage(path=configurator.data["path"]["maps"]["B0"]).load_nii().data
    ###loaded_B1 = file.NeuroImage(path=configurator.data["path"]["maps"]["B1"]).load_nii().data
    ###
    ###zoom_factor_B0 = np.divide(metabolic_mask.shape, loaded_B0.shape)
    ###zoom_factor_B1 = np.divide(metabolic_mask.shape, loaded_B1.shape)

    ##loaded_B0_interpolated = zoom(input=loaded_B0, zoom=zoom_factor_B0, order=3)
    ##loaded_B1_interpolated = zoom(input=loaded_B1, zoom=zoom_factor_B1, order=3)
    ##Console.printf("success", f"Interpolated B0: from {loaded_B0.shape} to --> {loaded_B0_interpolated.shape}")
    ##Console.printf("success", f"Interpolated B1: from {loaded_B1.shape} to --> {loaded_B1_interpolated.shape}")




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

    # Create MetabolicPropertyMap (theoretically for each metabolite; however, only Glu used so far)
    metabolic_property_map_dict = {}
    for fid, concentration_map in fid_and_concentration:
        #concentration_map = loaded_metabolic_map_interpolated
        t1_map = loaded_T1_map_interpolated * 1e-3  # Assuming it is in [ms]
        t2_random_map = np.random.uniform(low=55 * 1e-3, high=70 * 1e-3, size=metabolic_mask.data.shape)

        metabolic_property_map = MetabolicPropertyMap(chemical_compound_name=fid.name[0],
                                                      block_size=(112/2, 128/2, 80/2), # 10,10,10
                                                      t1=t1_map,
                                                      t1_unit=u.ms,
                                                      t2=t2_random_map,
                                                      t2_unit=u.ms,
                                                      concentration=concentration_map,
                                                      concentration_unit=u.mmol)

        metabolic_property_map_dict[metabolic_property_map.chemical_compound_name] = metabolic_property_map

    # Create spectral spatial model
    simulation = spectral_spatial_simulation.Model(path_cache='/home/mschuster/projects/Synthetic_MRSI/cache/dask_tmp',
                                                   block_size=(1, 112/2, 128/2, 80/2), # TODO: was 1536x10x10x10
                                                   TE=0.0013,
                                                   TR=0.6,
                                                   alpha=45)
                                                    # TODO: Also state size of one block after created model!

    # Add components to the Model
    Console.ask_user("Create computational graph?")
    simulation.add_metabolic_property_maps(metabolic_property_map_dict)  # all maps

    ###fid_prepared_2 = spectral_spatial_simulation.FID()
    ###for fid in fid_prepared:
    ###    # Generate 100000 real and imaginary parts
    ###    real_parts = np.random.rand(100_000).astype(np.float32)
    ###    imaginary_parts = np.random.rand(100_000).astype(np.float32)
    ###    # Combine into complex64
    ###    complex_numbers = real_parts + 1j * imaginary_parts
    ###    complex_numbers = complex_numbers.astype(np.complex64)
    ###    fid.signal = complex_numbers
    ###    fid.time = np.arange(0,100_000)
    ###    fid_prepared_2 += fid

    simulation.add_fid(fid_prepared)  # all fid signals
    simulation.add_mask(metabolic_mask.data)

    computational_graph = simulation.assemble_graph()

    #computational_graph_one_metabolite.visualize(filename='dask_graph.png')
    computational_graph.dask.visualize(filename='visualisation/dask_graph_high_level.png')

    # Start client & and start dasboard, since computation is below
    memory_limit_per_worker = "20GB"
    #client = Client()
    #Client().close()
    # TODO Plot num worker and threads per worker (in sum 40) and time taken!
    n_workers = [10,9,8,7,6,5,4]
    threads_per_worker = [1,2,3,4,5,6,7]

    n_workers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    threads_per_worker = [1,2,3,4,5,6,7,8,9,10]

    import time
    #import pandas as pd
    # Create an empty DataFrame
    df = pd.DataFrame(columns=['num_worker', 'num_threads', 'time_sec'])
    current_path = os.getcwd()     # Get the current working directory
    file_name = 'performance/data.txt'  # Specify the file name

    # Create the full file path
    file_path = os.path.join(current_path, file_name)

    #####for i in range(len(n_workers)-1):
    ####for n_workers_chosen in n_workers:
    ####    for threads_per_worker_chosen in threads_per_worker:
    ####        if (n_workers_chosen * threads_per_worker_chosen) >= 20:
    ####        #---> continue statement?
    ####            threads_per_worker_chosen = np.floor(20/n_workers_chosen)
    ####
    ####        #n_workers_chosen = n_workers[i]
    ####        #threads_per_worker_chosen = threads_per_worker[i]
    ####        time_previous = time.time()
    ####        cluster = LocalCluster(n_workers=n_workers_chosen, threads_per_worker=threads_per_worker_chosen, memory_limit=memory_limit_per_worker)  # 5,8
    ####        client = Client(cluster)
    ####        dashboard_url = client.dashboard_link
    ####        print(dashboard_url)
    ####        try:
    ####
    ####            transversal = np.abs(computational_graph_one_metabolite[100, :, :, 40].compute())
    ####            coronal = np.abs(computational_graph_one_metabolite[100, :, 50, :].compute())
    ####            sagittal = np.abs(computational_graph_one_metabolite[100, 50, :, :].compute())
    ####            duration_sec = time.time() - time_previous
    ####        except Exception as e:
    ####            print(f"An error occurred: {e}")
    ####            duration_sec = -1.0
    ####        finally:
    ####            df = df._append({'num_worker': n_workers_chosen, 'num_threads': threads_per_worker_chosen, 'time_sec': duration_sec}, ignore_index=True)
    ####            df.to_csv(file_path, sep='\t', index=False)
    ####            print(df)
    ####            try:
    ####                client.close()
    ####                cluster.close()
    ####            except Exception as e:
    ####                pass
    ####
    ####print(df)

    ####import seaborn as sns
    ####import matplotlib.pyplot as plt
    ##### Filter out rows with time_sec == -1
    ####filtered_df = df[df['time_sec'] != -1]
    ##### Plotting with time_sec on the y-axis
    ####plt.figure(figsize=(10, 6))
    ####sns.scatterplot(data=filtered_df, x='num_worker', y='num_threads', hue='time_sec', palette='viridis',
    ####                size='time_sec', sizes=(5, 500), legend='brief')
    ####plt.title('Number of Workers vs. Number of Threads')
    ####plt.xlabel('Number of Workers')
    ####plt.ylabel('Number of Threads')
    ####plt.legend(title='Time (sec)')
    ####plt.show()
    ########input("END!")

    cluster = LocalCluster(n_workers=1, threads_per_worker=5, memory_limit=memory_limit_per_worker)  # --> 5,8 fro cross sectional view;  --> 4,5 for whole 1536
    client = Client(cluster)
    dashboard_url = client.dashboard_link
    print(dashboard_url)

    # Display one slice per axis of time point 100
    Console.ask_user("Plot cross-sectional view?")
    Console.start_timer()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    # Plotting data on each subplot using imshow
    ms = MemorySampler()
    all_blocks = computational_graph.blocks.ravel()

    print(type(all_blocks))
    print(type(all_blocks[0]))
    print(len(all_blocks))
    print(all_blocks[0].shape)
    print("==================================")

    Console.start_timer()
    with ms.sample("collection 1"), ProgressBar():
        for block in all_blocks:
            block.compute()
    Console.stop_timer()

        #whole_data = computational_graph.compute()


    Console.stop_timer()
    ms.plot(align=True)
    plt.show()

    input("---")
    transversal = computational_graph[100, :, :, 40].compute()
    coronal = computational_graph[100, :, 50, :].compute()
    sagittal = computational_graph[100, 50, :, :].compute()

    #transversal = computational_graph[100, :, :, 40].compute()
    transversal = np.abs(transversal)
    im1 = axes[0].imshow(transversal)
    axes[0].set_title('transversal')
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    #coronal = computational_graph[100, :, 50, :].compute()
    coronal = np.abs(coronal)
    im2 = axes[1].imshow(coronal)
    axes[1].set_title('coronal')
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    #sagittal = computational_graph[100, 50, :, :].compute()
    sagittal = np.abs(sagittal)
    im3 = axes[2].imshow(sagittal)
    axes[2].set_title('sagittal')
    cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    Console.stop_timer()
    # Adding some space between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()

    client.close()

    #####metabolic_property_map = metabolic_property_map_1 * metabolic_property_map_2
####
##### TODO: Do not delete!
##### To get all the chunks (dask.array) in a list
##### metabolic_property_map.t2
####
####Console.start_timer()
####
####
####
####print(metabolic_property_map_1.chemical_compound_name)
####for i, m in enumerate(metabolic_property_map_1):
####
####    block_number = i
####    x,y,z = m.t1.get_global_index_in_sub_volume(main_volume_shape=(112, 128, 80), block_number=i, indices_sub_volume=(0,0,0))
####    print(m.t1.blocks[0][0,0,0].compute()) # block 0, just one block here when using __iter__
####    print(t1_random_map[x,y,z]) # coordinates in big block!
####
####
#####objects = dask.compute(*metabolic_property_maps)
####Console.stop_timer()
####
####sys.exit()
####input("~~~~~~~~~~~~+STOP+~~~~~~~~~~~~")
####
##### Create random distribution with values in the range [0.0, 1.0]
####
##### Load FID of metabolites
####metabolites = file.FID(configurator=configurator)  # ,
##### concentrations=np.asarray([1, 1, 4, 10, 1, 2.5, 1, 6, 12, 4, 1]),
##### t2_values=np.asarray([170, 131, 121, 105, 170, 105, 131, 170, 170, 121, 131]) / 1000)
####metabolites.load(fid_name="metabolites",
####                 signal_data_type=np.complex64)
####
####loaded_fid = metabolites.loaded_fid
##### set the unit with pint
##### loaded_fid.time *= u.ms
##### loaded_fid.signal *= u.dimensionless  # TODO
####
####print(loaded_fid)
####
##### plot_FID(signal=fid.signal[0], time=fid.time)
##### print(metabolites.loaded_fid.get_partially_FID([1,5]))
####
##### plot_FID(signal=loaded_fid.signal, time=loaded_fid.time)
####
##### Simulate volume with desired FID
##### path_cache = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/mschuster/SimulationMRSI/cache/'
####
##### path_cache = "/home/mschuster/projects/Synthetic_MRSI/cache/" # TODO: use configurator, or at least config file for it!!!! DOnt define it manually!
####configurator.load()  # to load data from json file
####path_cache = configurator.data['path']['cache']  # TODO: maybe input configurator to spectral and sampling model?
####spectral_model = spectral_spatial_simulation.Model(path_cache=path_cache, file_name_cache="25022024")
##### spectral_model.add_fid(metabolites.loaded_fid)
##### spectral_model.add_mask(metabolic_mask.data)
##### spectral_model.add_fid_scaling_map(random_scaling)
##### spectral_model.build()
####
##### loaded_fid.sum_all_signals()
####
####random_scaling_maps_list = []
####for name in loaded_fid.name:
####    random_scaling = np.random.uniform(low=0.0, high=1.0, size=metabolic_mask.data.shape)
####    scaling_map = MetabolicScalingMap(name=name, map=random_scaling, unit=None)
####    random_scaling_maps_list.append(scaling_map)
####
####spectral_model.add_fid(loaded_fid)
####spectral_model.add_mask(metabolic_mask.data)
####spectral_model.add_fid_scaling_maps(fid_scaling_maps=random_scaling_maps_list)  # add ScalingMap (for each metabolite)
####spectral_model.build()
####
####sampling_model = sampling.Model(spectral_model=spectral_model, sub_volume_shape=(14, 16, 10))  # for main volume 112, 128, 80
####sampling_model.cartesian_FT(file_name_cache="25022024")
####sampling_model.create_sub_volume_iterative()
##### TODO
##### sampling.cartesian_FT(spectral_model, auto_gpu=True, path_cache=path_cache, file_name_cache="25022024", custom_batch_size=200)


if __name__ == '__main__':
    config = Config(max_depth=4)
    graphviz = GraphvizOutput(output_file='pycallgraph.png')
    with PyCallGraph(output=GraphvizOutput(), config=config):
        main_entry()