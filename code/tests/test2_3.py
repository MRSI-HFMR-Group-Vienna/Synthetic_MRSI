import multiprocessing

import numpy as np
import dask.array as da
from dataclasses import dataclass
from tqdm import tqdm
import dask
import time
from dask.diagnostics import ProgressBar, visualize
import xarray as xr
from dask.distributed import Client, LocalCluster, progress
dask.config.set(temporary_directory='/home/mschuster/projects/Synthetic_MRSI/cache/dask_tmp')
from sklearn.utils import gen_batches
import ctypes

# dask suggested it: to forcefully release allocated but unutilized memory
def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


# TODO: Bei TR kein TR*t?
def t1_transform(volume, alpha, TR, T1):
    # alpha ... 45°,
    # TR    ... 0.6sec

    # T1 -> volume
    return volume * np.sin(np.deg2rad(alpha)) * (1-da.exp(-TR/T1)) / (1-(da.cos(np.deg2rad(alpha)) * da.exp(-TR/T1))) # TODO: radiants

####def t2_transform(TE, T2, t):
####    # TE + delta_t (=dwell time) => thus: TE*t where t is time vector
####    # t ... time vector
####    # TE... 1.3ms e.g. TODO: transform to seconds
####
####    # T2 -> volume
####    # t  -> time vector
####    return np.exp(-(TE*t)/T2) #TODO <---- somit für jeden FID point extra

def transform_T2(volume, time_vector, T2, TE):
    for index, t in enumerate(time_vector):
        volume[index, :, :, :] = volume[index, :, :, :] * da.exp((TE*t)/T2)
        #volume[metabolite_number, index, :, :, :] = volume[metabolite_number, index, :, :, :] * np.exp((-TE * t) / T2)
    return volume

@dataclass
class MetabolicPropertyMap:
    t1: da.array
    t2: da.array
    concentration: da.array

    def get_multiplied_maps(self):
        return self.t1 * self.t2 * self.concentration


if __name__ == "__main__":
    """
    (0) The possibility to create/show a dashboard (TODO: need free forwarded port. E.g., 7799)
    """

    dask.config.set({"distributed.comm.timeouts.tcp": "50s"})
    #cluster = LocalCluster(dashboard_address=':7799')
    memory_limit_per_worker = "40GB"
    client = Client()
    client.close()

    # Create a LocalCluster
    # for numerical operation better less workers, but more threads: https://stackoverflow.com/questions/49406987/how-do-we-choose-nthreads-and-nprocs-per-worker-in-dask-distributed
    # for strings more workers and less threads!
    cluster = LocalCluster(n_workers=5, threads_per_worker=8, memory_limit=memory_limit_per_worker) #20,2 ==> 2, 20 good!
    client = Client(cluster)


    #client = Client(n_workers=8, threads_per_worker=1, memory_limit=memory_limit_per_worker)

    # Restart workers that exceed memory budget
    # Memory budget exceeded warning: This warning indicates that a worker has exceeded its memory budget and needs to be restarted.
    #client.restart(timeout="60s")  # Adjust timeout as needed
    # Restart workers if connection reset by peer
    #client.restart()

    # Access the dashboard URL
    dashboard_url = client.dashboard_link
    print(dashboard_url)

    # Get the number of CPU cores available
    # Get the number of CPU cores available

    # Create a Dask cluster with maximum available workers
    # client = Client(n_workers=num_cores)
    num_cores = multiprocessing.cpu_count()
    print(f"Number of cores available: {num_cores}")
    input("-----")

    complex_dtype = np.complex64

    """
    (1) Create metabolic Property Maps with random values
    """
    # Define the shape
    shape = (150, 150, 150)
    # Define chuck size
    chunks = (8, 8, 8)

    metabolic_property_maps: dict = {}

    metabolites_name = [
        "Metabolite 1",
        "Metabolite 2",
        "Metabolite 3",
        "Metabolite 4",
        "Metabolite 5",
        "Metabolite 6",
        "Metabolite 7",
        "Metabolite 8",
        "Metabolite 9",
        "Metabolite 10",
        'Metabolite 11']

    for i in range(11):
        # Generate random values
        map_t1 = np.random.random(shape)
        map_t2 = np.random.random(shape)
        map_B0 = np.random.random(shape)

        map_t1 = map_t1.astype(np.float64)  # Maybe faster if converted to complex? E.g., 5 ====> 5+0j
        map_t2 = map_t2.astype(np.float64)
        map_B0 = map_B0.astype(np.float64)

        # Create dask arrays from numpy
        map_t1_dask = da.from_array(map_t1, chunks=chunks)
        map_t2_dask = da.from_array(map_t2, chunks=chunks)
        map_B0_dask = da.from_array(map_B0, chunks=chunks)

        metabolic_property_map_dataset = MetabolicPropertyMap(t1=map_t1_dask, t2=map_t2_dask, concentration=map_B0_dask)
        metabolic_property_maps[metabolites_name[i]] = metabolic_property_map_dataset

    """
    (2) Create FID with ascending values & also mask
    """
    # mask:
    mask_random = np.random.randint(low=0, high=2, size=shape)  # matrix of 0s and 1s
    mask_random_dask = da.from_array(mask_random, chunks=chunks)

    # FIDs:
    shape = 100_000
    ascending_fid_dask_dict: dict = {}
    for i in range(11):
        # Create FID with shape 100_000,1,1,1
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):  # to allow large chunks & silent warning
            ascending_fid_numpy = np.arange(0, shape).reshape(shape, 1, 1, 1).astype(np.complex64)
            ascending_fid_dask = da.from_array(ascending_fid_numpy)
            ascending_fid_dask_dict[metabolites_name[i]] = ascending_fid_dask

    array_list = []
    for i in range(11):
        # Multiplication => FID * T1 * T2 * concentration ==> implement exp func
        # Metabolic Property Maps shape: 150,150,150
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):  # to allow large chunks & silent warning

            # (1) FID with 3D mask volume yields 4d volume
            volume_with_mask = ascending_fid_dask_dict[metabolites_name[i]] * mask_random_dask

            # (2) Include T2 effects
            #metabolite_number = i
            time_vector = np.arange(0,100_000)/1000
            T2 = metabolic_property_maps[metabolites_name[i]].t2
            TE = 0.0013
            volume_with_T2 = da.map_blocks(transform_T2, volume_with_mask, time_vector, T2, TE)

            # (3) Include T1 effects
            alpha = 45
            TR = 0.6
            T1 = metabolic_property_maps[metabolites_name[i]].t1
            volume = da.map_blocks(t1_transform, volume_with_T2, alpha, TR, T1)

            # (3) Include concentration
            volume *= metabolic_property_maps[metabolites_name[i]].concentration

            # TODO: also include T1 and concentration
            #volume = ascending_fid_dask_dict[metabolites_name[i]] * metabolic_property_maps[metabolites_name[i]].get_multiplied_maps() * mask_random_dask


            volume = da.expand_dims(volume, axis=0)
            array_list.append(volume)

    # Put all arrays together -> 1,100_000,150,150,150 + ... + 1,100_000,150,150,150 = 11,100_000,150,150,150
    concatenated_array = da.concatenate(array_list, axis=0)

    print("====================================== Whole volume (all metabolites separately):")
    print(xr.DataArray(concatenated_array))

    # concatenated_array.dask.visualize(filename='main_array.png')
    # print(xr.DataArray(concatenated_array.blocks.ravel()[0]))
    # input("----")
    ### Takes about 7 sec
    # time_previous = time.time()
    # print(xr.DataArray(concatenated_array.blocks.ravel()[0]).compute())
    # print(f"time passed: {time.time() - time_previous} sec")

    """
    (3) Sum all 11 metabolites -> 11,100_000,... ==> 1,100_000,...
    """
    # It seems to be the size of the biggest chuck is necessary (defines the bottleneck of RAM)
    # ----> 11 * 100000 * 20 * 20 * 20 * 16 / (1024 * 1024 * 1024) = ~131GB (with 8 bit takes only about 10 hours)
    # ----> 11 * 100000 * 25 * 25 * 20 * 16 / (1024 * 1024 * 1024) = ~205GB => KILLED
    # ----> 11 * 100000 * 30 * 30 * 30 * 8  / (1024 * 1024 * 1024) = ~222GB
    # ----> 11 * 100000 * 20 * 20 * 20 * 8 / (1024 * 1024 * 1024) = ~66GB (with num workers = 10, takes about 5 hours)
    #     |---> doesn't make huge difference between num workers 10 & 20 & 40 --> about 4h40min
    sum_array = da.sum(concatenated_array, axis=0)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):  # to allow large chunks & silent warning
        sum_array = sum_array.rechunk((100_000, 8, 8, 8)) #(100000, 20, 20, 20)

    # print dask array as xarray to get more informations
    print("====================================== Whole volume (all metabolites summed up):")
    print(xr.DataArray(sum_array))
    # visualize the graph (only high level!)
    sum_array.dask.visualize(filename='main_array.png')

    #with ProgressBar():
    #    sum_array.compute()
    #input("-------------------")

    """
    Example #1 -> calculate one FID
    """
    ### calculate one FID ==> took about 114 sec
    ##time_previous = time.time()
    ##with ProgressBar():
    ##    print(sum_array[:, 0, 0, 0].compute(num_workers=4))
    ##print(f"time passed: {time.time() - time_previous} sec")
    ##input("-------- calculate one FID")

    """
    Example #2 -> calculate one Block
    """
    ### calculate one block of shape 100_000,20,20,20 ==> took about 102 sec
    ##time_previous = time.time()
    ##with ProgressBar():
    ##    print(sum_array[:, 0, 0, 0].compute(num_workers=4))
    ##print(f"time passed: {time.time() - time_previous} sec")
    ##input("-------- calculate one Block")

    """
    (4) Compute each block (one after another)
    """
    all_blocks = sum_array.blocks.ravel()

    print("====================================== A chunk:")
    print(xr.DataArray(all_blocks[0]))

    ###print("====================================== Compute all chunks:")
    #### Took about 6 min
    ###for b in tqdm(all_blocks):
    ###    #with ProgressBar():
    ###    b.compute(num_workers=num_cores)

    ## Took about 3 min with fid length 800
    ## Took about 35 min with fid length 8_000
    print("====================================== Compute chunks in parallel:")
    blocks = num_cores
    batches_slice = list(gen_batches(len(all_blocks), blocks))
    print(f"Number of batches {len(batches_slice)}, each {blocks} blocks")
    for i, s in enumerate(batches_slice):
        with ProgressBar():
            #print(client.scheduler_info())
            #big_future = client.scatter(all_blocks[s])
            #future = client.submit(dask.compute, all_blocks[s])
            #future.result()
            #client.submit(dask.compute, all_blocks[s])
            time_previous = time.time()
            client.run(trim_memory)
            dask.compute(all_blocks[s])
            client.run(trim_memory)

        print(f"Estimated time for one block {(time.time() - time_previous)/60} min")
        print(f"Estimated time for all {len(all_blocks)} set of blocks: {(time.time() - time_previous)/3600 * len(batches_slice)} hours")
        print(f"One set of blocks DONE ({i}/{len(batches_slice)})")

    client.close()
    print("THE END")
    ##### takes about 18 sec
    ####import time
    ####time_previous = time.time()
    ####print(concatenated_array[0,:,0,0,0].compute())
    ####print(f"time passed: {time.time() - time_previous} sec")

   # ==> 800, 1, 1, 1  * 1, 150, 150, 150 ===> 800, 150, 150, 150
   # ==>