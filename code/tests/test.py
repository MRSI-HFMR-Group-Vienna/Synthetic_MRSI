import numpy as np
import dask.array as da
import xarray as xr
import os
from dask.diagnostics import ProgressBar

from dask.distributed import Client
import multiprocessing

# Create a Dask cluster with multiple workers


if __name__ == "__main__":
    # Get the number of CPU cores available
    # Get the number of CPU cores available
    num_cores = multiprocessing.cpu_count()

    # Create a Dask cluster with maximum available workers
    client = Client(n_workers=num_cores)

    # Define the shape
    shape = (150, 150, 150)
    # Define chuck size
    chunks = (20, 20, 20)

    metabolic_property_maps: dict = {}
    metabolic_property_maps_:list = []

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

        # Create dask arrays from numpy
        map_t1_dask = da.from_array(map_t1, chunks=chunks)
        map_t2_dask = da.from_array(map_t2, chunks=chunks)
        map_B0_dask = da.from_array(map_B0, chunks=chunks)

        # Create xarrays from dask arrays
        map_t1_xarray = xr.DataArray(map_t1_dask, dims=('x', 'y', 'z'), coords={'x': np.arange(0, 150), 'y': np.arange(0, 150), 'z': np.arange(0, 150)})
        map_t2_xarray = xr.DataArray(map_t2_dask, dims=('x', 'y', 'z'), coords={'x': np.arange(0, 150), 'y': np.arange(0, 150), 'z': np.arange(0, 150)})
        map_B0_xarray = xr.DataArray(map_B0_dask, dims=('x', 'y', 'z'), coords={'x': np.arange(0, 150), 'y': np.arange(0, 150), 'z': np.arange(0, 150)})

        # Create xarray dataset from xarrays (can I specify here the axes name and coordinates for all arrays at once?)
        metabolic_property_map_dataset = xr.Dataset({
            't1': map_t1_xarray,
            't2': map_t2_xarray,
            'B0': map_B0_xarray
        })

        # xarray dataset
        #metabolic_property_maps[metabolites_name[i]] = metabolic_property_map_dataset
        metabolic_property_maps_.append(metabolic_property_map_dataset)

    #####Create memmap with certain shape#####
    # desired shape
    shape = (100_000, 150, 150, 150)

    # complex data with 0j as values
    dtype = np.cdouble
    filename = 'memmap_test_array.npy'
    memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)

    # get theoretical required fileszize
    file_size = os.path.getsize('memmap_test_array.npy') / (1024 * 1024)
    print("Theoretical size of file on disk: ~", round(file_size), "MB")

    #####Re-open memmap & combine with metabolic property maps#####
    shape = (100_000, 150, 150, 150)
    dtype = np.cdouble
    memmap_array = np.memmap('memmap_test_array_2.npy', dtype=dtype, mode='r+', shape=shape)

    chunks = (100_000, 20, 20, 20)
    memmap_array_dask = da.from_array(memmap_array, chunks=chunks)
    memmap_array_xarray = xr.DataArray(memmap_array_dask, dims=('t', 'x', 'y', 'z'))
    print(memmap_array_xarray)

    ####Combine the FIDs with the memmap####
    shape = 100_000
    ascending_fid_numpy_list = []
    for i in range(11):
        ascending_fid_numpy = np.arange(0, shape).astype(np.complex128)
        ascending_fid_numpy_list.append(ascending_fid_numpy)

    ascending_fid_dask_list = []
    for i in range(11):
        ascending_fid_dask = da.from_array(ascending_fid_numpy_list[i])
        ascending_fid_dask_list.append(ascending_fid_dask)

    ascending_fid_xarray_list = []
    for i in range(11):
        ascending_fid_xarray = xr.DataArray(ascending_fid_dask_list[i], dims=('t'))
        ascending_fid_xarray_list.append(ascending_fid_xarray)

    # Add dimension
    memmap_array_xarray = memmap_array_xarray.expand_dims(dim={"m": 11})


    import time
    from tqdm import tqdm
    time_previous = time.time()
    # FOR ASSIGNING VALUES INCREASE THE CHUNKSIZE! FOR ACCESSIG IT DECREASE IT!
    for i in tqdm(range(11)):
        # memmap_array_xarray[i,:,:,:,:] = ascending_fid_xarray_list[i].compute()
        # print(ascending_fid_xarray_list[i].expand_dims({'x':1, 'y':1,'z':1}).transpose('t', 'x', 'y', 'z').shape)
        # print(memmap_array_xarray[i,:,:,:,:].shape)

        L = ascending_fid_xarray_list[i].expand_dims({'x': 1, 'y': 1, 'z': 1}).transpose('t', 'x', 'y', 'z')

        #metabolic_property_map = metabolic_property_maps_[i].t1 * metabolic_property_maps_[i].t2 * metabolic_property_maps_[i].B0
        # TODO ==> put this down?
        metabolic_property_map = metabolic_property_maps_[i].t1 * metabolic_property_maps_[i].t2 * metabolic_property_maps_[i].B0

        #memmap_array_xarray[i, :, :, :, :] = ascending_fid_xarray_list[i].expand_dims({'x': 1, 'y': 1, 'z': 1}).transpose('t', 'x', 'y', 'z') * metabolic_property_map
        ascending_fid_xarray_one_item = ascending_fid_xarray_list[i].expand_dims({'x': 1, 'y': 1, 'z': 1}).transpose('t', 'x', 'y', 'z')
        print(ascending_fid_xarray_one_item) #* metabolic_property_map
        print("+++++++++++++++++++++++++++")
        metabolic_property_map_with_t_dim = metabolic_property_map.expand_dims({'t': 1})
        print(metabolic_property_map_with_t_dim)
        print("+++++++++++++++++++++++++++")
        print(ascending_fid_xarray_one_item * metabolic_property_map_with_t_dim)
        print("+++++++++++++++++++++++++++")
        input("???")






    ###########just for test purposes get certain value in array####
    ######## This took 67 sec!
    ########with ProgressBar():
    ########    print(memmap_array_xarray.isel(m=0, t=np.arange(0,100_000), x=0, y=0, z=0).compute())
#######
#######
    ###########Multiply with the maps####
    ########for i in range(11):
    #######print("A--------------------------")
    #######print(metabolic_property_maps_[0].prod(dim=None))
#######
    #######print("B--------------------------")
#######
    ########print(P.isel(x=slice(0, 20), y=slice(0, 20), z=slice(0, 20)))
#######
    ######## THIS TAKES ABOUT
    #######for i in tqdm(range(11)):
    #######    print("---------------start")
    #######    metabolic_property_map = metabolic_property_maps_[i].t1 * metabolic_property_maps_[i].t2 * metabolic_property_maps_[i].B0
#######
    #######    # this brings all values from x,y,z => to x,y,z,t. Thus e.g., x=0,y=0,z=0 --> 10 ==> x=0,y=0,z=0,t=: --> (10,10,10, ..., 10)
    #######    ###metabolic_property_map = metabolic_property_map.expand_dims(dim={'t':100000})
    #######    ###print(metabolic_property_map)
    #######    ###print(memmap_array_xarray)
#######
    #######    print("--->")
    #######    ###memmap_array_xarray[i, :, :, :, :] = memmap_array_xarray.isel(m=i) * metabolic_property_map ### only multiply one value 1* 10000
    #######    ####print()
#######
    #######    memmap_array_xarray[i, :, :, :, :] = memmap_array_xarray.isel(m=i, t=slice(0,100_000)) * metabolic_property_map  ### only multiply one value 1* 10000
    #######    print("--------------")

    print(memmap_array_xarray)