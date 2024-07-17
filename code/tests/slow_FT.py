import matplotlib.pyplot as plt
import dask.array as da
import numpy as np
import cupy as cp
import time
from tqdm import tqdm
from dask import delayed, compute
from dask.distributed import Client, LocalCluster

#### synthetic 3D Gaussian volume (non-parametric, thus just -x²)
###x_length, y_length, z_length = 150, 150, 150
###x = np.linspace(-4, 4, x_length, dtype=np.complex64)  # 150 datapoints from -4 to 4
###y = np.linspace(-4, 4, y_length, dtype=np.complex64)  # same
###z = np.linspace(-4, 4, z_length, dtype=np.complex64)  # same
###x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
###image = np.exp(-(x_grid ** 2 + y_grid ** 2 + z_grid ** 2))
###
#### Create cross
###image[75 - 10:75 + 10, 75 - 10:75 + 10, 75 - 10:75 + 10] = 0
###image[75 - 10:75 + 10, 75 - 1:75 + 1, 75 - 1:75 + 1] = 0.5
###image[75 - 1:75 + 1, 75 - 10:75 + 10, 75 - 1:75 + 1] = 0.5
###image[75 - 10:75 + 10, 75 - 1:75 + 1, 75 - 10:75 + 10] = 0.5
###
#### Outside of radius 0
###radius = 2
###distances = np.sqrt(x_grid ** 2 + y_grid ** 2 + z_grid ** 2)  # Calculate the distance from the center for each voxel
###mask = distances <= radius
###image[~mask] = 0
###
###image = image * np.ones((20, 1, 1, 1))  # --> wie skaliert es?
###print(image.dtype)
###
#### plt.imshow(image[:,:,75])
#### plt.show()
###
#### Cartesian fft CPU
###time_previous = time.time()
###fft_cpu = np.fft.fftn(image, axes=(1, 2, 3))
###print(time.time() - time_previous)
###
#### Cartesian fft GPU
###time_previous = time.time()
###image_gpu = cp.asarray(image)
###fft_gpu = cp.fft.fftn(image_gpu, axes=(1, 2, 3))
###print(time.time() - time_previous)
###
#### Non-uniform cartesian FT CPU
###t_length, x_length, y_length, z_length = image.shape
###k_space = np.zeros(image.shape, dtype=np.complex128)
###
###if False is True:
###    for t in tqdm(range(t_length)):
###        for z in range(z_length):
###            for y in range(y_length):
###                for x in tqdm(range(x_length)):
###                    for kz in range(z_length):
###                        for ky in range(y_length):
###                            for kx in range(x_length):
###                                if image[t, x, y, z] != 0:
###                                    k_space[t, kx, ky, kz] = image[t, x, y, z] * np.exp(-1j * 2 * np.pi * (x * kx / x_length)) * np.exp(
###                                        -1j * 2 * np.pi * (y * ky / y_length)) * np.exp(-1j * 2 * np.pi * (z * kz / y_length))
###
#### Create the meshgrid for vectorized computation
###x_indices = np.arange(x_length)
###y_indices = np.arange(y_length)
###z_indices = np.arange(y_length)
###x_grid, y_grid, z_grid = np.meshgrid(x_indices, y_indices, z_indices)
###kx_grid, ky_grid, kz_grid = np.meshgrid(x_indices, y_indices, z_indices)
###
###if False is True:
###    for t in tqdm(range(t_length)):
###        for z in range(z_length):
###            for y in range(y_length):
###                for x in range(x_length):
###                    exponent = -1j * 2 * np.pi * ((kx_grid * x / x_length) + (ky_grid * y / y_length) + (kz_grid * z / z_length))
###                    k_space[t, :, :, :] = np.sum(image[t, :, :, :] * np.exp(exponent))
###
#### -> check if correct!
###
#### Non-uniform cartesian FT GPU
#### -> check if correct!
###
###kx_grid_gpu, ky_grid_gpu, kz_grid_gpu = cp.asarray(kx_grid), cp.asarray(ky_grid), cp.asarray(kz_grid)
#### x_length, y_length, z_length = cp.asarray(x_length), cp.asarray(y_length), cp.asarray(z_length)
###k_space_gpu = cp.zeros(image.shape, dtype=np.complex128)
###
###if False is True:
###    for t in tqdm(range(t_length)):
###        for z in range(z_length):
###            for y in range(y_length):
###                for x in range(x_length):
###                    exponent = -1j * 2 * cp.pi * ((kx_grid_gpu * x / x_length) + (ky_grid_gpu * y / y_length) + (kz_grid_gpu * z / z_length))
###                    k_space_gpu[t, :, :, :] = cp.sum(image_gpu[t, :, :, :] * cp.exp(exponent))
###

# Non-uniform cartesian FT CPU+dask
print("===================START LOCAL CLUSTER=====================")
cluster = LocalCluster()
LocalCluster(n_workers=4,threads_per_worker=8,memory_limit="30GB")
client = Client(cluster)

@delayed
def nudft(image, t, x, y, z, x_length, y_length, z_length, kx_grid, ky_grid, kz_grid):
    # image on cpu

    exponent = -1j * 2 * np.pi * ((kx_grid * x / x_length) + (ky_grid * y / y_length) + (kz_grid * z / z_length))
    k_space_one_voxel = np.sum(image[t, :, :, :] * np.exp(exponent))

    return k_space_one_voxel


if __name__ == "__main__":
    # Create data to work with:
    # synthetic 3D Gaussian volume (non-parametric, thus just -x²)
    x_length, y_length, z_length = 150, 150, 150
    x = np.linspace(-4, 4, x_length, dtype=np.complex64)  # 150 datapoints from -4 to 4
    y = np.linspace(-4, 4, y_length, dtype=np.complex64)  # same
    z = np.linspace(-4, 4, z_length, dtype=np.complex64)  # same
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
    image = np.exp(-(x_grid ** 2 + y_grid ** 2 + z_grid ** 2))

    # Create cross
    image[75 - 10:75 + 10, 75 - 10:75 + 10, 75 - 10:75 + 10] = 0
    image[75 - 10:75 + 10, 75 - 1:75 + 1, 75 - 1:75 + 1] = 0.5
    image[75 - 1:75 + 1, 75 - 10:75 + 10, 75 - 1:75 + 1] = 0.5
    image[75 - 10:75 + 10, 75 - 1:75 + 1, 75 - 10:75 + 10] = 0.5

    # Outside of radius 0
    radius = 2
    distances = np.sqrt(x_grid ** 2 + y_grid ** 2 + z_grid ** 2)  # Calculate the distance from the center for each voxel
    mask = distances <= radius
    image[~mask] = 0

    image = image * np.ones((20, 1, 1, 1))  # --> wie skaliert es?

    # plt.imshow(image[:,:,75])
    # plt.show()

    # Create the meshgrid for vectorized computation
    x_indices = np.arange(x_length)
    y_indices = np.arange(y_length)
    z_indices = np.arange(y_length)
    kx_grid, ky_grid, kz_grid = np.meshgrid(x_indices, y_indices, z_indices)

    k_space = np.zeros(image.shape, dtype=np.complex64)
    t_length, x_length, y_length, z_length = image.shape

    # Convert grids to Dask arrays
    kx_grid_dask = da.from_array(kx_grid, chunks=(x_length, y_length, z_length))
    ky_grid_dask = da.from_array(ky_grid, chunks=(x_length, y_length, z_length))
    kz_grid_dask = da.from_array(kz_grid, chunks=(x_length, y_length, z_length))

    x_indices = np.arange(x_length)
    y_indices = np.arange(y_length)
    z_indices = np.arange(y_length)

    k_spaces_tasks = []

    for t in tqdm(range(t_length)):
        for z in range(z_length):
            for y in range(y_length):
                for x in tqdm(range(x_length)):
                    # k_space[t,:,:,:] =
                    k_space_task = nudft(image, t, x, y, z, x_length, y_length, z_length, kx_grid, ky_grid, kz_grid)
                    k_spaces_tasks.append(k_space_task)

    results = compute(*k_spaces_tasks)
    k_space = np.array(results).reshape((t_length, x_length, y_length, z_length))

    # Non-uniform cartesian FT GPU+dask
    # -> check if correct!
