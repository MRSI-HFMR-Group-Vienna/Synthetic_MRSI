import numpy as np
import cupy as cp
import dask.array as da
from dask import delayed, compute
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from tqdm import tqdm

# Create synthetic 3D Gaussian volume
x_length, y_length, z_length = 150, 150, 150
x = np.linspace(-4, 4, x_length, dtype=np.complex64)
y = np.linspace(-4, 4, y_length, dtype=np.complex64)
z = np.linspace(-4, 4, z_length, dtype=np.complex64)
x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
image = np.exp(-(x_grid ** 2 + y_grid ** 2 + z_grid ** 2))

# Create cross
image[75 - 10:75 + 10, 75 - 10:75 + 10, 75 - 10:75 + 10] = 0
image[75 - 10:75 + 10, 75 - 1:75 + 1, 75 - 1:75 + 1] = 0.5
image[75 - 1:75 + 1, 75 - 10:75 + 10, 75 - 1:75 + 1] = 0.5
image[75 - 10:75 + 10, 75 - 1:75 + 1, 75 - 10:75 + 10] = 0.5

# Mask outside of radius 2
radius = 2
distances = np.sqrt(x_grid ** 2 + y_grid ** 2 + z_grid ** 2)
mask = distances <= radius
image[~mask] = 0

# Extend image to have a time dimension
image = image * np.ones((20, 1, 1, 1), dtype=np.complex64)

# Function to compute NDFT using CuPy on GPU
def nudft_gpu(image_chunk, kx_grid, ky_grid, kz_grid, x_length, y_length, z_length):
    image_gpu = cp.asarray(image_chunk)
    exponent_gpu = -1j * 2 * cp.pi * ((kx_grid * cp.arange(x_length)[:, None, None] / x_length) +
                                      (ky_grid * cp.arange(y_length)[None, :, None] / y_length) +
                                      (kz_grid * cp.arange(z_length)[None, None, :] / z_length))
    k_space_one_voxel_gpu = cp.sum(image_gpu * cp.exp(exponent_gpu), axis=(0, 1, 2))
    return k_space_one_voxel_gpu.get()

# Main function to handle the Dask computation
def main(image):
    t_length, x_length, y_length, z_length = image.shape

    x_indices = np.arange(x_length)
    y_indices = np.arange(y_length)
    z_indices = np.arange(z_length)

    kx_grid, ky_grid, kz_grid = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')

    kx_grid_gpu = cp.asarray(kx_grid)
    ky_grid_gpu = cp.asarray(ky_grid)
    kz_grid_gpu = cp.asarray(kz_grid)

    k_space = np.zeros((t_length, x_length, y_length, z_length), dtype=np.complex64)

    # Create Dask delayed objects for each time step
    delayed_results = []
    for t in range(t_length):
        delayed_result = delayed(nudft_gpu)(image[t], kx_grid_gpu, ky_grid_gpu, kz_grid_gpu, x_length, y_length, z_length)
        delayed_results.append(delayed_result)

    # Compute results in parallel
    k_space_results = compute(*delayed_results)

    # Combine results back into a single array
    for t, result in enumerate(k_space_results):
        k_space[t, :, :, :] = result

    return k_space

if __name__ == '__main__':
    # Initialize Dask CUDA cluster
    cluster = LocalCUDACluster()
    client = Client(cluster)

    # Compute the k-space representation
    k_space = main(image)
    print("k-space shape:", k_space.shape)
