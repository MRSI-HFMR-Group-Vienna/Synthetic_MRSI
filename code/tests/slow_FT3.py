import numpy as np
import cupy as cp
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

# Function to compute NDFT for a specific point using CuPy on GPU
def nudft_point_gpu(image, t, x, y, z):
    image_gpu = cp.asarray(image[t])
    exponent_gpu = -1j * 2 * cp.pi * ((x * cp.arange(x_length)[:, None, None] / x_length) +
                                      (y * cp.arange(y_length)[None, :, None] / y_length) +
                                      (z * cp.arange(z_length)[None, None, :] / z_length))
    k_space_one_voxel_gpu = cp.sum(image_gpu * cp.exp(exponent_gpu), axis=(0, 1, 2))
    return k_space_one_voxel_gpu.get()

# Main function to handle the Dask computation
def main(image, points):
    t_length, x_length, y_length, z_length = image.shape

    # Create Dask delayed objects for each point
    delayed_results = []
    for i, (t, x, y, z) in tqdm(enumerate(points)):
        delayed_result = delayed(nudft_point_gpu)(image, t, x, y, z)
        delayed_results.append(delayed_result)
        if i >= 100_000:
            break

    # Compute results in parallel
    k_space_results = compute(*delayed_results)

    return k_space_results

if __name__ == '__main__':
    # Initialize Dask CUDA cluster
    cluster = LocalCUDACluster()
    client = Client(cluster)

    # Define points of interest (list of tuples: (t, x, y, z))
    points_of_interest = [
        (0, 75, 75, 75),
        (0, 4, 75, 3),
        # Add more points as needed
    ]

    points_of_interest = []

    for z in tqdm(range(150)):
        for y in range(150):
            for x in range(150):
                for t in range(20):
                    points_of_interest.append((t,x,y,z))

    # Compute the k-space representation for specific points
    k_space = main(image, points_of_interest)
    print("k-space results:", k_space)