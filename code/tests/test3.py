import numpy as np
import dask.array as da
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler, visualize

if __name__ == "__main__":
    # Generate random binary values (0 or 1) for a volume of 150x150x150
    random_values = np.random.randint(2, size=(150, 150, 150))

    # Create Dask array from the generated random values
    dask_array = da.from_array(random_values, chunks=(50, 50, 50))  # Chunks can be adjusted based on your computation needs

    # Define your computation
    mean_result = dask_array.mean()

    # Profile your computation
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
        result = mean_result.compute()

    # Visualize profiling results in the console
    visualize([prof, rprof, cprof], show=False)