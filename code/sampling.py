from spectral_spatial_simulation import Model as SpectralSpatialModel
import cupy as cp
import numpy as np


class Model:
    def __init__(self, spectral_spatial_model: SpectralSpatialModel, data_type: str = "complex64", compute_on_device: str = "cpu", return_on_device: str = "cpu"):

        # Input only dask graph or also numpy, cupy array?
        self.spectral_spatial_model: SpectralSpatialModel = spectral_spatial_model
        self.computational_graph = self.spectral_spatial_model.assemble_graph()
        self.compute_on_device = compute_on_device
        self.return_on_device = return_on_device

        if self.compute_on_device == "cuda":
            self.xp = cp  # use cupy
        elif self.compute_on_device == "cpu":
            self.xp = np  # use numpy

    def cartesian_FFT_graph(self):
        computational_graph = self.computational_graph.map_blocks(self.xp.fft.fftn, dtype=self.xp.complex64, axes=(1, 2, 3))

        if self.compute_on_device == "cuda" and self.return_on_device == "cpu":
            computational_graph = self.computational_graph.map_blocks(cp.asnumpy)
        elif self.compute_on_device == "cpu" and self.return_on_device == "cuda":
            computational_graph = self.computational_graph.map_blocks(cp.asarray)

        return computational_graph

