from matplotlib.pyplot import axes
from rmm import rmm_cupy_allocator
from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster
#from printer import Console
#import torch
import pint
import rmm
import sys

import warnings
import functools
from typing import Callable
from printer import Console

u = pint.UnitRegistry()  # for using units

import dask.array as da
import dask
import os
import math
import pint
import numpy as np
import cupy as cp
import bibtexparser
import socket
import time
import operator

# For CustomArray extend visualisation jupyter lab:
import html as _html
import json as _json

from dask.distributed import LocalCluster, Worker

# For JupyterPlotManager
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, TextBox, Button

# For overloading functions
from typing import overload

# for interpolation on cpu/gpu
from cupyx.scipy.ndimage import zoom as zoom_gpu
from scipy.ndimage import zoom as zoom_cpu

# for FFT and FFT shift
from scipy.fft import fftn as fftn_cpu, fftshift as fftshift_cpu # cpu case: using scipy instead of numpy since numpy automatically data type promotion (e.g., complex64 -> fft -> complex128)
from cupy.fft  import fftn as fftn_gpu, fftshift as fftshift_gpu # gpu case: for gpu directly cupy is used




class CitationManager:
    def __init__(self, bib_file_path):
        """
        Initialize the CitationManager by loading a BibTeX file.

        :param bib_file_path: Path to the BibTeX file.
        """
        with open(bib_file_path, encoding="utf-8") as bib_file:
            bib_database = bibtexparser.load(bib_file)
        # Create a dictionary mapping citation keys to BibTeX entries.
        self.entries = {entry['ID']: entry for entry in bib_database.entries}

    def cite(self, key):
        """
        Mimic LaTeX \\cite behavior by returning a formatted citation string.

        :param key: The BibTeX key to cite the entry.
        :return: A formatted citation string.
        """
        if key not in self.entries:
            raise Exception(f"Citation key '{key}' not found.")

        entry = self.entries[key]
        authors = entry.get("author", "Unknown authors")
        title = entry.get("title", "No title")
        year = entry.get("year", "n.d.")
        doi = entry.get("doi", None)
        url = entry.get("url", None)

        # Format the citation with improved alignment.
        citation = f"    {title}\n    {authors} ({year})."
        if doi:
            citation += f"\n    DOI: {doi}"
        elif url:
            citation += f"\n    URL: {url}"
        return citation

    def print_all_citations(self):
        """
        Print all citations loaded from the BibTeX file in a formatted style.
        """
        print("Bibliography entries ==========================================")
        for i, key in enumerate(self.entries):
            print(f"Source {i}:")
            print(f"    {key}: {self.cite(key)}\n")
        print("=============================================================")


class NamedAxesArray_NEW:
    """
    TODO: Might have an issue
    An array that builds on NumPy and CuPy and inherits its functionalities.
    The main extensions are:
        * Use labelled Axes: E.g., Axis1, Axis2 ... AxisN or X, Y, Z. Not restricted to 3D.
        * Ability to interpolate between values using get_value().
          Example: For a 3D array, you can query get_value(X=1.5, Y=1.3, Z=2.1).
        * No extrapolation; values outside the axis range cause an error.

    """

    def __init__(self, input_array: np.ndarray | cp.ndarray, axis_values, device="cpu"):
        """
        Initialize with NumPy or CuPy depending on `device` argument.
        """
        if not isinstance(input_array, (np.ndarray, cp.ndarray)):
            raise TypeError(f"Input array must be numpy.ndarray or cupy.ndarray, got {type(input_array)}")

        if device not in ["cpu", "gpu"]:
            raise ValueError(f"'device' must be 'cpu' or 'gpu', got {device}")

        self.device = device
        self.xp = self._get_backend()
        self.BaseArray = self._create_base_array_class(self.xp)
        self.array = self.BaseArray(input_array, axis_values, self.xp)

    def _get_backend(self):
        """Returns NumPy or CuPy as the backend."""
        return cp if self.device == "gpu" else np

    def _create_base_array_class(self, xp):
        """
        Dynamically defines BaseArray with NumPy or CuPy.
        """
        class BaseArray(xp.ndarray):
            """
            Custom array class with named axes and interpolation.
            """

            def __new__(cls, input_array, axis_values, xp):
                # Create a raw mutable array (plain xp array)
                raw = xp.asarray(input_array).copy()

                # Create the subclass view
                obj = raw.view(cls)

                # Store the axis metadata and interpolation settings
                obj.axis_values = {axis: xp.array(values) for axis, values in axis_values.items()}
                obj.interpolation_method = "linear"
                obj.xp = xp

                # Instead of self-referencing, store 'raw' in base_array
                # so we do not create an infinite reference cycle.
                obj.base_array = raw

                return obj

            def __array_finalize__(self, obj):
                """Ensure attributes persist when slicing."""
                if obj is None:
                    return
                self.axis_values = getattr(obj, 'axis_values', None)
                self.interpolation_method = getattr(obj, 'interpolation_method', "linear")
                self.xp = getattr(obj, 'xp', None)
                # Retain reference to the original raw array if present
                self.base_array = getattr(obj, 'base_array', None)

            def __array_wrap__(self, out_arr, context=None):
                """Ensure that operations return BaseArray instead of plain NumPy/CuPy array."""
                return np.ndarray.__array_wrap__(self, out_arr, context)

            def __setitem__(self, index, value):
                """Support normal NumPy/CuPy-style item assignment."""
                # Write directly into the underlying raw array
                self.base_array[index] = value

            def get_value(self, **axis_values):
                """Get interpolated value using axis labels."""
                if set(axis_values.keys()) != set(self.axis_values.keys()):
                    raise ValueError(f"Expected axes: {list(self.axis_values.keys())}, but got {list(axis_values.keys())}")

                # Build 'points' = list of 1D coordinate arrays along each axis
                points = [self.axis_values[axis] for axis in self.axis_values.keys()]
                # Build 'query_points' = M x N array of coordinate sets
                # Here, N = number of axes, M = number of query points
                query_points = self.xp.array([axis_values[axis] for axis in self.axis_values.keys()]).T

                # Use the correct interpolation function based on backend
                if self.xp.__name__ == "cupy":
                    from cupyx.scipy.interpolate import interpn
                else:
                    from scipy.interpolate import interpn

                interpolated_values = interpn(
                    points,
                    self,  # 'self' is our nD data
                    query_points,
                    method=self.interpolation_method
                )

                if interpolated_values.size > 1:
                    return interpolated_values
                else:
                    return interpolated_values.item()

            def set_value(self, value, **axis_values):
                """
                Set a value using axis labels.

                Example:
                  .set_value(value=10, X=1, Y=1, Z=1)
                """
                if set(axis_values.keys()) != set(self.axis_values.keys()):
                    raise ValueError(f"Expected axes: {list(self.axis_values.keys())}, but got {list(axis_values.keys())}")

                # Convert axis labels to zero-based indices
                indices = tuple(
                    int(self.xp.where(self.axis_values[axis] == axis_values[axis])[0][0])
                    for axis in self.axis_values.keys()
                )
                # Write to the underlying raw array
                self.base_array[indices] = value

            def set_interpolation_method(self, method="linear"):
                """Change interpolation method dynamically."""
                self.interpolation_method = method

            def rename_axes(self, new_names_dict):
                """
                Rename axes dynamically.
                Example: rename_axes({"X": "Frequency", "Y": "Amplitude"})
                """
                if not all(axis in self.axis_values for axis in new_names_dict.keys()):
                    raise ValueError(f"Invalid axis names: {list(new_names_dict.keys())}")

                self.axis_values = {
                    new_names_dict.get(axis, axis): values
                    for axis, values in self.axis_values.items()
                }

            def to_numpy(self):
                """Convert to a NumPy array on CPU."""
                if self.xp.__name__ == "cupy":
                    return cp.asnumpy(self.base_array)
                else:
                    # Already NumPy, just return a copy or the underlying array
                    return self.base_array

            def to_cupy(self):
                """Convert to a CuPy array on GPU."""
                return cp.asarray(self.base_array)

            def get_axes_and_values(self):
                """Return a dictionary of axes and values."""
                return {axis: self.axis_values[axis].tolist() for axis in self.axis_values.keys()}

            def __repr__(self):
                """String representation."""
                backend = "CuPy" if self.xp.__name__ == "cupy" else "NumPy"
                return (
                    f"NamedAxesArray(shape={self.shape}, "
                    f"axes={list(self.axis_values.keys())}, "
                    f"backend={backend}, interpolation={self.interpolation_method})"
                )

            def __str__(self):
                """Custom print output."""
                axes_info = "\n".join(
                    [f"  {axis}: {self.axis_values[axis].tolist()}" for axis in self.axis_values.keys()]
                )
                backend = "CuPy" if self.xp.__name__ == "cupy" else "NumPy"
                return (
                    f"NamedAxesArray (shape={self.shape}, axes={len(self.axis_values)}, "
                    f"backend={backend}, interpolation={self.interpolation_method})\n"
                    f"Axes:\n{axes_info}\n"
                    f"Values:\n{super().__str__()}"
                )

        return BaseArray  # Return dynamically created class

    def used_device(self):
        """Returns the used device as string ('cpu' or 'gpu')."""
        return self.device

    def backend_type(self):
        """Returns 'CuPy' or 'NumPy' depending on which library is used."""
        return "CuPy" if self.device == "gpu" else "NumPy"

    def __getattr__(self, name):
        """Delegate attribute access to the inner BaseArray."""
        return getattr(self.array, name)

    def __repr__(self):
        return repr(self.array)

    def __str__(self):
        return str(self.array)

    def __getitem__(self, item):
        """Delegate item retrieval to the underlying BaseArray."""
        return self.array.__getitem__(item)

    def __setitem__(self, key, value):
        """Delegate item assignment to the underlying BaseArray."""
        self.array.__setitem__(key, value)


class NamedAxesArray:
    """
    An array that build on numpy and cupy and inherits its functionalities.
    The main extensions are:
        * Use labelled Axes: E.g., Axis1, Axis2 ... AxisN or X, Y, Z. Note: It is not restricted to 3D.
        * Be able to interpolate between this values: E.g., for 3D array [1.5, 1.3, 2] = value. The "value" does not exit sice [1.5, 1.3, 2] does not exits. However, with interpolation it is possible to get the desired value.

    NOTE: Extrapolation does not work / is not activated -> thus values outside the given range of the input array yield an error when using the method get_value()

    TODO: GPU does not fully work?

    TODO: Check the class again in deep!!!
    """

    def __init__(self, input_array: np.ndarray | cp.ndarray, axis_values, device="cpu"):
        """
        Initialize with NumPy or CuPy depending on `device` argument.
        """
        if not isinstance(input_array, (np.ndarray, cp.ndarray)):
            raise TypeError(f"Input array must be numpy.ndarray or cupy.ndarray, got {type(input_array)}")

        if device not in ["cpu", "gpu"]:
            raise ValueError(f"'device' must be 'cpu' or 'gpu', got {device}")

        self.device = device
        self.xp = self._get_backend()
        self.BaseArray = self._create_base_array_class(self.xp)
        self.array = self.BaseArray(input_array, axis_values, self.xp)

    def _get_backend(self):
        """Returns NumPy or CuPy as the backend."""
        return cp if self.device == "gpu" else np

    def _create_base_array_class(self, xp):
        """
        Dynamically defines BaseArray with NumPy or CuPy.
        """
        class BaseArray(xp.ndarray):
            """
            Custom array class with named axes and interpolation.
            """

            def __new__(cls, input_array, axis_values, xp):
                obj = xp.asarray(input_array).copy().view(cls)  # Ensure mutable array
                obj.axis_values = {axis: xp.array(values) for axis, values in axis_values.items()}
                obj.interpolation_method = "linear"
                obj.xp = xp
                obj.base_array = obj  # Store reference to ensure item assignment works
                return obj

            def __array_finalize__(self, obj):
                """Ensure attributes persist when slicing."""
                if obj is None:
                    return
                self.axis_values = getattr(obj, 'axis_values', None)
                self.interpolation_method = getattr(obj, 'interpolation_method', "linear")
                self.xp = getattr(obj, 'xp', None)
                self.base_array = getattr(obj, 'base_array', None)

            def __array_wrap__(self, out_arr, context=None):
                """Ensure that operations return BaseArray instead of plain NumPy array."""
                return np.ndarray.__array_wrap__(self, out_arr, context)

            def __setitem__(self, index, value):
                """Support normal NumPy/CuPy-style item assignment."""
                self.base_array.view(np.ndarray)[index] = value  # Modify the original array

            def get_value(self, **axis_values):
                """
                Get interpolated value using axis labels with conditional extrapolation.

                Steps:
                1. Validate that the provided axis labels match the expected axes.
                2. Build the grid definition (points) from self.axis_values.
                3. Build query_points so that its shape is (..., ndim), with the last axis
                   holding the coordinate for each dimension.
                4. Check for each axis if any query coordinate is outside the grid bounds.
                   - If a coordinate is less than the minimum or greater than the maximum,
                     we flag that extrapolation is required and print a warning.
                5. Depending on whether extrapolation is needed, call interpn with or without
                   the parameters to enable extrapolation.
                6. Return a scalar if a single value is interpolated, or the full array otherwise.
                """
                # 1. Validate that the provided axes match the defined axes.
                if set(axis_values.keys()) != set(self.axis_values.keys()):
                    raise ValueError(
                        f"Expected axes: {list(self.axis_values.keys())}, but got {list(axis_values.keys())}")

                # 2. Retrieve the grid coordinate arrays (in the order of the keys).
                points = [self.axis_values[axis] for axis in self.axis_values.keys()]

                # 3. Build query_points:
                #    - Stack the user-provided coordinate arrays. Each should have shape (d0, d1, ..., d_{N-1}).
                #    - moveaxis moves the 0th axis (which holds the stacked coordinates) to the end,
                #      so the final shape is (d0, d1, ..., d_{N-1}, N).
                query_points = self.xp.moveaxis(
                    self.xp.array([axis_values[axis] for axis in self.axis_values.keys()]),
                    0, -1
                )

                # 4. Check if any query coordinate is outside the grid bounds for each axis.
                extrapolate = False
                for i, (axis_name, coord_array) in enumerate(self.axis_values.items()):
                    min_val = float(coord_array.min())
                    max_val = float(coord_array.max())
                    # query_points[..., i] extracts the coordinates for the current axis.
                    if self.xp.any(query_points[..., i] < min_val) or self.xp.any(query_points[..., i] > max_val):
                        print(
                            f"Warning: Extrapolation required for axis '{axis_name}' (range: {min_val} to {max_val}).")
                        extrapolate = True

                # 5. Choose the appropriate interpolation function based on the backend.
                if self.xp.__name__ == "cupy":
                    from cupyx.scipy.interpolate import interpn
                else:
                    from scipy.interpolate import interpn

                # 6. Call interpn with extrapolation parameters if required.
                if extrapolate:
                    # bounds_error=False tells interpn not to raise an error for out-of-bound points,
                    # and fill_value=None instructs it to extrapolate.
                    interpolated_values = interpn(
                        points,
                        self,
                        query_points,
                        method=self.interpolation_method,
                        bounds_error=False,
                        fill_value=None
                    )
                else:
                    interpolated_values = interpn(
                        points,
                        self,
                        query_points,
                        method=self.interpolation_method
                    )

                # 7. Return a scalar if only one value is produced, or the array otherwise.
                return interpolated_values if interpolated_values.size > 1 else interpolated_values.item()

            def set_value(self, value, **axis_values):
                """
                Set a value using axis labels.

                Example:
                .set_value(value=10, X=1, Y=1, Z=1)
                """
                if set(axis_values.keys()) != set(self.axis_values.keys()):
                    raise ValueError(f"Expected axes: {list(self.axis_values.keys())}, but got {list(axis_values.keys())}")

                # Convert axis labels to indices
                indices = tuple(
                    int(self.xp.where(self.axis_values[axis] == axis_values[axis])[0][0])
                    for axis in self.axis_values.keys()
                )
                self.base_array[indices] = value  # Modify in place

            def set_interpolation_method(self, method="linear"):
                """Change interpolation method dynamically."""
                self.interpolation_method = method

            def rename_axes(self, new_names_dict):
                """Rename axes dynamically."""
                if not all(axis in self.axis_values for axis in new_names_dict.keys()):
                    raise ValueError(f"Invalid axis names: {list(new_names_dict.keys())}")

                self.axis_values = {new_names_dict.get(axis, axis): values for axis, values in self.axis_values.items()}

            def to_numpy(self):
                """Convert to NumPy."""
                return self.xp.asnumpy(self) if self.xp.__name__ == "cupy" else self.xp.asarray(self)

            def to_cupy(self):
                """Convert to CuPy."""
                return cp.asarray(self)

            def get_axes_and_values(self):
                """Return a dictionary of axes and values."""
                return {axis: self.axis_values[axis].tolist() for axis in self.axis_values.keys()}

            def __repr__(self):
                """String representation."""
                backend = "CuPy" if self.xp.__name__ == "cupy" else "NumPy"
                return f"NamedAxesArray(shape={self.shape}, axes={list(self.axis_values.keys())}, backend={backend}, interpolation={self.interpolation_method})"

            def __str__(self):
                """Custom print output."""
                axes_info = "\n".join([f"  {axis}: {self.axis_values[axis].tolist()}" for axis in self.axis_values.keys()])
                backend = "CuPy" if self.xp.__name__ == "cupy" else "NumPy"
                return (
                    f"NamedAxesArray (shape={self.shape}, axes={len(self.axis_values)}, backend={backend}, interpolation={self.interpolation_method})\n"
                    f"Axes:\n{axes_info}\n"
                    f"Values:\n{super().__str__()}"
                )

        return BaseArray  # Return dynamically created class

    def used_device(self):
        """Returns the used device as string (cpu or gpu)."""
        return self.device

    def backend_type(self):
        """Returns 'CuPy' or 'NumPy'."""
        return "CuPy" if self.device == "gpu" else "NumPy"

    def __getattr__(self, name):
        """Delegate attribute access to the inner BaseArray."""
        return getattr(self.array, name)

    def __repr__(self):
        return repr(self.array)

    def __str__(self):
        return str(self.array)

    def __getitem__(self, item):
        """
        Delegate item retrieval to the underlying BaseArray.
        """
        return self.array.__getitem__(item)

    def __setitem__(self, key, value):
        """
        Delegate item assignment to the underlying BaseArray.
        """
        self.array.__setitem__(key, value)


class CustomArray2(da.Array):
    def __new__(cls,
                dask_array: da.Array,
                block_number: int = None,
                block_idx: tuple = None,
                main_volume_shape: tuple = None,
                main_volume_blocks: tuple = None,
                total_number_blocks: int = None,
                unit: pint.UnitRegistry = None,
                meta: dict = None):
        instance = super().__new__(cls, dask_array.dask, dask_array.name, dask_array.chunks, dtype=dask_array.dtype)
        instance.block_number = block_number
        instance.block_idx = block_idx
        instance.unit = unit
        instance.size_mb = dask_array.dtype.itemsize * dask_array.size / (1024 ** 2)

        if main_volume_shape is None:
            instance.main_volume_shape = instance.shape
        else:
            instance.main_volume_shape = main_volume_shape

        if total_number_blocks is None:
            instance.total_number_blocks = math.prod(dask_array.blocks.shape)
        else:
            instance.total_number_blocks = total_number_blocks

        if main_volume_blocks is None:
            instance.main_volume_blocks = instance.numblocks
        else:
            instance.main_volume_blocks = main_volume_blocks

        instance.meta = meta

        return instance

    def __repr__(self) -> str:
        arr_repr = super().__repr__()

        if self.block_idx is None:
            meta_repr = (f"\n  {'Type:':.<20} main volume"
                         f"\n  {'Estimated size:':.<20} {round(self.size_mb, 3)} MB "
                         f"\n  {'Unit:':.<20} {str(self.unit)} "
                         f"\n  {'Metadata:':.<20} {str(self.meta)} ")

            return arr_repr + meta_repr + "\n"

        else:
            t_global_start, x_global_start, y_global_start, z_global_start = self.get_global_index(0, 0, 0, 0)
            t_global_end, x_global_end, y_global_end, z_global_end = self.get_global_index(
                self.shape[-4] - 1, self.shape[-3] - 1, self.shape[-2] - 1, self.shape[-1] - 1)

            meta_repr = (f"\n  {'Type:':.<25} sub-volume of main volume "
                         f"\n  {'Block number:':.<25} {self.block_number}/{self.total_number_blocks - 1}"
                         f"\n  {'Total number blocks':.<25} {self.main_volume_blocks}"
                         f"\n  {'Block coordinates:':.<25} {self.block_idx} "
                         f"\n  {'Estimated size:':.<25} {round(self.size_mb, 3)} MB "
                         f"\n  {'Main volume shape:':.<25} {self.main_volume_shape} "
                         f"\n  {'Main volume coordinates:':.<25} t={t_global_start}:{t_global_end} "
                         f"x={x_global_start}:{x_global_end} y={y_global_start}:{y_global_end} z={z_global_start}:{z_global_end} "
                         f"\n  {'Unit:':.<25} {self.unit} "
                         f"\n  {'Metadata:':.<25} {self.meta}")

            return arr_repr + meta_repr + "\n"

    def __mul__(self, other):
        result = super().__mul__(other)
        result = CustomArray2(dask_array=result,
                              block_number=self.block_number,
                              block_idx=self.block_idx,
                              main_volume_shape=self.main_volume_shape,
                              total_number_blocks=self.total_number_blocks,
                              main_volume_blocks=self.main_volume_blocks,
                              meta=self.meta)

        return result

    def get_global_index(self, t, x, y, z):
        # print(f"Block index: {self.block_idx}")
        # print(f"Shape: {self.shape}")

        if not isinstance(self.block_idx, tuple) or len(self.block_idx) != 4:
            raise ValueError("block_idx must be a tuple of length 4")

        if not isinstance(self.shape, tuple) or len(self.shape) != 4:
            raise ValueError("shape must be a tuple of length 4")

        if t >= self.shape[0]:
            raise ValueError(f"Error: t block shape is only {self.shape[0]}, but index {t} was given")
        if x >= self.shape[1]:
            raise ValueError(f"Error: x block shape is only {self.shape[1]}, but index {x} was given")
        if y >= self.shape[2]:
            raise ValueError(f"Error: y block shape is only {self.shape[2]}, but index {y} was given")
        if z >= self.shape[3]:
            raise ValueError(f"Error: z block shape is only {self.shape[3]}, but index {z} was given")

        global_t = self.block_idx[-4] * self.shape[-4] + t
        global_x = self.block_idx[-3] * self.shape[-3] + x
        global_y = self.block_idx[-2] * self.shape[-2] + y
        global_z = self.block_idx[-1] * self.shape[-1] + z

        return global_t, global_x, global_y, global_z

    @property
    def blocks(self):
        block_view_object = CustomBlockView2(self,
                                             main_volume_shape=self.main_volume_shape,
                                             total_number_blocks=self.total_number_blocks,
                                             main_volume_blocks=self.main_volume_blocks)
        return block_view_object


class CustomBlockView2(da.core.BlockView):
    def __init__(self, custom_array: CustomArray2, main_volume_shape: tuple = None, total_number_blocks: int = None, main_volume_blocks: tuple = None):
        self.main_volume_shape = main_volume_shape
        self.total_number_blocks = total_number_blocks
        self.main_volume_blocks = main_volume_blocks
        self.block_number = 0
        self.unit = custom_array.unit
        super().__init__(custom_array)

    def __getitem__(self, index) -> CustomArray2:
        if isinstance(index, int):
            index = (index,)

        if len(index) == 1:
            index = (index[0], 0, 0, 0)
        elif len(index) == 2:
            index = (index[0], index[1], 0, 0)
        elif len(index) == 3:
            index = (index[0], index[1], index[2], 0)
        elif len(index) != 4:
            raise ValueError("block_idx must be a tuple of length 4")

        dask_array = super(CustomBlockView2, self).__getitem__(index)
        custom_dask_array = CustomArray2(dask_array=dask_array,
                                         block_number=self.block_number,
                                         block_idx=index,
                                         main_volume_shape=self.main_volume_shape,
                                         main_volume_blocks=self.main_volume_blocks,
                                         total_number_blocks=self.total_number_blocks,
                                         unit=self.unit)

        self.block_number += 1

        return custom_dask_array



class CustomArrayUnitOperationsExtension:
    """
    Made for the CustomArray class to inherit the functionality
    of unit-aware arithmetic.
    The subclass - CustomArray - requires the following methods to work:
      - _as_dask_array() -> da.Array                (View to the same Graph)
      - _wrap_like(darr, unit, meta) -> CustomArray (or Subclass)
      - _merge_meta(other) -> meta                  (optional; otherwise standard)
    """

    __array_priority__ = 1000  # for numpy to prefer this instead of ndarray

    ######### Methods, which the subclass need to provide #########
    def _as_dask_array(self) -> da.Array:
        raise NotImplementedError

    def _wrap_like(self, darr: da.Array, unit, meta):
        raise NotImplementedError

    def _merge_meta(self, other):
        m1 = getattr(self, "meta", None)
        m2 = getattr(other, "meta", None) if other is not None else None
        if m1 is None:
            return m2
        if m2 is None:
            return m1
        if isinstance(m1, dict) and isinstance(m2, dict):
            return {**m1, **m2}
        return f"{m1}; {m2}"

    ######### Unit Utilities #########
    def _unit(self):
        return getattr(self, "unit", None)

    @staticmethod
    def _is_dimensionless(u) -> bool:
        # Pint units are usually dimensionless
        return bool(getattr(u, "dimensionless", False))

    def _convert_other_to_self_unit(self, other):
        """
        For + and -: convert other to self.unit by scaling other-data (lazy).
        """
        u1 = self._unit()
        u2 = getattr(other, "unit", None)

        if u1 is None or u2 is None:
            # No clean conversion possible -> no scaling
            return other._as_dask_array(), u1

        # Calculate the factor using Pint: 1*u2 -> u1
        factor = (1 * u2).to(u1).magnitude
        return other._as_dask_array() * factor, u1

    # ######### Central binary operation #########
    def _binary_op(self, other, op, rule):
        a = self._as_dask_array()

        if isinstance(other, type(self)):
            b = other._as_dask_array()
            res_unit, a2, b2 = rule(self, other, a, b)
            res = op(a2, b2)
            return self._wrap_like(res, unit=res_unit, meta=self._merge_meta(other))
        else:
            # scalar/array-like without unit
            res_unit, a2, b2 = rule(self, None, a, other)
            res = op(a2, b2)
            return self._wrap_like(res, unit=res_unit, meta=getattr(self, "meta", None))

    # ######### Unit rules for binary ops #########
    @staticmethod
    def _rule_add_sub(self, other, a, b):
        if other is None:
            # scalar add/sub: only makes sense if scalar is interpreted as "in the same unit".
            return self._unit(), a, b
        b_conv, u = self._convert_other_to_self_unit(other)
        return u, a, b_conv

    @staticmethod
    def _rule_mul(self, other, a, b):
        u1 = self._unit()
        u2 = getattr(other, "unit", None) if other is not None else None
        if u1 is None and u2 is None:
            return None, a, b
        if u1 is None:
            return u2, a, b
        if u2 is None:
            return u1, a, b
        return (u1 * u2), a, b

    @staticmethod
    def _rule_truediv(self, other, a, b):
        u1 = self._unit()
        u2 = getattr(other, "unit", None) if other is not None else None
        if u1 is None and u2 is None:
            return None, a, b
        if u1 is None and u2 is not None:
            return (1 / u2), a, b
        if u1 is not None and u2 is None:
            return u1, a, b
        return (u1 / u2), a, b

    # ######### Operator Overloads #########
    def __add__(self, other): return self._binary_op(other, operator.add, self._rule_add_sub)
    def __sub__(self, other): return self._binary_op(other, operator.sub, self._rule_add_sub)
    def __mul__(self, other): return self._binary_op(other, operator.mul, self._rule_mul)
    def __truediv__(self, other): return self._binary_op(other, operator.truediv, self._rule_truediv)

    def __radd__(self, other): return self.__add__(other)
    def __rmul__(self, other): return self.__mul__(other)

    def __rsub__(self, other):
        # scalar - array
        res = operator.sub(other, self._as_dask_array())
        return self._wrap_like(res, unit=self._unit(), meta=getattr(self, "meta", None))

    def __rtruediv__(self, other):
        # scalar / array
        u = self._unit()
        res_unit = None if u is None else (1 / u)
        res = operator.truediv(other, self._as_dask_array())
        return self._wrap_like(res, unit=res_unit, meta=getattr(self, "meta", None))

    def __pow__(self, power):
        res = self._as_dask_array() ** power
        u = self._unit()
        res_unit = None if u is None else (u ** power)
        return self._wrap_like(res, unit=res_unit, meta=getattr(self, "meta", None))

    def __neg__(self):
        res = -self._as_dask_array()
        return self._wrap_like(res, unit=self._unit(), meta=getattr(self, "meta", None))

    # ######### NumPy ufuncs (Basic) #########
    _UFUNC_RULES = {
        np.negative: lambda u: u,
        np.absolute: lambda u: u,
        np.sqrt:     lambda u: None if u is None else (u ** 0.5),
        np.square:   lambda u: None if u is None else (u ** 2),
        np.reciprocal: lambda u: None if u is None else (1 / u),
    }

    _REQUIRE_DIMENSIONLESS = {np.exp, np.log, np.log10}

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        # Convert inputs: CustomArray -> da.Array
        conv = []
        units = []
        metas = []
        for x in inputs:
            if isinstance(x, type(self)):
                conv.append(x._as_dask_array())
                units.append(getattr(x, "unit", None))
                metas.append(getattr(x, "meta", None))
            else:
                conv.append(x)
                units.append(None)
                metas.append(None)

        # Dimensionless check for certain ufuncs
        if ufunc in self._REQUIRE_DIMENSIONLESS:
            u0 = units[0]
            if u0 is not None and not self._is_dimensionless(u0):
                raise ValueError(f"{ufunc.__name__} requires dimensionless input, got {u0}")

        rule = self._UFUNC_RULES.get(ufunc)
        res_unit = None if rule is None else rule(units[0])

        res = ufunc(*conv, **kwargs)

        # Meta: very conservative -> use self.meta
        return self._wrap_like(res, unit=res_unit, meta=getattr(self, "meta", None))


class CustomArray(CustomArrayUnitOperationsExtension, da.Array): # Note: inheritance oder matters!
    """
    Extending the dask Array class with additional functionalities.

    The Custom dask Array obtains the following additional features:
    * block number of the individual block
    * the block index (see how .blocks.reval() crates a list of blocks)
    * the shape of the main volume in each individual block
    * the total number of blocks in each individual block
    * the unit
    * custom metadata

    Further, consider the coordinate system of the axis 0 = x, axis 1 = y, axis 2 = z:

                   origin
                 +----------> (+z)
                /|
              /  |
            /    |
        (+x)     |
                 v (+y)

        This is important when dealing with individual blocks of the whole volume. The block number is
        increased first in z, then y and then x direction.
    """

    def __new__(cls,
                dask_array: da.Array,
                block_number: int = None,
                block_idx: tuple = None,
                main_volume_shape: tuple = None,
                main_volume_blocks: tuple = None,  # number blocks in each dimension
                total_number_blocks: int = None,  # number blocks
                unit: pint.UnitRegistry = None,
                meta: dict = None):
        """
        To create a new instance of the dask.Array class and add custom attributes.

        Note: The __new__ is called before __init__. It is responsible for creating a new instance of the class. After, the new object is created with
              __new__ the __init__ is called for instantiate the new created object.

        :param dask_array: dask.Array object
        :param block_number: when creating blocks out of an array with the argument in the dask from_array(): chunks=(size_x, size_y, size_z). A number in elements [0:number_blocks]
        :param block_idx: tuple containing the block index for each dimension. E,g., (1, 3, 5) for block position: x=1, y=3, z=5. Check coordinate system in description of the class.
        :param main_volume_shape: the shape of the main volume in x,y,z for a 3D array, for example.
        :param total_number_blocks: the total number of blocks of which the main volume consists.
        :param unit: unit corresponding to the values in the array.
        :param meta: custom data added by the user.
        """
        instance = super().__new__(cls, dask_array.dask, dask_array.name, dask_array.chunks, dtype=dask_array.dtype)
        instance.block_number = block_number
        instance.block_idx = block_idx
        instance.unit = unit
        instance.size_mb = dask_array.dtype.itemsize * dask_array.size / (1024 ** 2)

        # When no shape is given, it will be assumed it is the main volume rather than a block of the main volume. Otherwise, set the shape of the main volume in each block.
        # Important, if only the block is given and information of the main volume is required.
        if main_volume_shape is None:
            instance.main_volume_shape = instance.shape
        else:
            instance.main_volume_shape = main_volume_shape
        # The same as above. If no shape is given, it will be assumed that it is the main volume and not a block of it. Otherwise, assign the number of blocks to each block
        # to be able to know the total number of blocks.
        if total_number_blocks is None:
            instance.total_number_blocks = math.prod(dask_array.blocks.shape) - 1
        else:
            instance.total_number_blocks = total_number_blocks

        # When main volume shape is None it will be assumed it is the main volume, thus set shape.
        # If not None, then already set by main volume and it is a block!
        if main_volume_blocks is None:
            instance.main_volume_blocks = instance.numblocks
        else:
            instance.main_volume_blocks = main_volume_blocks

        instance.meta = meta

        return instance

    def __repr__(self) -> str:
        """
        To extend the __repr__ method from the superclass and to add additional information if a CustomArray gets printed to the console with print().

        Additional information:
        * Start and end indices of the block in the global volume (only for x,y,z dimensions supported yet)
        * The block number (e.g., 4/100)
        * The block coordinates (e.g., (0,0,4))
        * The estimated size of the block in [MB]
        * The shape of the main volume (e.g., 1000, 120,120,120)
        * The set unit
        * The custom metadata

        :return: formatted string.
        """
        arr_repr = super().__repr__()  # Get the standard representation of the Dask array

        # case 1: if main volume
        if self.block_idx is None:
            meta_repr = (f"\n  {'Type:':.<20} main volume"
                         f"\n  {'Estimated size:':.<20} {round(self.size_mb, 3)} MB "
                         f"\n  {'Unit:':.<20} {str(self.unit)} "
                         f"\n  {'Metadata:':.<20} {str(self.meta)} ")

            return arr_repr + meta_repr + "\n"  # Concatenate the two representations

        # case 2: if sub volume
        else:
            x_global_start, y_global_start, z_global_start = self.get_global_index(0, 0, 0)
            x_global_end, y_global_end, z_global_end = self.get_global_index(self.shape[-3] - 1, self.shape[-2] - 1, self.shape[-1] - 1)

            meta_repr = (f"\n  {'Type:':.<20} sub-volume of main volume "
                         f"\n  {'Block number:':.<20} {self.block_number}/{self.total_number_blocks} "
                         f"\n  {'Total number blocks':.<} {self.main_volume_blocks}"
                         f"\n  {'Block coordinates:':.<20} {self.block_idx} "
                         f"\n  {'Estimated size:':.<20} {round(self.size_mb, 3)} MB "
                         f"\n  {'Main volume shape:':.<20} {self.main_volume_shape} "
                         f"\n  {'Main volume coordinates:':.<20} x={x_global_start}:{x_global_end} y={y_global_start}:{y_global_end} z={z_global_start}:{z_global_end} "
                         f"\n  {'Unit:':.<20} {self.unit} "
                         f"\n  {'Metadata:':.<20} {self.meta}")

            return arr_repr + meta_repr + "\n"  # Concatenate the two representations

    def _repr_html_(self):
        base = super()._repr_html_()

        # --- Unit row (optional; remove this block if you don't want Unit in the table)
        unit_val = getattr(self, "unit", None)
        unit_str = _html.escape("" if unit_val is None else str(unit_val))

        unit_row = (
            "<tr>"
            "<th>Unit</th>"
            f"<td>{unit_str}</td>"
            "<td></td>"
            "</tr>"
        )

        # --- Metadata collapsible (always, regardless of type)
        meta_val = getattr(self, "meta", None)

        # Render meta safely + reasonably (pretty JSON for dicts/lists; str() otherwise)
        try:
            if isinstance(meta_val, (dict, list, tuple)):
                meta_text = _json.dumps(meta_val, indent=2, sort_keys=True, default=str)
            else:
                meta_text = str(meta_val)
        except Exception:
            meta_text = repr(meta_val)

        # Optional: truncate very large metadata to avoid huge notebook outputs
        MAX_CHARS = 50_000
        truncated = ""
        if len(meta_text) > MAX_CHARS:
            meta_text = meta_text[:MAX_CHARS]
            truncated = "\n… (truncated)"

        details_html = (
            "<details style='margin-top:0.5em'>"
            "<summary>Metadata</summary>"
            f"<pre style='white-space:pre-wrap; margin-top:0.5em'>"
            f"{_html.escape(meta_text)}{_html.escape(truncated)}"
            "</pre>"
            "</details>"
        )

        # If Dask fell back to non-HTML output, degrade gracefully
        if not isinstance(base, str) or "<tbody" not in base:
            return f"{base}\n\nMetadata:\n{meta_text}{truncated}"

        # Insert unit row into the stats table (leave everything else unchanged)
        out = base.replace("</tbody>", unit_row + "</tbody>", 1)

        # Append collapsible metadata below the Dask visualization
        out += details_html
        return out

###    def __mul__(self, other):
###        """
###        For preserving information from the left multiplicand. Add it to the result.
###
###        :param other: The right multiplicand. It has to be a dask.Array or a CustomArray
###        :return: product of array 1 and array 2
###        """
###        result = super().__mul__(other)
###        result = CustomArray(dask_array=result,
###                             block_number=self.block_number,
###                             block_idx=self.block_idx,
###                             main_volume_shape=self.main_volume_shape,
###                             total_number_blocks=self.total_number_blocks,
###                             main_volume_blocks=self.main_volume_blocks,
###                             meta=self.meta)
###
###        return result

    def get_global_index(self, x, y, z):
        """
        Useful if a block of a CustomArray is handled individually. To get the global indices (x,y,z) of the local indices in the respective block.
        local(x,y,z) ===> global(x,y,z)

        :param x: local index in x (thus in current block)
        :param y: local index in y (thus in current block)
        :param z: local index in z (thus in current block)
        :return: global indices as tuple (x,y,z)
        """
        # x,y,z is in block, and global in original global volume
        if x >= self.shape[0]:
            print(f"Error: x block shape is only {self.shape[0]}, but index {x} was given")
            return
        if y >= self.shape[1]:
            print(f"Error: y block shape is only {self.shape[1]}, but index {y} was given")
            return
        if z >= self.shape[2]:
            print(f"Error: z block shape is only {self.shape[2]}, but index {z} was given")
            return

        # with -3,-2,-1 use dimensions starting from the last position. Thus, it works for higher dimensionality arrays than 3D arrays too.
        global_x = self.block_idx[-3] * self.shape[-3] + x
        global_y = self.block_idx[-2] * self.shape[-2] + y
        global_z = self.block_idx[-1] * self.shape[-1] + z

        return global_x, global_y, global_z

    @property
    def blocks(self):
        """
        To override the method of the superclass. Create a CustomBLockView instead of the dask BlockView. However, the CustomBlockView inherits from
        the dask BlockView, but extends its functionality!

        :return: CustomBlockView
        """

        block_view_object = CustomBlockView(self,
                                            main_volume_shape=self.main_volume_shape,
                                            total_number_blocks=self.total_number_blocks,
                                            main_volume_blocks=self.main_volume_blocks)
        return block_view_object


    def _as_dask_array(self) -> da.Array:
        """
        Only necessary for the CustomArrayUnitOperationsExtension (which inherits the CustomArray)
        :return: dask array
        """
        # normal dask array view (without metadata)
        return da.Array(self.dask, self.name, self.chunks, dtype=self.dtype, shape=self.shape)

    def _wrap_like(self, darr: da.Array, unit, meta):
        """
        # Rebuild the result again as a CustomArray; this is for transferring the fields.

        :param darr: dask Array
        :param unit: pint Unit
        :param meta: metadata (e.g., string or dictionary)
        :return: CustomArray
        """
        return CustomArray(
            darr,
            block_number=None,
            block_idx=None,
            main_volume_shape=getattr(self, "main_volume_shape", None),
            main_volume_blocks=getattr(self, "main_volume_blocks", None),
            total_number_blocks=getattr(self, "total_number_blocks", None),
            unit=unit,
            meta=meta,
        )


class CustomBlockView(da.core.BlockView):
    """
    Extending the dask class BlockView with additional functionalities. This is required for the CustomArray class that inherits from dask Array.
    Additional functionalities of one block:
    * Shape of the main volume in each block
    * Number of total blocks in each block
    * Unit of the main volume in each block
    """

    def __init__(self, custom_array: CustomArray, main_volume_shape: tuple = None, total_number_blocks: int = None, main_volume_blocks: tuple = None):
        """
        Addition additional the main volume shape, the total number of blocks and the unit. Also, call the super constructor.

        :param custom_array: the CustomArray object
        :param custom_array: the CustomArray object
        :param main_volume_shape:
        :param total_number_blocks:
        """
        self.main_volume_shape = main_volume_shape
        self.total_number_blocks = total_number_blocks  # total number of blocks as int
        self.main_volume_blocks = main_volume_blocks  # total number blocks in each dimension as tuple
        self.block_number = 0
        self.unit = custom_array.unit
        super().__init__(custom_array)

    def __getitem__(self, index) -> CustomArray:
        """
        Override the __getitem__ of the superclass. In the workflow of dask each block is a new dask.Array. Thus, replacing the dask.Array with
        a CustomArray that inherits from dask.Array.

        Be aware that each block has a block number. The block number allows selecting the next block. The blocks are  ordered condescending in
        the following dimensions: first z, then y, then x, with the following coordinate system:

               origin
                 +----------> (+z)
                /|
              /  |
            /    |
        (+x)     |
                 v (+y)

        :param index: index of the block
        :return: a block as CustomArray
        """
        dask_array = super(CustomBlockView, self).__getitem__(index)
        custom_dask_array = CustomArray(dask_array=dask_array,
                                        block_number=self.block_number,
                                        block_idx=index,
                                        main_volume_shape=self.main_volume_shape,
                                        main_volume_blocks=self.main_volume_blocks,
                                        total_number_blocks=self.total_number_blocks,
                                        unit=self.unit)

        self.block_number += 1  # to get block number of in the main volume (first z, then y, then x)

        return custom_dask_array


class MyLocalCluster:
    # TODO Docstring and also describe remaining parts of class!

    def __init__(self, nvml_diagnostics=True, split_large_chunks=False):
        self.cluster = None
        self.cluster_type: str = None
        self.client = None

        # 1) Prevent rapids libs from initializing cuda at import time
        os.environ.setdefault("RAPIDS_NO_INITIALIZE", "1") # Prevent RAPIDS from auto‑initializing CUDA at import time

        # 2) Disable or enable nvml diagnostics. Disabling it can solve permission issues when computing on a gpu node.
        self.nvml_diagnostics = nvml_diagnostics

        # 3) To avoid/allow creating the large chunks (e.g., caused by reshaping)
        self.split_large_chunks = split_large_chunks


    def _config(self):
        dask.config.set({
            "distributed.diagnostics.nvml": self.nvml_diagnostics,
            "array.slicing.split_large_chunks": self.split_large_chunks,
            "distributed.worker.memory.target": 0.8,   # start spilling later
            "distributed.worker.memory.spill": 0.9,    # spill only when very full
            "distributed.worker.memory.pause": False,  # do not pause workers
        })

        if self.nvml_diagnostics is False:
            Console.printf("warning", "nvml diagnostics is disabled!")

        if self.split_large_chunks is True:
            Console.printf("info", "Splitting large chunks is activated")


    def start_cpu(self, number_workers: int, threads_per_worker: int, memory_limit_per_worker: str = "30GB"):

        self._config()
        self.cluster = LocalCluster(n_workers=number_workers,
                                    threads_per_worker=threads_per_worker,
                                    memory_limit=memory_limit_per_worker,
                                    dashboard_address=":55000")
        self.cluster_type = "cpu"
        self.__start_client()

    def start_cuda(self, device_numbers: list[int], device_memory_limit: str = "30GB", rmm_pool_size="28", use_rmm_cupy_allocator: bool = False, protocol="tpc", **kwargs): # protocol="ucx"
        """
        Protocols like tcp possible, or ucx. "ucx" is best for GPU to GPU, GPU to host communication: https://ucx-py.readthedocs.io/en/latest/install.html
        """

        self._config()

        #dask.config.set({"distributed.diagnostics.nvml": self.nvml_diagnostics})
        #if use_rmm_cupy_allocator:
            #rmm.reinitialize(pool_allocator=True)
            #cp.cuda.set_allocator(rmm_cupy_allocator)

        possible_protocols = ["tcp", "tls", "inproc", "ucx"]
        if protocol not in possible_protocols:
            Console.printf("error", f"Only protocols {possible_protocols} available for Local CUDA cluster, but you set {protocol}. Terminate program!")
            sys.exit()

        # 2) To spawn UCX cluster before any CUDA imports happen. Important!
        self.cluster = LocalCUDACluster(n_workers=len(device_numbers),
                                        device_memory_limit=device_memory_limit, # controls size of LRU-Cache (when worker starts to spill GPU -> Host)
                                        rmm_pool_size=rmm_pool_size,             # activates the RAPIDS Memory Manager (RMM) Pool and pre-allocates a large GPU memory pool
                                        jit_unspill=True,
                                        CUDA_VISIBLE_DEVICES=device_numbers,
                                        protocol=protocol,
                                        dashboard_address=":55000",
                                        **kwargs)#nanny=False)
        self.cluster_type = "cuda"
        self.__start_client()

        # 3) Now, if you want RMM on the workers, do it via client.run
        if use_rmm_cupy_allocator:
            def _init_rmm():
                import rmm, cupy as cp
                rmm.reinitialize(pool_allocator=True)
                cp.cuda.set_allocator(rmm_cupy_allocator)
            # push that init to each worker
            self.client.run(_init_rmm)

    def close(self):
        """
        Close both the Dask client and the underlying cluster,
        freeing up threads/processes (and GPU memory, if using CUDA).
        """
        # 1) Close client first
        if self.client is not None:
            try:
                self.client.close()
                Console.printf("success", "Successfully closed client")
            except Exception as e:
                Console.printf("error", f"Error closing client: {e}")
            finally:
                self.client = None


        time.sleep(1) # TODO: Maybe solves issue regarding that port still blocked?

        # 2) Then close cluster
        if self.cluster is not None:
            try:
                self.cluster.close()
                Console.printf("success", "Successfully closed cluster")
            except Exception as e:
                Console.printf("error", f"Error closing cluster: {e}")
            finally:
                self.cluster = None

        # 3) Reset cluster type
        self.cluster_type = None

    def __start_client(self):
        from printer import Console # TODO: To solve circular import issue in spectral_spatial_simulation

        self.client = Client(self.cluster)
        # self.client = self.cluster.get_client()
        dashboard_url = self.client.dashboard_link

        Console.printf("success",
                       f"Started {self.cluster_type} Cluster \n"
                       f" Link to dashboard: {dashboard_url} ")



###    def start_cuda_test(
###        self,
###        device_numbers: list[int] = 1,
###        threads_per_worker: int = 1,
###        memory_limit_per_worker: str = "30GB",
###        dashboard_port: int = 55000,
###        local_directory: str = "/tmp/dask-worker-space",
###        use_rmm_cupy_allocator: bool = False,
###    ):
###        # 0) Force-disable NVML diagnostics for this cluster, regardless of instance setting
###        prev_nvml_diagnostics = self.nvml_diagnostics
###        self.nvml_diagnostics = False
###
###        self._config()
###        self.nvml_diagnostics = prev_nvml_diagnostics
###
###        # 1) LocalCluster with *no Nanny* (processes=False or worker_class=Worker)
###        self.cluster = LocalCluster(
###            n_workers=1,
###            threads_per_worker=threads_per_worker,
###            processes=False,                         # <- critical: no Nanny
###            worker_class=Worker,                    # (optional but explicit)
###            memory_limit=memory_limit_per_worker,   # or "auto"
###            local_directory=local_directory,
###            dashboard_address=f":{dashboard_port}",
###        )
###        self.cluster_type = "cuda"
###        self.__start_client()
###
###        # 2) Optional: RMM / CuPy allocator init
###        if use_rmm_cupy_allocator:
###            def _init_rmm():
###                import rmm
###                import cupy as cp
###                rmm.reinitialize(pool_allocator=True) #, initial_pool_size="10GB")
###                cp.cuda.set_allocator(rmm_cupy_allocator)
###
###            self.client.run(_init_rmm)


###    def start_cuda_test(
###        self,
###        device_numbers: list[int] = 1,
###        threads_per_worker: int = 1,
###        memory_limit_per_worker: str = "30GB",
###        dashboard_port: int = 55000,
###        local_directory: str = "/tmp/dask-worker-space",
###        use_rmm_cupy_allocator: bool = False,
###        nanny=False
###    ):
###        """
###        Experimental GPU-friendly cluster that avoids dask_cuda / NVML.
###
###        - Uses a plain LocalCluster (no LocalCUDACluster, no UCX/NVML integration).
###        - Assumes CUDA_VISIBLE_DEVICES / SLURM already restrict visible GPUs,
###          but additionally sets CUDA_VISIBLE_DEVICES from `device_numbers`.
###        - All GPU work must be done explicitly via CuPy / Numba inside your tasks.
###        """
###
###        # 0) Force-disable NVML diagnostics for this cluster, regardless of instance setting
###        prev_nvml_diagnostics = self.nvml_diagnostics
###        self.nvml_diagnostics = False
###
###        # Apply dask config (nvml off, chunk behavior, memory thresholds, etc.)
###        self._config()
###
###        # Restore original flag so the object state isn't permanently altered
###        self.nvml_diagnostics = prev_nvml_diagnostics
###
###        ## 1) Restrict visible GPUs for this process (and its workers)
###        #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_numbers)
###
###        # 2) Start a plain LocalCluster that is "GPU aware" only via CUDA_VISIBLE_DEVICES
###        self.cluster = LocalCluster(
###            n_workers=1,
###            threads_per_worker=1,
###            processes=True,
###            memory_limit="600GB",
###            local_directory=local_directory,
###            dashboard_address=f":{dashboard_port}",
###        )
###        self.cluster_type = "cuda"  # still call it "cuda" for downstream logic
###        self.__start_client()
###
###        # 3) Optional: initialize RMM/CuPy allocator on all workers
###        if use_rmm_cupy_allocator:
###            def _init_rmm():
###                import rmm
###                import cupy as cp
###                # You must have rmm_cupy_allocator defined/imported at module level
###                rmm.reinitialize(pool_allocator=True, initial_pool_size="10GB") ### TODO: Added initial pool size !!!! Maybe remove again
###                cp.cuda.set_allocator(rmm_cupy_allocator)
###
###            self.client.run(_init_rmm)




###class ConfiguratorGPU():
###    """
###    TODO!!! Describe it
###    """
###
###    def __init__(self, required_space_gpu: np.ndarray):
###
###        torch.cuda.empty_cache()
###        self.available_gpus: int = torch.cuda.device_count()
###        self.selected_gpu: int = None
###        self.required_space_gpu = required_space_gpu
###        self.free_space_selected_gpu: torch.Tensor = None
###
###    def select_least_busy_gpu(self):
###        from code.tests.printer_logger_test import Console # TODO: To solve circular import issue in spectral_spatial_simulation
###
###        # based on percentage free
###        free_space_devices = []
###        for device in range(self.available_gpus):
###            space_total_cuda = torch.cuda.mem_get_info(device)[1]
###            space_free_cuda = torch.cuda.mem_get_info(device)[0]
###            percentage_free_space = space_free_cuda / space_total_cuda
###            free_space_devices.append(percentage_free_space)
###
###        self.selected_gpu = free_space_devices.index(max(free_space_devices))
###
###        torch.cuda.set_device(self.selected_gpu)
###        Console.printf("info", f"Selected GPU {self.selected_gpu} -> most free space at moment!")
###
###        self.free_space_selected_gpu = torch.cuda.mem_get_info(self.selected_gpu)[0]  # free memory on GPU [bytes]
###
###    def print_available_gpus(self):
###        from printer import Console # TODO: To solve circular import issue in spectral_spatial_simulation
###
###        Console.add_lines("Available GPU(s) and free space on it:")
###
###        for device in range(self.available_gpus):
###            space_total_cuda = torch.cuda.mem_get_info(device)[1]
###            space_free_cuda = torch.cuda.mem_get_info(device)[0]
###            percentage_free_space = space_free_cuda / space_total_cuda
###            Console.add_lines(f" GPU {device} [MB]: {space_free_cuda / (1024 ** 2)} / {space_total_cuda / (1024 ** 2)} ({round(percentage_free_space, 2) * 100}%)")
###
###        Console.printf_collected_lines("info")
###
###    def select_gpu(self, gpu_index: int = 0):
###        #from code.tests.printer_logger_test import Console # TODO: To solve circular import issue in spectral_spatial_simulation
###        from printer import Console
###
###        torch.cuda.set_device(gpu_index)
###        self.selected_gpu = gpu_index
###        Console.printf("info", f"Selected GPU {gpu_index} -> manually selected by the user!")
###
###    def enough_space_available(self) -> bool:
###        #from code.tests.printer_logger_test import Console # TODO: To solve circular import issue in spectral_spatial_simulation
###        from printer import Console
###
###        if self.selected_gpu is not None:
###            self.free_space_selected_gpu = torch.cuda.mem_get_info(self.selected_gpu)[0]  # free memory on GPU [bytes]
###
###            if self.required_space_gpu <= self.free_space_selected_gpu:
###                Console.printf("info",
###                               f"Possible to put whole tensor of size {self.required_space_gpu / (1024 ** 2)} [MB] on GPU. Available: {self.free_space_selected_gpu / (1024 ** 2)} [MB]")
###            else:
###                Console.printf("info",
###                               f"Not possible to put whole tensor of size {self.required_space_gpu / (1024 ** 2)} [MB] on GPU. Available: {self.free_space_selected_gpu / (1024 ** 2)} [MB]")
###
###        else:
###            Console.printf("error", f"No GPU is selected. Number of available GPUs: {self.available_gpus}")
###            sys.exit()
###
###        # TODO: Check return type!
###        return self.required_space_gpu <= self.free_space_selected_gpu


###class SpaceEstimatorOLD:
###    """
###    To calculate the required space on the disk. This can be done with an already existing array, or a desired array of
###    a certain shape and data type. Useful, to check if it exceeds the current available space.
###    """
###
###    @overload
###    @staticmethod
###    def for_numpy(data: np.ndarray, unit: str = "MiB", verbose: bool = False) -> pint.Quantity:
###        """
###        For getting the size required for the respective array.
###
###        By default, the unit Mebibyte (MiB) is used.
###        :param data: numpy array
###        :param unit: unit as string (e.g., MiB, GiB, or MB, GB,...)
###        :return: required space as Pint Quantity in the requested unit
###        """
###        ...
###
###    @overload
###    @staticmethod
###    def for_numpy(data_shape: tuple, data_type, unit: str = "MiB", verbose: bool = False) -> pint.Quantity:
###        """
###        For estimating the size of a desired array.
###
###        :param data_shape: desired shape as tuple
###        :param data_type: numpy dtype or dtype specifier (e.g., np.float32, np.dtype("float32"), "float32")
###        :param unit: unit as string (e.g., MiB, GiB, or MB, GB,...)
###        :return: required space as Pint Quantity in the requested unit
###        """
###        ...
###
###    @staticmethod
###    def for_numpy(data_or_shape: np.ndarray| tuple,data_type=None,unit: str = "MiB", verbose: bool = False) -> pint.Quantity:
###        """
###        Just the runtime implementation for both overload variants.
###
###        :param data_or_shape: numpy array OR desired shape as tuple
###        :param data_type: dtype if data_or_shape is a shape tuple; otherwise, ignored if data_or_shape is an array
###        :param unit: unit as string (e.g., MiB, GiB, or MB, GB,...)
###        :return: required space as Pint Quantity in the requested unit
###        """
###        if isinstance(data_or_shape, np.ndarray):
###            shape = data_or_shape.shape
###            data_type = data_or_shape.dtype
###        else:
###            shape = data_or_shape
###            if data_type is None:
###                Console.printf("error", "When passing a shape tuple also the 'data type' must be provided!")
###                sys.exit()
###            data_type = np.dtype(data_type)
###
###        space_required_bytes = int(np.prod(shape)) * data_type.itemsize
###        space_required_desired_unit = (space_required_bytes * u.byte).to(unit)
###
###        # Print if desired
###        Console.printf("info", f"The required space is {space_required_desired_unit:.5g}", mute=not verbose)
###
###        return space_required_desired_unit

class SpaceEstimator:
    """
    To calculate the required space on the disk. This can be done with an already existing array, or a desired array of
    a certain shape and data type. Useful, to check if it exceeds the current available space.
    It works both for numpy and cupy.
    """

    @overload
    @staticmethod
    def for_array(data, unit: str = "MiB", verbose: bool = False) -> pint.Quantity:
        """
        For getting the size required for the respective array.

        By default, the unit Mebibyte (MiB) is used.
        :param data: numpy or cupy array
        :param unit: unit as string (e.g., MiB, GiB, or MB, GB,...)
        :return: required space as Pint Quantity in the requested unit
        """
        ...

    @overload
    @staticmethod
    def for_array(data_shape: tuple, data_type, unit: str = "MiB", verbose: bool = False) -> pint.Quantity:
        """
        For estimating the size of a desired array.

        :param data_shape: desired shape as tuple
        :param data_type: dtype or dtype specifier (e.g., np.float32, np.dtype("float32"), "float32")
        :param unit: unit as string (e.g., MiB, GiB, or MB, GB,...)
        :return: required space as Pint Quantity in the requested unit
        """
        ...

    @staticmethod
    def for_array(data_or_shape, data_type=None, unit: str = "MiB", verbose: bool = False) -> pint.Quantity:
        """
        Just the runtime implementation for both overload variants.

        :param data_or_shape: numpy/cupy array OR desired shape as tuple
        :param data_type: dtype if data_or_shape is a shape tuple; otherwise, ignored if data_or_shape is an array
        :param unit: unit as string (e.g., MiB, GiB, or MB, GB,...)
        :return: required space as Pint Quantity in the requested unit
        """
        if hasattr(data_or_shape, "shape") and hasattr(data_or_shape, "dtype"):
            shape = data_or_shape.shape
            data_type = np.dtype(data_or_shape.dtype)
            space_required_bytes = int(getattr(data_or_shape, "nbytes", int(np.prod(shape)) * data_type.itemsize))
        else:
            shape = data_or_shape
            if data_type is None:
                Console.printf("error", "When passing a shape tuple also the 'data type' must be provided!")
                sys.exit()
            data_type = np.dtype(data_type)

            space_required_bytes = int(np.prod(shape)) * data_type.itemsize

        space_required_desired_unit = (space_required_bytes * u.byte).to(unit)

        # Print if desired
        Console.printf("info", f"The required space is {space_required_desired_unit:.5g}", mute=not verbose)

        return space_required_desired_unit






###class SpaceEstimator:
###    """
###    To calculate the required space on the disk for a numpy array of a given shape and data type. Usefully, if a big array
###    should be created and to check easily if it does not exceed a certain required space.
###    """
###
###    @staticmethod
###    def for_numpy(data_shape: tuple, data_type: np.dtype, unit: str = "MB") -> np.ndarray:
###        """
###        For estimating the required space of a numpy array with a desired shape and data type.
###
###        :param data_shape: desired shape of the numpy array
###        :param data_type: desired data type of the numpy array (e.g., np.int64)
###        :param unit: desired unit the numpy array
###        :return: required space on disk as a numpy array (with defined unit: [bytes], [KB], [MB], [GB]). Standard unit is [MB].
###        """
###
###        # np.prod...  To get total numpy array size
###        # itemsize... bytes required by each element in array
###        space_required_bytes = np.prod(data_shape) * data_type.itemsize
###
###        if unit == "byte":
###            return space_required_bytes
###
###        elif unit == "KB":
###            return space_required_bytes * 1 / 1024
###
###        elif unit == "MB":
###            return space_required_bytes * 1 / (1024 ** 2)
###
###        elif unit == "GB":
###            return space_required_bytes * 1 / (1024 ** 3)




class JsonConverter:
    """
    To convert unusual Json entries to the desired format. This is for already read in json data. It is not a reader.
    """

    @staticmethod
    def complex_string_list_to_numpy(array: list[str]) -> np.ndarray:
        """
        To convert from list of strings which holds
                ["Re,Im",
                 "Re,Im",
                 "Re,Im",
                 .......]

        ...to numpy array of complex values:
               np.array(Re+Im,
                        Re+Im,
                        Re+Im,
                        .....)

        :param array: list of strings with complex values
        :return: one numpy array of complex numbers
        """
        # (1) From list holding strings of complex numbers to list of numpy complex numbers
        # Explanation:
        #  * list comprehension --> instead of for loop
        #  * 'map'              --> to apply at each entry (2 for complex number)
        #  * starred expression --> to unpack the to resulting entries (Re, Im) for the 'complex' as input
        #
        #   For example: from string: complex(*map("3,5j".split(","))) --> to complex number: (3+5j)
        #
        complex_numbers_numpy = np.asarray([
            complex(*map(float, number.split(',')))  # get both entries and split therefore at ","
            for number in array                                # iterate through all elements ...
        ])

        return complex_numbers_numpy


def deprecated(reason, replacement=None) -> Callable:
    """
    Decoration to mark functions or classes as deprecated and mention the new alternatives. \n

    It consists of the nested functions:

        (1) deprecated => Outer function: pass arguments (reason, replacement) to the decorator.\n
        (2) decorator  => Mid inner function: receives function or class and wraps it and creating warning message. \n
        (3) wrapper    => Inner most function: This function issues the warning and calls the original function or class. \n

    :param reason: Reason for deprecating the function or class (string).
    :param replacement: Suggestion for what to use instead (string). Default is None.
    :return: Nothing
    """

    # This function will act as the actual decorator applied to the function or class.
    def decorator(function_or_class):
        """
        Receives function or class and wraps it and creating warning message

        :param function_or_class:
        :return:
        """
        message = f"{function_or_class.__name__} is deprecated: {reason}."  # deprecated message
        if replacement:
            message += f" Use {replacement} instead."

        # Defining the wrapper function for class or function
        @functools.wraps(function_or_class)
        def wrapper(*args, **kwargs):
            warnings.warn(
                message,
                category=DeprecationWarning,
                stacklevel=2
            )
            return function_or_class(*args, **kwargs)

        return wrapper

    return decorator


class GPUTools:

###    @staticmethod
###    def run_cupy_local_pool(working_function, target_gpu: int = None, return_device: str = 'cpu') -> np.ndarray | cp.ndarray:
###        """
###        Run a cupy function under a local memory pool on a specific GPU.
###        NOTE: If returning a GPU array, we must NOT free the pool blocks here because the returned
###        array may still reference pooled memory.
###
###        IMPORTANT: Pass the function with the arguments a lambda, e.g.: lambda: function, so this creates an anonymous
###        function that can be called later!
###
###        :param return_device: if 'cpu' the converted to numpy array and thus returned on
###        :param target_gpu: the index of the target gpu
###        :param working_function: an anonymous function, created with: lambda: my_function
###        :return: numpy or cupy array
###        """
###        with cp.cuda.Device(target_gpu):
###            pool = cp.cuda.MemoryPool()
###            with cp.cuda.using_allocator(pool.malloc):
###                result = working_function()
###                cp.cuda.get_current_stream().synchronize()
###
###                if return_device == "cpu":
###                    out = cp.asnumpy(result)
###                    # Ensure result is released before freeing the pool
###                    del result
###                    cp.cuda.get_current_stream().synchronize()
###                    pool.free_all_blocks()
###                    return out
###
###                elif return_device in ("gpu", "cuda"):
###                    # Do NOT free pool blocks here; returned array may depend on them.
###                    return result
###
###                else:
###                    Console.printf("error", f"return_device must be 'cpu' or 'gpu'/'cuda', but got '{return_device}'")
###                    return

    @staticmethod
    def run_cupy_local_pool(working_function, target_gpu=None, return_device="cpu"):
        """
        Run a cupy function under a local memory pool on a specific GPU.
        NOTE: If returning a GPU array, we must NOT free the pool blocks here because the returned
        array may still reference pooled memory.

        IMPORTANT: Pass the function with the arguments a lambda, e.g.: lambda: function, so this creates an anonymous
        function that can be called later!

        :param return_device: if 'cpu' the converted to numpy array and thus returned on
        :param target_gpu: the index of the target gpu
        :param working_function: an anonymous function, created with: lambda: my_function
        :return: numpy or cupy array
        """
        with cp.cuda.Device(target_gpu):
            if return_device == "cpu":
                pool = cp.cuda.MemoryPool()
                with cp.cuda.using_allocator(pool.malloc):
                    result = working_function()
                    cp.cuda.get_current_stream().synchronize()
                    out = cp.asnumpy(result)
                    del result
                    cp.cuda.get_current_stream().synchronize()
                    pool.free_all_blocks()
                    return out

            elif return_device in ("gpu", "cuda"):
                # Do not use local pool, otherwise lifetime-problem (e.g. when transferring to other GPU)
                result = working_function()
                cp.cuda.get_current_stream().synchronize()
                return result

            else:
                raise ValueError(...)



    @staticmethod
    def dask_map_blocks(dask_array: da.array, device: str) -> da.array:
        """
        To convert the backbone arrays between numpy to cupy via mapping the chunks. Numpy is for
        cpu computations and cupy for gpu computations.

        :param dask_array: a dask array with numpy or cupy inside.
        :param device: either cpu or gpu/cuda is possible.
        :return: dask array with numpy or cupy inside
        """

        meta = cp.empty((0,) * dask_array.ndim, dtype=dask_array.dtype) if device in ("gpu", "cuda") \
            else np.empty((0,) * dask_array.ndim, dtype=dask_array.dtype)

        return dask_array.map_blocks(GPUTools.to_device,
                                     device=device,
                                     dtype=dask_array.dtype,
                                     meta=meta)
                                     #meta=GPUTools.meta_like(dask_array._meta))


    @staticmethod
    def dask_map_blocks_many(*arrays: da.Array, device: str) -> da.Array | tuple[da.Array, ...]:
        """
        It uses the method 'GPUTools.dask_map_blocks' and accepts multiple arrays as input.
        This is useful, if many arrays need to be brought to the same device.

        :param arrays: one or many (preferred) dask arrays with cupy or numpy inside.
        :param device: either cpu or gpu/cuda is possible.
        :return: a) one dask array or b) a tuple of many dask arrays
        """

        mapped = tuple(GPUTools.dask_map_blocks(a, device=device) for a in arrays)
        return mapped[0] if len(mapped) == 1 else mapped

    @staticmethod
    def to_device(x: np.ndarray|cp.ndarray|pint.Quantity, device, target_gpu:int | None = None, verbose: bool=True) -> np.ndarray|cp.ndarray:
        """
        To convert between cupy and numpy array and thus between gpu and cpu.

        Note: if pint Quantity is inserted it will be converted back to the base array (e.g., numpy or cupy)

        :param x: either cupy or numpy array
        :param device: a string with the possible options 'cpu' and 'gpu'/'cuda'
        :param target_gpu: if 'gpu'/'cuda' is selected then an index for the desired GPU can be specified
        """
        if isinstance(x, pint.Quantity):
            x = UnitTools.remove_unit(x)

        if device == "cpu":
            x_cpu = GPUTools.to_numpy(x, verbose=verbose)
            return x_cpu
        elif device in ("gpu", "cuda"):
            return GPUTools.to_cupy(x, target_gpu=target_gpu, verbose=verbose)
        else:
            raise ValueError(f"Chosen device must be either 'cpu' or 'gpu' or 'cuda', got '{device}'")


    @staticmethod
    def to_cupy(x: np.ndarray | cp.ndarray, target_gpu: int = None, verbose: bool = True) -> cp.ndarray:
        """
        To convert numpy to cupy array. This is mainly to lazy convert numpy array to cupy array
        via dask, to not allocate memory in advance. However, can also be used without dask in combination.

        :param x: a numpy or cupy array that should be converted to a cupy array.
        :param target_gpu: optional gpu index; if set, allocate/copy on the GPU with desired index
        """

        if target_gpu is None:
            if isinstance(x, cp.ndarray):  # already right data type, do nothing
                return x
            elif isinstance(x, np.ndarray):  # convert
                return cp.asarray(x)
            else:  # wrong datatype
                raise TypeError(f"Input must be a numpy or cupy array, got {type(x).__name__}")

        else:
            with cp.cuda.Device(target_gpu):
                # if already on desired GPU
                if isinstance(x, cp.ndarray) and x.device.id == target_gpu:
                    return x
                # if not on desired GPU, need GPU <-> GPU transfer
                else:
                    Console.printf("warning", f"Moved cupy array from GPU {x.device.id} to GPU {target_gpu}. Delete with 'del' old variable and call 'GPUTools.free_cuda_after_del_cupy' to free up unused cuda memory of old variable.", mute=not verbose)
                    x_new = cp.asarray(x)
                    return x_new

    @staticmethod
    def free_cuda_after_del_cupy(device_idx: int):
        """
        Also make sure that the respective variable that holds the cupy array is deleted before

        :param device_idx: GPU device index
        :return: Nothing
        """
        with cp.cuda.Device(device_idx):
            cp.cuda.get_current_stream().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()  # optional, for host-pinned buffers


    @staticmethod
    def to_numpy(x: cp.ndarray, verbose:bool=True) -> np.ndarray:
        """
        To convert a cupy to a numpy array. This is mainly to lazy convert a cupy array to a numpy
        array via dask, to not allocate memory in advance.

        :param x: a cupy array that should be converted to a numpy array.
        """

        if isinstance(x, cp.ndarray):   # convert to cupy array and make user know that GPU memory is still allocated
            Console.printf("warning",
                           f"Cupy array transferred from GPU {x.device.id} to CPU. Delete old cupy variable if not longer used and call 'GPUTools.free_cuda_after_del_cupy(device_idx={x.device.id})'", mute=not verbose)
            return x.get()
        elif isinstance(x, np.ndarray): # already of type numpy array, do nothing
            return x
        else:                           # wrong datatype
            raise TypeError(f"Input must be a cupy array, got {type(x).__name__}")

    @staticmethod
    def meta_like(array):
        # zero-size numpy array with correct ndim/dtype
        return np.empty((0,) * array.ndim, dtype=array.dtype)



class DaskTools:

    @staticmethod
    def meta_for_map_blocks(array: da.Array, device: str):
        """
        To let dask know with which data type it is dealing. Since and array with a zero for each axis is created it does
        not pre-allocate - especially important for GPU VRAM - in advance memory.
        This solves the issue, that although 'dask map blocks' is lazy executed it pre-allocates in advance GPU memory on
        the default GPU, which is typically the GPU with index 0. This behavior is unwanted in most cases.

        (!) Only use together with: dask.map_blocks(..., meta=meta_from_here)

        :param array: dask array
        :param device: either 'cpu' or 'gpu'/'cuda'
        :return: the 'meta data', an array with zero for each axis.
        """

        if device in ("gpu", "cuda"):
            meta = cp.empty((0,) * array.ndim, dtype=array.dtype)
            return meta
        elif device == "cpu":
            meta = np.empty((0,) * array.ndim, dtype=array.dtype)
            return meta
        else:
            Console.printf("error", f"Cannot provide meta for array. Chosen device must be either 'cpu' or 'gpu'/'cuda', got '{device}'.")
            return


    @staticmethod
    def rechunk(array, chunksize: tuple[int, ...]|str) -> da.Array:
        if not isinstance(array, da.Array):
            raise TypeError(f"Cannot rechunk non-dask array. Got: {type(array)!r}")

        if chunksize == "auto":
            return array.rechunk(chunks="auto")
        elif array.chunksize == chunksize:
            return array
        else:
            return array.rechunk(chunks=chunksize)


    @staticmethod
    def to_dask(array: np.ndarray, chunksize: tuple | str = "auto", verbose=False) -> da.Array:
        """
        To safely wrap a numpy or cupy array into a dask array.

        It is possible to:
            -> transform a numpy or cupy array to a dask array
                -> with a certain chunk size
                -> or an 'auto' chunk size chosen by dask
            -> re-chunk a dask array

        Note: If the desired shape is invalid this class will handle it in several ways.
        """
        # TODO: Maybe use new method rechunk here?

        _CHUNK_DIM_MISMATCH = "Chunks and shape must be of the same length/dimension"

        # Invalid chunking or re-chunking
        if isinstance(chunksize, tuple) and len(chunksize) != array.ndim:
            # Case 1: Already dask array, invalid rechunking size -> maintain current size
            if isinstance(array, da.Array):
                if verbose:
                    Console.printf(
                        "error",
                        f"Already dask array, but invalid chunk size chosen; skipping rechunk! \n "
                        f" => You have chosen number dimensions {len(chunksize)}, but array has {array.ndim}. \n "
                        f" => Therefore, the chunk size {array.chunksize} is retained instead of being split into {chunksize}."
                    )
                return array
            # Case 2: Transform to dask array, invalid rechunking size -> maintain current size
            else:
                array = da.asarray(array, chunks="auto")
                if verbose:
                    Console.printf(
                        "warning",
                        f"Transformed to dask array, but invalid chunk size chosen; falling back to 'auto' \n "
                        f" => You have chosen number dimensions {len(chunksize)}, but array has {array.ndim}. \n "
                        f" => Therefore, the chunk size {array.chunksize} is automatically chosen instead of being split into {chunksize}."
                    )
                return array

        try:
            if isinstance(array, da.Array):
                # Case 3: Already dask array, maintaining chunksize
                if (chunksize == "auto") or (isinstance(chunksize, tuple) and array.chunksize == chunksize):
                    return array
                # Case 4: Already dask array, rechunking due to valid chunksize
                else:
                    return array.rechunk(chunks=chunksize)
            else:
                # Case 5: Transform to dask array, either via 'auto' chunk size or desired chunk size
                return da.asarray(array, chunks="auto" if chunksize == "auto" else chunksize)


        except ValueError as e:
            if isinstance(array, da.Array):
                if verbose:
                    Console.printf("error", f"Invalid chunk size; skipping rechunk: {e}")
                return array
            else:
                if verbose:
                    Console.printf("warning", f"Invalid chunk size; falling back to 'auto': {e}")

    @staticmethod
    def to_zarr(array: da.Array, path, checkpoint_folder_name, overwrite: bool=True, lazy: bool=True) -> da.Array:
        """
        This is to write dask array to disk. Ideally, apply before compute().

        :param array: dask array
        :param path_and_folder: path and folder as string
        :param overwrite: overwrites existing checkpoint
        :param lazy: if True then .compute() required afterward. If not lazy, the writes at the moment to the disk.
        :return: Nothing
        """
        return array.to_zarr(os.path.join(path, checkpoint_folder_name), overwrite=overwrite, compute=not lazy)


class FourierTools:
    """
    For any kind of Fourier Transformation. In future should also contain non cartesian Fourier Transformation.
    """
    @staticmethod
    def cartesian_FT_dask(array: da.Array, direction: str, fft_shift: bool = True, axes: tuple = None, device: str = 'cpu', data_type_output=None, verbose: bool=True):

        if axes is None:
            axes = tuple(range(array.ndim))
            Console.printf("warning", "No axes were provided. Therefore, performing FT to all axes!")

        # (1) Check if possible options are selected
        possible_directions = ("forward", "inverse")
        if direction not in possible_directions:
            Console.printf("error", f"Can only apply FT in the following directions: {possible_directions}. But it was given: {direction}.")
            return

        # (2) Define the right numpy and cupy functions
        if device in ("gpu", "cuda"):
            fftn = cp.fft.fftn
            fftshift = cp.fft.fftshift
            ifftshift = cp.fft.ifftshift
            ifftn = cp.fft.ifftn
        elif device == "cpu":
            fftn = np.fft.fftn
            fftshift = np.fft.fftshift
            ifftshift = np.fft.ifftshift
            ifftn = np.fft.ifftn
        else:
            Console.printf("error", f"Only possible to select device 'cpu' or 'gpu'/'cuda'. But it was given: {device}")
            return

        # (3) Rechunking
        # Make axes which should be FFT to one chunk (via -1) and keep chunksize for other dimensions
        chunksize = np.array(array.chunksize)
        axes = np.array(axes) # (!) only convert to numpy array to enable specific numpy indexing policy. Later need to re-convert to tuple.
        chunksize[axes] = -1
        axes = tuple(axes) # back to tuple for map_blocks

        chunksize_old = array.chunksize
        array = array.rechunk(chunksize) # <- keep dimensions that will not be affected by the FFT. Rechunking is a data-movement operation, therefore doing on old device.

        # (4) Bring array to desired device
        array = GPUTools.dask_map_blocks(array, device=device)

        # (5) To ensure to get the right data type for the output
        array_dtype_old = array.dtype
        data_type_output = FourierTools._ensure_complex_dtype(array_dtype=array.dtype, data_type=data_type_output)


        # (6) (!) Important to provide the meta if map.blocks is used in dask, otherise if device gpu and no meta then it allocates already cuda memory in advance!!!
        meta = DaskTools.meta_for_map_blocks(array=array, device=device)

        # (6) Perform fftshift and fft or ifft
        if direction == "inverse":
            if fft_shift:
                array = array.map_blocks(ifftshift, axes=axes, dtype=data_type_output, meta=meta)
            array = array.map_blocks(ifftn, axes=axes, dtype=data_type_output, meta=meta)
        elif direction == "forward":
            array = array.map_blocks(fftn, axes=axes, dtype=data_type_output, meta=meta)
            if fft_shift:
                array = array.map_blocks(fftshift, axes=axes, dtype=data_type_output, meta=meta)



        Console.add_lines(f"Adding Fourier Transformation to Dask Graph:")
        Console.add_lines(f" => {'Direction:':.<25} {direction}")
        Console.add_lines(f" => {'Applying (i)FFT-shift:':.<25} {fft_shift}")
        Console.add_lines(f" => {'Affected axes:':.<25} {axes} of {len(array.shape)} total axes")
        Console.add_lines(f" => {'Rechunking:':.<25} {chunksize_old} => {tuple(array.chunksize)}")
        Console.add_lines(f" => {'Data type:':.<25} {array_dtype_old} => {data_type_output}")
        Console.add_lines(f" => {'Performing on device:':.<25} {device}")
        Console.printf_collected_lines("success", mute=not verbose)

        return array

    @staticmethod
    def _ensure_complex_dtype(array_dtype, data_type=None):
        """
        Ensure the FT output dtype is complex.

        Cases:
            If data_type is None:
                -> If input os complex: then keep the same complex precision (e.g., complex64)
                -> If input is float32: promote to complex64
                -> Otherwise (e.g., float64, int, ...): promote to complex 128.
            If data_type is provided but not complex: force complex64
            If data_type is already complex: then keep it
        """
        input_dtype = np.dtype(array_dtype)

        # Case: No data type is explicitly provided. Infer it from the input.
        if data_type is None:
            # Preserve the complex precision of provided array if the input is already complex
            if np.issubdtype(input_dtype, np.complexfloating):
                return input_dtype

            # Promote real input to an appropriate complex dtype
            if input_dtype == np.float32:
                return np.complex64

            # The default is complex128 (e.g., for float64 and most other real dtypes)
            return np.complex128

        # Case: A desired data type is provided by the user
        else:
            requested_dtype = np.dtype(data_type)

            # But if the requested dtype is not complex, force a complex data type...
            if not np.issubdtype(requested_dtype, np.complexfloating):
                return np.complex64

            # Requested dtype is already complex, then it is simply returned
            return requested_dtype












class SortingTools:

    @staticmethod
    def sort_dict_reference(reference_dictionary, *dictionaries) -> tuple[dict, ...]:
        """
        To sort all given dictionaries relative to a reference set (first argument). It sorts, based on the keys, the keys
        and the values.

        :param reference_dictionary: simply a reference dictionary
        :param dictionaries: simply the dictionaries which should get sorted.
        :return: Sorted sets including the reference set in the same order as the input.
        """
        Console.printf(
            "info",
            "Sorting the dictionaries via their keys based on the reference dictionary.")

        if not SortingTools.check_dict_same_keys(reference_dictionary, *dictionaries):
            Console.printf(
                "error",
                "Not all dictionaries obtaining the same keys! Therefore, returning unsorted dictionaries.")
            return reference_dictionary, *dictionaries
        else:
            return reference_dictionary, *({k: d.get(k) for k in reference_dictionary} for d in dictionaries) # to return reference dict and sorted dicts

    @staticmethod
    def check_dict_same_keys(*dictionaries) -> bool:
        """
        To check if all given dictionaries obtaining the same keys.

        :param dictionaries: dictionaries to compare
        :return: True or False regarding obtaining the same keys of the respective dictionaries
        """
        all_keys = set().union(*(d.keys() for d in dictionaries))
        return all(set(d.keys()) == all_keys for d in dictionaries)


class UnitTools:

    @staticmethod
    def to_base(values: np.ndarray | int | float | u.Quantity, units: pint.Unit=None | str, return_separate=False, verbose=False) -> pint.Quantity | tuple:
        """
        This converts to the base in the selected unit system. For example cm --> m, mmol -> mol, and so on.

        As input, it is possible to insert single values or array (e.g., numpy array), or already a pint Quantity which
        then contains values.magnitude and value.units. For the units it is possible to insert a Unit object from pint
        or also a string (e.g., 'mm').
        For the output, two options are available either a pint Quantity object with the converted values to the base
        unit or separate the converted values and the unit as Unit object from pint.

        :param values: single value or array
        :param units: pint Unit object ot pint conform string
        :param return_separate: whether a pint Quantity object should be returned or a tuple with (converted values, pint units)
        :param verbose: if true, then it prints the results including the old and new min max values for verification purposes.
        :return: either a pint Quantity or a tuple (values, units)
        """
        if isinstance(values, pint.Quantity):
            quantity = values
        else:
            quantity = u.Quantity(values, units)

        quantity_before = quantity
        quantity = quantity.to_base_units()
        values = quantity.magnitude
        units = quantity.units

        if verbose is True:
            Console.add_lines(f"{'Converted units':<25} {quantity_before.units}' ==> '{quantity.units}'")
            Console.add_lines(f"{'Values range before:':<25} [{np.min(quantity_before.magnitude):<7.2f}, {np.max(quantity_before.magnitude):>7.2f}]")
            Console.add_lines(f"{'Values range after:':<25} [{np.min(quantity.magnitude):<7.2f}, {np.max(quantity.magnitude):>7.2f}]")

        if return_separate:
            Console.add_lines(f"{'Note: Return separate':<25} (values, pint unit)")
            result = values, units
        else:
            result = quantity
            Console.add_lines("Note: Return one pint Quantity with Quantity.magnitude, Quantity.units")

        Console.printf_collected_lines("success", mute=not verbose)

        return result

    @staticmethod
    def to_unit(values: np.ndarray | pint.Quantity,
                current_units: pint.Unit | str | None = None,
                target_units: pint.Unit | str | None = None,
                return_separate: bool = False,
                verbose: bool = False) -> pint.Quantity | tuple:
        """
        This performs a unit conversion by applying a conversion factor (e.g., multiplying meters by 1000 to return millimeters).

        :param values: an array or pint Quantity
        :param current_units: as string (e.g. mm)
        :param target_units: as string (e.g. cm)
        :param return_separate: if true: then return (array, pint unit)
        :param verbose: if true output progress and conversation
        :return: either a pint Quantity (with .magnitude, .units) or .magnitude (array) and .units (pint.Unit) separate
        """

        quantity_before = None # just initial value
        quantity_after = None  # just initial value

        # Error case #1: Provide either a pint Quantity or at least units
        if not isinstance(values, pint.Quantity) and current_units is None:
            Console.printf("error", "Need either a pint Quantity or the current unit provided separately.")
        # Error case #2: A pint Quantity is provided but also separately also units
        elif isinstance(values, pint.Quantity) and current_units is not None:
            Console.printf("error", "A pint Quantity provided and also separate units. Just use one.")
        # Possible cases:
        else:
            # A) A pint quantity is provided
            if isinstance(values, pint.Quantity):
                quantity_before = values
                quantity_after = values.to(target_units)              # convert to desired unit (e.g. mm -> m)
                current_units = values.units
            # B) Separate units are provided
            elif target_units is not None:
                quantity_before = u.Quantity(values, current_units)   # convert to pint quantity and
                quantity_after = quantity_before.to(target_units)     # convert to desired unit (e.g. mm -> m)
            # Error: No case fits
            else:
                Console.printf("error", "A different error occurred while trying to convert units.")

            # In any case now a pint Quantity is given at this step
            Console.add_lines(f"{'Converted units':<25} '{current_units}' ==> '{target_units}'")
            Console.add_lines(f"{'Values range before:':<25} [{np.min(quantity_before.magnitude):<7.2g}, {np.max(quantity_before.magnitude):>7.2g}]")
            Console.add_lines(f"{'Values range after:':<25} [{np.min(quantity_after.magnitude):<7.2g}, {np.max(quantity_after.magnitude):>7.2g}]")
            Console.printf_collected_lines("success", mute=not verbose)

            if not return_separate:
                return quantity_after
            else:
                return quantity_after.magnitude, quantity_after.units

    @staticmethod
    def remove_unit(array: pint.Quantity | np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray | int | float:
        """
        If pint.Quantity is inserted then the underlying array/values are returned.

        :param array: an array
        :return: plain array, int, float and so on
        """

        if isinstance(array, pint.Quantity):
            return array.magnitude
        else:
            return array

class ArrayTools:

    @staticmethod
    def to_data_type(array: np.ndarray | cp.ndarray, data_type: str, verbose: bool = True):
        """
        Change the dtype of a numpy or cupy array.

        :param array: input array (numpy or cupy)
        :param data_type: target dtype as string (e.g. "float32", "float64", "complex128")
        :param verbose: Print to console if True
        :return: array with changed dtype
        """
        xp = ArrayTools.get_backend(array)

        data_type_before = array.dtype
        array = array.astype(xp.dtype(data_type))
        data_type_after = array.dtype

        if data_type_before != data_type_after:
            Console.printf("success", f"Changed the data type from {data_type_before} -> {data_type_after}", mute=not verbose)
        else:
            Console.printf("warning", f"Data type already {data_type_after}. No change was made.", mute=not verbose)

        return array

    @staticmethod
    def get_backend(array: np.ndarray | cp.ndarray):
        """
        Get the cupy or numpy module, based on the array. Then, the module can be used for example for module.array([]...).

        :param array: input array (numpy or cupy)
        :return: numpy or cupy module
        """
        return cp.get_array_module(array)

    @staticmethod
    def enforce_min_eps(array: np.ndarray | cp.ndarray,
               eps: float | None = None,
               convert_negative: bool = True,
               convert_zeros: bool = True,
               verbose: str = True,
    ):
        """
        Convert selected values to a positive epsilon value:

        - convert_negative=True  -> values < 0 become eps
        - convert_zeros=True     -> values == 0 become eps
        - both True              -> values <= 0 become eps
        - both False             -> array unchanged

        :param array: numpy or cupy array
        :param eps: if None then default eps is used. Otherwise, eps need float.
        :param convert_negative: if True then values < 0 are converted to eps
        :param convert_zeros:  if True then values == 0 are to eps
        :param verbose:
        :return:
        """
        xp = ArrayTools.get_backend(array)

        if xp.issubdtype(array.dtype, xp.complexfloating):
            Console.printf("error", "Complex values are not yet supported! Returning array unchanged.")
            return array

        if eps is None:
            if xp.issubdtype(array.dtype, xp.integer):
                array = array.astype(xp.float32, copy=False)
                eps = xp.finfo(array.dtype).eps
                Console.printf(
                    "warning",
                    f"No eps was given and input array is of type {xp.integer}. Therefore using float32 eps: {eps}",
                )
            else:
                eps = xp.finfo(array.dtype).eps


        convert_cases_str = ""
        # Case: convert negative and zeros
        if convert_negative and convert_zeros:
            convert_cases_str = "negative and zero"
            xp.maximum(array, eps, out=array, where=(array <= 0))
        # Case: only convert negative
        elif convert_negative:
            convert_cases_str = "negative"
            xp.maximum(array, eps, out=array, where=(array < 0))
        # Case: only convert zeros
        elif convert_zeros:
            convert_cases_str = "zero"
            xp.maximum(array, eps, out=array, where=(array == 0))
        else:
            pass # do nothing at this place. Maybe future implementations take place here.

        Console.printf("success", f"Converted {convert_cases_str} values to eps: {eps}", mute=not verbose)


        return array

    @staticmethod
    def check_nan(array: np.ndarray | cp.ndarray, verbose:bool = True) -> bool:
        """
        To check if NaNs are present. Numpy or cupy array is possible.


        :param array: numpy or cupy array
        :return: True if NaNs are present
        """

        xp = ArrayTools.get_backend(array)

        size_array = int(array.size)

        nan_count = xp.isnan(array).sum()
        has_nan = xp.isnan(array).any()

        nan_percent = (nan_count / size_array * 100.0) if size_array else 0.0

        if has_nan:
            if verbose:
                Console.printf("warning", f"NaNs are presents ({nan_percent:.5g})%. NaNs/(array size): {nan_count}/{size_array}.")
            return True
        else:
            return False

    @staticmethod
    def count_zeros(array: np.ndarray | cp.ndarray, verbose:bool = True):
        """
        To count the zeros present in the array. E.g., useful to check for 0/0 division.

        :param array: numpy or cupy array
        :param verbose: Print number of found zeros to console
        :return: number of zeros found
        """

        number_zeros = ArrayTools.get_backend(array).sum(array == 0)

        Console.printf("info", f"Found number of zeros: {number_zeros}", mute=not verbose)

        return number_zeros

class InterpolationTools:
    ## TODO: here also add fft interpolation (e.g., which used for the FID?)

    @staticmethod
    def interpolate(array: np.ndarray | cp.ndarray,
                    target_size: tuple,
                    order: int,
                    compute_on_device: str = "cuda",
                    return_on_device: str = "cpu",
                    target_gpu: int = 0,
                    verbose: bool = True) -> np.ndarray | cp.ndarray:
        """
        To interpolate on cpu or gpu via zoom (gpu: cupyx.scipy.ndimage.zoom; cpu: scipy.ndimage.zoom). Therefore, it
        is possible to interpolate on the desired target device (gpu or cpu) and if gpu is chosen also the device index
        is possible to choose. Further, it can be chosen if it should be returned on cpu or gpu.

        :param verbose: if True then print result to console (old and new shape and size)
        :param array: numpy or cupy array
        :param target_size: the desired target size as tuple
        :param order: the order (see scipy documentation)
        :param compute_on_device: desired device ('cpu', or 'gpu'/'cuda')
        :param return_on_device: desire device ('cpu', or 'gpu'/'cuda')
        :param target_gpu: if computation on gpu then the index of the gpu can be selected
        :return: numpy or cupy array
        """

        shape_old = array.shape
        array_old = array
        zoom_factor = np.divide(target_size, shape_old)

        # Case 1: compute on the cpu (no need for specifying the device index)
        if compute_on_device == 'cpu':
            zoom = zoom_cpu
            # if is cupy array then convert to numpy first
            if isinstance(array, cp.ndarray):
                array = cp.asnumpy(array)
            array = zoom(input=array, zoom=zoom_factor, order=order)

            # NOTE: do NOT convert to GPU here; do it once at the end so return_on_device is enforced consistently

        # Case 2: compute on gpu (need to specify the gpu index in multi-gpu setting)
        elif compute_on_device in ("cuda", "gpu"):
            zoom = zoom_gpu
            Console.printf("info", f"Interpolate on GPU: {target_gpu}")

            array = GPUTools.run_cupy_local_pool(
                working_function=lambda: zoom(
                    input=(array if isinstance(array, cp.ndarray) else cp.asarray(array)),
                    zoom=zoom_factor,
                    order=order
                ),
                target_gpu=target_gpu,
                return_device=return_on_device
            )

        else:
            Console.printf("error", f"Invalid compute_on_device='{compute_on_device}'. Use 'cpu' or 'cuda'/'gpu'.")
            return array

        # To return on desired device
        if return_on_device == "cpu":
            if isinstance(array, cp.ndarray):
                array = cp.asnumpy(array)

        elif return_on_device in ("gpu", "cuda"):
            # Respect target_gpu even when converting CPU result -> GPU
            with cp.cuda.Device(target_gpu):
                if not isinstance(array, cp.ndarray):
                    array = cp.asarray(array)
                else:
                    # If already on GPU but not the target GPU, move/copy to target GPU
                    if array.device.id != target_gpu:
                        array = cp.asarray(array)

        else:
            Console.printf("error", f"Invalid return_on_device='{return_on_device}'. Use 'cpu' or 'cuda'/'gpu'.")
            return array

        Console.printf(
            "success",
            f"Interpolated on {compute_on_device} and returned on {return_on_device} \n"
            f"Shape {shape_old} ({SpaceEstimator.for_array(array_old)}) => {array.shape} ({SpaceEstimator.for_array(array)})",
            mute=not verbose
        )
        return array



class JupyterPlotManager:
    """
    Convenience wrapper for the volume grid viewer.

    Usage:
        fig, axs, state = JupyterPlotManager.volume_grid_viewer(
            vols, rows=1, cols=8, titles=titles
        )

    Defaults are intentionally set to the "perfect" configuration you currently have.
    You can still override any keyword argument supported by the underlying implementation.
    """

    @staticmethod
    def get_slice(vol, axis, i):
        if axis == 0:
            return vol[i, :, :]
        if axis == 1:
            return vol[:, i, :]
        return vol[:, :, i]

    @staticmethod
    def _auto_figsize_px(
        rows, cols,
        *,
        unit_img_px,
        internal_h_px,
        unit_wgap_px,
        unit_hgap_px,
        left_panel_px,
        left_panel_gap_px,
        outer_pad_px=(30, 20, 30, 30),  # (left, right, bottom, top)
        dpi=100,
    ):
        pad_l, pad_r, pad_b, pad_t = outer_pad_px
        unit_w_px = unit_img_px
        unit_h_px = unit_img_px + internal_h_px

        grid_w_px = cols * unit_w_px + (cols - 1) * unit_wgap_px
        grid_h_px = rows * unit_h_px + (rows - 1) * unit_hgap_px

        fig_w_px = pad_l + left_panel_px + (left_panel_gap_px if left_panel_px > 0 else 0) + grid_w_px + pad_r
        fig_h_px = pad_b + grid_h_px + pad_t

        return (fig_w_px / dpi, fig_h_px / dpi)

    @staticmethod
    def volume_grid_viewer(
        vols,
        rows,
        cols,
        titles=None,
        **kwargs,
    ):
        """
        Minimal required args: vols, rows, cols, titles (titles optional)

        All other settings default to your current "perfect" configuration, and can be overridden via **kwargs.
        """

        # --- Defaults (your current "perfect" call configuration) ---
        defaults = dict(
            figsize="auto",
            auto_figsize=False,
            dpi=None,
            unit_img_px=220,
            axis=0,

            enable_axis_change=True,
            per_panel_axis=True,
            enable_click_select=True,
            enable_keyboard_nav=True,
            enable_markers=True,

            # margins
            left=0.05, right=0.98, bottom=0.08, top=0.95,

            # widget sizing
            slider_h_px=18,
            slider_pad_px=20,   # NOTE: your call uses 20
            slider_len=0.70,
            cbar_h_px=14,
            cbar_gap_px=5,      # NOTE: your call uses 5

            # gaps
            unit_wgap_px=6,
            unit_hgap_px=30,
            left_panel_gap_px=12,

            grid_align_x="left",
            grid_align_y="center",
            panel_box_rect=(0.03, 0.68, 0.08, 0.05),
        )

        # user overrides
        cfg = {**defaults, **kwargs}

        # ---- Implementation (same as your current perfect version) ----

        def get_slice(vol, axis, i):
            return JupyterPlotManager.get_slice(vol, axis, i)

        def _auto_figsize_px(*args, **kws):
            return JupyterPlotManager._auto_figsize_px(*args, **kws)

        n_panels = rows * cols
        vols = list(vols)[:n_panels]

        if titles is None:
            titles = [f"Vol {i+1}" for i in range(len(vols))]
        else:
            titles = list(titles) + [f"Vol {i+1}" for i in range(len(titles), len(vols))]
            titles = titles[:len(vols)]

        dpi = cfg["dpi"]
        if dpi is None:
            dpi = float(plt.rcParams.get("figure.dpi", 100))

        cbar_h_px = cfg["cbar_h_px"]
        cbar_gap_px = cfg["cbar_gap_px"]
        slider_h_px = cfg["slider_h_px"]
        slider_pad_px = cfg["slider_pad_px"]
        internal_h_px = cbar_h_px + cbar_gap_px + slider_h_px + slider_pad_px

        enable_axis_change = cfg["enable_axis_change"]
        per_panel_axis = cfg["per_panel_axis"]
        enable_markers = cfg["enable_markers"]

        left_panel_px = 0
        if (enable_axis_change and per_panel_axis) or enable_markers:
            left_panel_px = 78

        figsize = cfg["figsize"]
        if cfg["auto_figsize"] or (isinstance(figsize, str) and figsize.lower() == "auto"):
            figsize = _auto_figsize_px(
                rows, cols,
                unit_img_px=cfg["unit_img_px"],
                internal_h_px=internal_h_px,
                unit_wgap_px=cfg["unit_wgap_px"],
                unit_hgap_px=cfg["unit_hgap_px"],
                left_panel_px=left_panel_px,
                left_panel_gap_px=cfg["left_panel_gap_px"],
                dpi=dpi,
            )

        fig, axs = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
        axs = np.array(axs).reshape(rows, cols)
        fig.subplots_adjust(left=cfg["left"], right=cfg["right"], bottom=cfg["bottom"], top=cfg["top"])

        images = {}
        sliders = {}
        slider_axes = {}
        cbar_axes = {}
        cbars = {}
        frames = {}

        axis_per_panel = [int(cfg["axis"])] * len(vols)
        selected_panel = [0]

        axes_to_panel = {}
        image_axes_set = set()

        # per-panel axis badge (top-left, yellow)
        axis_badge = {}

        # Left UI (axis stuff)
        radio = None
        radio_ax = None
        panel_box = None
        panel_box_ax = None
        slice_box = None
        slice_box_ax = None
        slice_box_guard = [False]
        panel_label = None
        slice_label = None

        # Marker UI
        mark_toggle_ax = None
        mark_toggle_btn = None
        undo_ax = None
        undo_btn = None
        clear_ax = None
        clear_btn = None

        marking_enabled = [False]
        marker_stack = []
        marker_counter = [0]

        # caches for "real coordinates" readout
        slice_hw = {}
        slice_off = {}
        slice_data = {}

        # slider ticks: (panel, axis, k) -> count / line
        marker_counts = {}
        slider_tick_lines = {}

        # Left UI sizes (px)
        BOX_W_PX = 78
        BOX_H_PX = 22
        BOX_GAP_PX = 28
        LABEL_PAD_PX = 3

        RADIO_W_PX = BOX_W_PX
        RADIO_H_PX = 70
        RADIO_GAP_PX = 10

        BTN_W_PX = BOX_W_PX
        BTN_H_PX = 22
        BTN_GAP_PX = 8

        left_panel_anchor_px = [None, None]

        AXIS_LABELS = ("axis 1", "axis 2", "axis 3")
        LABEL_TO_AXIS = {lab: i for i, lab in enumerate(AXIS_LABELS)}

        MARKER_REMOVE_TOL_PX = 10

        def canvas_wh_px():
            return fig.canvas.get_width_height()

        def set_slider_range(sl, n_slices, new_val):
            new_val = int(np.clip(int(new_val), 0, n_slices - 1))
            sl.eventson = False
            sl.valmin = 0
            sl.valmax = n_slices - 1
            sl.ax.set_xlim(0, n_slices - 1)
            sl.valstep = 1
            sl.set_val(new_val)
            sl.eventson = True
            return new_val

        def centered_extent(frame_h, frame_w, h, w):
            x0 = (frame_w - w) / 2.0
            y0 = (frame_h - h) / 2.0
            return (x0 - 0.5, x0 + w - 0.5, y0 + h - 0.5, y0 - 0.5)

        def update_selected_slice_box():
            if slice_box is None:
                return
            i = selected_panel[0]
            if i not in sliders or slice_box_guard[0]:
                return
            slice_box_guard[0] = True
            try:
                slice_box.set_val(str(int(sliders[i].val)))
            finally:
                slice_box_guard[0] = False

        def _index_to_label(n: int) -> str:
            n = int(n)
            s = ""
            while True:
                n, r = divmod(n, 26)
                s = chr(ord("A") + r) + s
                if n == 0:
                    break
                n -= 1
            return s

        VALUE_FMT = "{:.4g}"

        def _slice_offsets_in_frame(panel_i: int, slc_shape):
            frame_h, frame_w = frames[panel_i]
            h, w = slc_shape
            x0 = (frame_w - w) / 2.0
            y0 = (frame_h - h) / 2.0
            return x0, y0

        def _update_slice_cache(panel_i: int, slc: np.ndarray):
            h, w = slc.shape
            x0, y0 = _slice_offsets_in_frame(panel_i, (h, w))
            slice_hw[panel_i] = (h, w)
            slice_off[panel_i] = (x0, y0)
            slice_data[panel_i] = slc

        def _axis_short(a: int) -> str:
            return ("Scroll 1", "Scroll 2", "Scroll 3")[int(a)]

        def update_axis_badge(panel_i: int):
            if panel_i in axis_badge:
                axis_badge[panel_i].set_text(f"{_axis_short(axis_per_panel[panel_i])}")

        def make_format_coord(panel_i: int):
            def _fmt(xf, yf):
                if panel_i not in slice_hw or panel_i not in slice_off:
                    return ""
                h, w = slice_hw[panel_i]
                x0, y0 = slice_off[panel_i]

                xs = xf - x0
                ys = yf - y0
                if not (0 <= xs < w and 0 <= ys < h):
                    return ""

                xi = int(np.floor(xs + 0.5))
                yi = int(np.floor(ys + 0.5))
                xi = int(np.clip(xi, 0, w - 1))
                yi = int(np.clip(yi, 0, h - 1))

                slc = slice_data.get(panel_i, None)
                v_txt = "?"
                if slc is not None and slc.shape == (h, w):
                    v_txt = VALUE_FMT.format(float(slc[yi, xi]))

                ax_sel = axis_per_panel[panel_i]
                k = int(sliders[panel_i].val) if panel_i in sliders else 0

                if ax_sel == 0:
                    z, yv, xv = k, yi, xi
                elif ax_sel == 1:
                    z, yv, xv = yi, k, xi
                else:
                    z, yv, xv = yi, xi, k

                return f"vol[{z},{yv},{xv}] = {v_txt}"
            return _fmt

        # slider tick management
        def _mk_key(panel_i: int, ax_sel: int, k: int):
            return (int(panel_i), int(ax_sel), int(k))

        def _ensure_slider_tick(panel_i: int, ax_sel: int, k: int):
            key = _mk_key(panel_i, ax_sel, k)
            if key in slider_tick_lines:
                return
            sax = slider_axes.get(panel_i, None)
            if sax is None:
                return
            ln = sax.axvline(k, ymin=0.0, ymax=1.0, color="yellow", lw=1.2, alpha=0.9, zorder=50)
            slider_tick_lines[key] = ln

        def _remove_slider_tick(panel_i: int, ax_sel: int, k: int):
            key = _mk_key(panel_i, ax_sel, k)
            ln = slider_tick_lines.pop(key, None)
            if ln is not None:
                try:
                    ln.remove()
                except Exception:
                    pass

        def _refresh_slider_ticks_for_panel(panel_i: int):
            cur_axis = int(axis_per_panel[panel_i])
            for (p, a, _k), ln in list(slider_tick_lines.items()):
                if p != panel_i:
                    continue
                ln.set_visible(a == cur_axis)

        def _inc_marker_count(panel_i: int, ax_sel: int, k: int):
            key = _mk_key(panel_i, ax_sel, k)
            marker_counts[key] = marker_counts.get(key, 0) + 1
            if marker_counts[key] == 1:
                _ensure_slider_tick(panel_i, ax_sel, k)
                _refresh_slider_ticks_for_panel(panel_i)

        def _dec_marker_count(panel_i: int, ax_sel: int, k: int):
            key = _mk_key(panel_i, ax_sel, k)
            if key not in marker_counts:
                return
            marker_counts[key] -= 1
            if marker_counts[key] <= 0:
                marker_counts.pop(key, None)
                _remove_slider_tick(panel_i, ax_sel, k)
                _refresh_slider_ticks_for_panel(panel_i)

        # marker helpers
        def refresh_markers_for_panel(panel_i: int):
            if not marker_stack or panel_i not in sliders:
                return

            ax_sel = axis_per_panel[panel_i]
            k_now = int(sliders[panel_i].val)

            if panel_i in slice_off:
                x0, y0 = slice_off[panel_i]
            else:
                slc = get_slice(vols[panel_i], ax_sel, k_now)
                h, w = slc.shape
                x0, y0 = _slice_offsets_in_frame(panel_i, (h, w))

            for m in marker_stack:
                if m["panel"] != panel_i:
                    continue
                visible = (m["axis"] == ax_sel and m["k"] == k_now)
                m["line"].set_visible(visible)
                m["ann"].set_visible(visible)
                if visible:
                    xf = x0 + m["x"]
                    yf = y0 + m["y"]
                    m["line"].set_data([xf], [yf])
                    m["ann"].xy = (xf, yf)

        def _find_marker_hit(panel_i: int, event, x0: float, y0: float):
            if event.x is None or event.y is None:
                return None
            ax_sel = int(axis_per_panel[panel_i])
            k_now = int(sliders[panel_i].val)

            ax = axs.ravel()[panel_i]
            click_xy_disp = np.array([event.x, event.y], dtype=float)

            best = None
            best_d = float("inf")

            for m in marker_stack:
                if m["panel"] != panel_i or m["axis"] != ax_sel or m["k"] != k_now:
                    continue
                xf = x0 + m["x"]
                yf = y0 + m["y"]
                pt_disp = np.array(ax.transData.transform((xf, yf)), dtype=float)
                d = float(np.hypot(*(pt_disp - click_xy_disp)))
                if d <= MARKER_REMOVE_TOL_PX and d < best_d:
                    best_d = d
                    best = m
            return best

        def _remove_marker(m, *, adjust_counter_if_last: bool):
            _dec_marker_count(m["panel"], m["axis"], m["k"])
            try:
                m["line"].remove()
            except Exception:
                pass
            try:
                m["ann"].remove()
            except Exception:
                pass

            try:
                idx = marker_stack.index(m)
            except Exception:
                idx = None

            if idx is not None:
                is_last = (idx == len(marker_stack) - 1)
                marker_stack.pop(idx)
                if adjust_counter_if_last and is_last:
                    marker_counter[0] = max(0, marker_counter[0] - 1)

        def update_panel(i, k):
            vol = vols[i]
            ax_sel = axis_per_panel[i]

            slc = get_slice(vol, ax_sel, int(k))
            _update_slice_cache(i, slc)

            h, w = slc.shape
            frame_h, frame_w = frames[i]
            ext = centered_extent(frame_h, frame_w, h, w)

            im = images[i]
            im.set_data(slc)
            im.set_extent(ext)

            n = vol.shape[ax_sel]
            axs.ravel()[i].set_title(f"{titles[i]}")
            sliders[i].valtext.set_text(f"{int(k)}/{n-1}")

            update_axis_badge(i)

            if enable_markers:
                refresh_markers_for_panel(i)
                _refresh_slider_ticks_for_panel(i)

            fig.canvas.draw_idle()

        def set_selected_panel(i):
            i = int(np.clip(i, 0, len(vols) - 1))
            selected_panel[0] = i

            if panel_box is not None:
                panel_box.set_val(str(i + 1))

            if radio is not None:
                want_axis = axis_per_panel[i]
                want_label = AXIS_LABELS[want_axis]
                if getattr(radio, "value_selected", None) != want_label:
                    radio.set_active(want_axis)

            if per_panel_axis:
                update_selected_slice_box()

            fig.canvas.draw_idle()

        # ---------------- build panels ----------------
        for i, ax in enumerate(axs.ravel()):
            if i >= len(vols):
                ax.axis("off")
                continue

            vol = vols[i]
            Z, Y, X = vol.shape

            frame_h = max(Z, Y)
            frame_w = max(X, Y)
            frames[i] = (frame_h, frame_w)

            ax_sel = axis_per_panel[i]
            n = vol.shape[ax_sel]
            idx0 = n // 2

            vmin = float(vol.min())
            vmax = float(vol.max())

            slc0 = get_slice(vol, ax_sel, idx0)
            _update_slice_cache(i, slc0)

            h0, w0 = slc0.shape
            ext0 = centered_extent(frame_h, frame_w, h0, w0)

            #im = ax.imshow(slc0, cmap="gray", vmin=vmin, vmax=vmax, extent=ext0)
            # choose cmap: either a single cmap or a per-panel list/tuple
            cmap_cfg = cfg.get("cmap", "gray")
            cmap_i = cmap_cfg[i] if isinstance(cmap_cfg, (list, tuple)) else cmap_cfg
            im = ax.imshow(slc0, cmap=cmap_i, vmin=vmin, vmax=vmax, extent=ext0)

            images[i] = im
            im.format_cursor_data = lambda data: ""

            ax.set_aspect("equal", adjustable="box")
            ax.set_anchor("C")
            ax.set_xlim(-0.5, frame_w - 0.5)
            ax.set_ylim(frame_h - 0.5, -0.5)

            ax.set_title(f"{titles[i]}")
            ax.set_xticks([]); ax.set_yticks([])

            ax.format_coord = make_format_coord(i)

            t = ax.text(
                0.02, 0.98, "",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=9,
                color="yellow",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.35, lw=0),
                zorder=10
            )
            axis_badge[i] = t
            update_axis_badge(i)

            # colorbar
            cax = fig.add_axes([0, 0, 0.01, 0.01])
            cbar_axes[i] = cax
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=7, pad=1)
            cb.outline.set_visible(False)
            for sp in cb.ax.spines.values():
                sp.set_visible(False)
            cbars[i] = cb

            # slider
            sax = fig.add_axes([0, 0, 0.01, 0.01])
            slider_axes[i] = sax
            s = Slider(sax, "", 0, n - 1, valinit=idx0, valstep=1)
            sliders[i] = s
            s.valtext.set_text(f"{idx0}/{n-1}")

            # hide red vline (optional)
            if hasattr(s, "vline") and s.vline is not None:
                s.vline.set_visible(False)

            def make_update(panel_index):
                def _update(_val):
                    ax_sel_local = axis_per_panel[panel_index]
                    n_ref = vols[panel_index].shape[ax_sel_local]
                    k = int(np.clip(int(sliders[panel_index].val), 0, n_ref - 1))
                    update_panel(panel_index, k)
                return _update

            s.on_changed(make_update(i))

            axes_to_panel[ax] = i
            axes_to_panel[cax] = i
            axes_to_panel[sax] = i
            image_axes_set.add(ax)

        # ---------------- optional left UI (axis) ----------------
        if enable_axis_change:
            if per_panel_axis:
                panel_box_ax = fig.add_axes([0, 0, 0.01, 0.01])
                panel_box = TextBox(panel_box_ax, "", initial="1")
                panel_label = fig.text(0, 0, "panel", transform=fig.transFigure,
                                       ha="left", va="bottom", fontsize=9)

                def on_panel_submit(text):
                    try:
                        i_user = int(text.strip())
                        i0 = i_user - 1
                    except Exception:
                        panel_box.set_val(str(selected_panel[0] + 1))
                        return
                    set_selected_panel(i0)

                panel_box.on_submit(on_panel_submit)

                slice_box_ax = fig.add_axes([0, 0, 0.01, 0.01])
                slice_box = TextBox(slice_box_ax, "", initial=str(int(sliders[selected_panel[0]].val)))
                slice_label = fig.text(0, 0, "slice", transform=fig.transFigure,
                                       ha="left", va="bottom", fontsize=9)

                def apply_left_slice(text):
                    if slice_box_guard[0]:
                        return
                    i0 = selected_panel[0]
                    try:
                        k = int(str(text).strip())
                    except Exception:
                        update_selected_slice_box()
                        return

                    ax_sel_local = axis_per_panel[i0]
                    n_ref = vols[i0].shape[ax_sel_local]
                    k = int(np.clip(k, 0, n_ref - 1))

                    sl = sliders[i0]
                    sl.eventson = False
                    sl.set_val(k)
                    sl.eventson = True

                    update_panel(i0, k)

                    slice_box_guard[0] = True
                    try:
                        slice_box.set_val(str(k))
                    finally:
                        slice_box_guard[0] = False

                slice_box.on_submit(apply_left_slice)

            radio_ax = fig.add_axes([0, 0, 0.01, 0.01])
            radio = RadioButtons(
                radio_ax,
                AXIS_LABELS,
                active=axis_per_panel[selected_panel[0]] if per_panel_axis else int(cfg["axis"]),
            )

            def on_axis_change(label):
                #new_axis = LABEL_TO_AXIS.get(label, None)
                #if new_axis is None:
                #    new_axis = 0 if "Z" in label else (1 if "Y" in label else 2)
                new_axis = LABEL_TO_AXIS.get(label, 0)

                if per_panel_axis:
                    i0 = selected_panel[0]
                    axis_per_panel[i0] = new_axis

                    new_n = vols[i0].shape[new_axis]
                    old_n = max(1, int(sliders[i0].valmax) + 1)
                    frac = float(sliders[i0].val) / max(old_n - 1, 1)
                    new_k = int(round(frac * (new_n - 1))) if new_n > 1 else 0

                    new_k = set_slider_range(sliders[i0], new_n, new_k)
                    update_panel(i0, new_k)
                    if slice_box is not None:
                        update_selected_slice_box()
                else:
                    for j in range(len(vols)):
                        axis_per_panel[j] = new_axis
                        new_n = vols[j].shape[new_axis]
                        old_n = max(1, int(sliders[j].valmax) + 1)
                        frac = float(sliders[j].val) / max(old_n - 1, 1)
                        new_k = int(round(frac * (new_n - 1))) if new_n > 1 else 0
                        new_k = set_slider_range(sliders[j], new_n, new_k)
                        update_panel(j, new_k)

            radio.on_clicked(on_axis_change)

        # ---------------- marker controls (left panel) ----------------
        if enable_markers:
            mark_toggle_ax = fig.add_axes([0, 0, 0.01, 0.01])
            mark_toggle_btn = Button(mark_toggle_ax, "Marking: OFF")

            undo_ax = fig.add_axes([0, 0, 0.01, 0.01])
            undo_btn = Button(undo_ax, "Undo")

            clear_ax = fig.add_axes([0, 0, 0.01, 0.01])
            clear_btn = Button(clear_ax, "Clear")

            def _set_marking_label():
                mark_toggle_btn.label.set_text("Mark: ON" if marking_enabled[0] else "Mark: OFF")
                fig.canvas.draw_idle()

            def _toggle_marking(_evt):
                marking_enabled[0] = not marking_enabled[0]
                _set_marking_label()

            def _undo_last(_evt):
                if not marker_stack:
                    return
                m = marker_stack[-1]
                _remove_marker(m, adjust_counter_if_last=True)
                fig.canvas.draw_idle()

            def _clear_all(_evt):
                if not marker_stack:
                    return

                while marker_stack:
                    m = marker_stack.pop()
                    try: m["line"].remove()
                    except Exception: pass
                    try: m["ann"].remove()
                    except Exception: pass

                for ln in list(slider_tick_lines.values()):
                    try:
                        ln.remove()
                    except Exception:
                        pass
                slider_tick_lines.clear()
                marker_counts.clear()

                marker_counter[0] = 0
                fig.canvas.draw_idle()

            mark_toggle_btn.on_clicked(_toggle_marking)
            undo_btn.on_clicked(_undo_last)
            clear_btn.on_clicked(_clear_all)

            _set_marking_label()

        # ---------------- fixed pixel layout ----------------
        _need_relayout = [True]
        _draw_guard = [False]

        def layout_all():
            w_px, h_px = canvas_wh_px()

            if ((enable_axis_change and per_panel_axis and (panel_box_ax is not None)) or enable_markers):
                if left_panel_anchor_px[0] is None:
                    left_panel_anchor_px[0] = int(cfg["panel_box_rect"][0] * w_px)
                    left_panel_anchor_px[1] = int(cfg["panel_box_rect"][1] * h_px)

            area_x0 = int(cfg["left"] * w_px)
            area_x1 = int(cfg["right"] * w_px)
            area_y0 = int(cfg["bottom"] * h_px)
            area_y1 = int(cfg["top"] * h_px)

            if left_panel_anchor_px[0] is not None and left_panel_px > 0:
                lp_x0 = left_panel_anchor_px[0]
                area_x0 = max(area_x0, lp_x0 + BOX_W_PX + cfg["left_panel_gap_px"])

            avail_w = max(area_x1 - area_x0, 10)
            avail_h = max(area_y1 - area_y0, 10)

            internal_h = cfg["cbar_h_px"] + cfg["cbar_gap_px"] + cfg["slider_h_px"] + cfg["slider_pad_px"]

            total_v_gaps = (rows - 1) * cfg["unit_hgap_px"]
            row_h = (avail_h - total_v_gaps) / max(rows, 1)

            img_side_v = row_h - internal_h
            total_h_gaps = (cols - 1) * cfg["unit_wgap_px"]
            img_side_h = (avail_w - total_h_gaps) / max(cols, 1)
            img_side = int(max(5, min(img_side_v, img_side_h)))

            unit_w = img_side
            unit_h = img_side + internal_h

            grid_w = cols * unit_w + (cols - 1) * cfg["unit_wgap_px"]
            grid_h = rows * unit_h + (rows - 1) * cfg["unit_hgap_px"]

            start_x = area_x0 if cfg["grid_align_x"] != "center" else area_x0 + max(0, (avail_w - grid_w) / 2)
            start_y = area_y0 if cfg["grid_align_y"] != "center" else area_y0 + max(0, (avail_h - grid_h) / 2)

            for r in range(rows):
                for c in range(cols):
                    i = r * cols + c
                    if i >= len(vols):
                        continue

                    ux0 = start_x + c * (unit_w + cfg["unit_wgap_px"])
                    uy0 = start_y + (rows - 1 - r) * (unit_h + cfg["unit_hgap_px"])

                    sx0, sy0, sw, sh = ux0, uy0, unit_w * cfg["slider_len"], cfg["slider_h_px"]
                    cx0, cy0, cw, ch = ux0, sy0 + cfg["slider_h_px"] + cfg["slider_pad_px"], unit_w, cfg["cbar_h_px"]
                    ix0, iy0, iw, ih = ux0, cy0 + cfg["cbar_h_px"] + cfg["cbar_gap_px"], unit_w, img_side

                    axs.ravel()[i].set_position([ix0 / w_px, iy0 / h_px, iw / w_px, ih / h_px])
                    cbar_axes[i].set_position([cx0 / w_px, cy0 / h_px, cw / w_px, ch / h_px])
                    slider_axes[i].set_position([sx0 / w_px, sy0 / h_px, sw / w_px, sh / h_px])

            if enable_axis_change and per_panel_axis and (panel_box_ax is not None):
                lp_x0 = left_panel_anchor_px[0]
                lp_y0 = left_panel_anchor_px[1]

                panel_box_ax.set_position([lp_x0 / w_px, lp_y0 / h_px, BOX_W_PX / w_px, BOX_H_PX / h_px])
                panel_label.set_position((lp_x0 / w_px, (lp_y0 + BOX_H_PX + LABEL_PAD_PX) / h_px))

                slice_y0 = max(lp_y0 - BOX_H_PX - BOX_GAP_PX, 1)
                slice_box_ax.set_position([lp_x0 / w_px, slice_y0 / h_px, BOX_W_PX / w_px, BOX_H_PX / h_px])
                slice_label.set_position((lp_x0 / w_px, (slice_y0 + BOX_H_PX + LABEL_PAD_PX) / h_px))

                radio_y0 = max(slice_y0 - RADIO_H_PX - RADIO_GAP_PX, 1)
                radio_ax.set_position([lp_x0 / w_px, radio_y0 / h_px, RADIO_W_PX / w_px, RADIO_H_PX / h_px])

                if enable_markers:
                    btn1_y0 = max(radio_y0 - BTN_H_PX - BTN_GAP_PX, 1)
                    btn2_y0 = max(btn1_y0 - BTN_H_PX - BTN_GAP_PX, 1)
                    btn3_y0 = max(btn2_y0 - BTN_H_PX - BTN_GAP_PX, 1)

                    mark_toggle_ax.set_position([lp_x0 / w_px, btn1_y0 / h_px, BTN_W_PX / w_px, BTN_H_PX / h_px])
                    undo_ax.set_position([lp_x0 / w_px, btn2_y0 / h_px, BTN_W_PX / w_px, BTN_H_PX / h_px])
                    clear_ax.set_position([lp_x0 / w_px, btn3_y0 / h_px, BTN_W_PX / w_px, BTN_H_PX / h_px])

            elif enable_markers and left_panel_anchor_px[0] is not None:
                lp_x0 = left_panel_anchor_px[0]
                lp_y0 = left_panel_anchor_px[1]

                btn1_y0 = lp_y0
                btn2_y0 = max(btn1_y0 - BTN_H_PX - BTN_GAP_PX, 1)
                btn3_y0 = max(btn2_y0 - BTN_H_PX - BTN_GAP_PX, 1)

                mark_toggle_ax.set_position([lp_x0 / w_px, btn1_y0 / h_px, BTN_W_PX / w_px, BTN_H_PX / h_px])
                undo_ax.set_position([lp_x0 / w_px, btn2_y0 / h_px, BTN_W_PX / w_px, BTN_H_PX / h_px])
                clear_ax.set_position([lp_x0 / w_px, btn3_y0 / h_px, BTN_W_PX / w_px, BTN_H_PX / h_px])

        def _on_resize(_evt):
            _need_relayout[0] = True
            fig.canvas.draw_idle()

        def _on_draw(_evt):
            if _draw_guard[0]:
                _draw_guard[0] = False
                return
            if _need_relayout[0]:
                _need_relayout[0] = False
                layout_all()
                _draw_guard[0] = True
                fig.canvas.draw_idle()

        # ---------------- click selection + marking ----------------
        click_cid = None
        if cfg["enable_click_select"] or enable_markers:
            def on_click(event):
                if event.inaxes is None:
                    return
                if event.inaxes in axes_to_panel:
                    panel_i = axes_to_panel[event.inaxes]
                    set_selected_panel(panel_i)

                    if enable_markers and marking_enabled[0] and (event.inaxes in image_axes_set) and (event.button == 1):
                        if event.xdata is None or event.ydata is None:
                            return

                        ax_sel = int(axis_per_panel[panel_i])
                        k = int(sliders[panel_i].val)

                        if panel_i not in slice_off or panel_i not in slice_data or panel_i not in slice_hw:
                            slc_tmp = get_slice(vols[panel_i], ax_sel, k)
                            _update_slice_cache(panel_i, slc_tmp)

                        h, w = slice_hw[panel_i]
                        x0, y0 = slice_off[panel_i]
                        slc = slice_data[panel_i]

                        x = int(np.floor((event.xdata - x0) + 0.5))
                        y = int(np.floor((event.ydata - y0) + 0.5))
                        if not (0 <= x < w and 0 <= y < h):
                            return

                        hit = _find_marker_hit(panel_i, event, x0, y0)
                        if hit is not None:
                            _remove_marker(hit, adjust_counter_if_last=True)
                            fig.canvas.draw_idle()
                            return

                        val = float(slc[y, x])
                        v_txt = VALUE_FMT.format(val)

                        if ax_sel == 0:
                            z, yv, xv = k, y, x
                        elif ax_sel == 1:
                            z, yv, xv = y, k, x
                        else:
                            z, yv, xv = y, x, k

                        letter = _index_to_label(marker_counter[0])
                        label = f"{letter}: [{z},{yv},{xv}]={v_txt}"
                        marker_counter[0] += 1

                        xf = x0 + x
                        yf = y0 + y

                        ax = axs.ravel()[panel_i]
                        line, = ax.plot(
                            [xf], [yf],
                            marker="x", linestyle="None",
                            markersize=6, markeredgewidth=1.4,
                            color="yellow", zorder=5
                        )
                        ann = ax.annotate(
                            label,
                            (xf, yf),
                            xytext=(6, 6),
                            textcoords="offset points",
                            fontsize=8,
                            color="yellow",
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, lw=0),
                            zorder=6,
                        )

                        marker_stack.append(
                            {"panel": panel_i, "axis": ax_sel, "k": k, "x": x, "y": y, "line": line, "ann": ann}
                        )

                        _inc_marker_count(panel_i, ax_sel, k)
                        fig.canvas.draw_idle()

            click_cid = fig.canvas.mpl_connect("button_press_event", on_click)

        # ---------------- keyboard navigation ----------------
        key_cid = None
        if cfg["enable_keyboard_nav"]:
            def on_key(event):
                if event.key is None:
                    return
                parts = str(event.key).split("+")
                base = parts[-1]
                mods = set(parts[:-1])

                if base not in ("left", "right"):
                    return

                i = selected_panel[0]
                if i not in sliders:
                    return

                step = 1
                if "shift" in mods:
                    step = 10
                if "ctrl" in mods or "control" in mods:
                    step = 5
                if base == "left":
                    step = -step

                ax_sel_local = axis_per_panel[i]
                n_ref = vols[i].shape[ax_sel_local]
                k0 = int(sliders[i].val)
                k = int(np.clip(k0 + step, 0, n_ref - 1))
                if k == k0:
                    return

                sl = sliders[i]
                sl.eventson = False
                sl.set_val(k)
                sl.eventson = True
                update_panel(i, k)

                if per_panel_axis and (slice_box is not None):
                    slice_box_guard[0] = True
                    try:
                        slice_box.set_val(str(k))
                    finally:
                        slice_box_guard[0] = False

            key_cid = fig.canvas.mpl_connect("key_press_event", on_key)

        # keep refs alive (ipympl)
        fig._volume_grid_layout_all = layout_all
        fig._volume_grid_axes_to_panel = axes_to_panel
        fig._volume_grid_click_cid = click_cid
        fig._volume_grid_key_cid = key_cid
        fig._volume_grid_radio = radio
        fig._volume_grid_panel_box = panel_box
        fig._volume_grid_slice_box = slice_box
        fig._volume_grid_marker_buttons = (mark_toggle_btn, undo_btn, clear_btn)

        fig.canvas.draw()
        layout_all()
        fig.canvas.draw_idle()
        fig.canvas.mpl_connect("resize_event", _on_resize)
        fig.canvas.mpl_connect("draw_event", _on_draw)

        plt.show()

        state = {
            "layout_all": layout_all,
            "selected_panel": selected_panel,
            "axis_per_panel": axis_per_panel,
            "sliders": sliders,
            "images": images,
            "markers": marker_stack,
            "marking_enabled": marking_enabled,
            "click_cid": click_cid,
            "key_cid": key_cid,
            "marker_counts": marker_counts,
            "slider_tick_lines": slider_tick_lines,
        }
        return fig, axs, state