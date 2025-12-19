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

from dask.distributed import LocalCluster, Worker


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


class CustomArray(da.Array):
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

    def __mul__(self, other):
        """
        For preserving information from the left multiplicand. Add it to the result.

        :param other: The right multiplicand. It has to be a dask.Array or a CustomArray
        :return: product of array 1 and array 2
        """
        result = super().__mul__(other)
        result = CustomArray(dask_array=result,
                             block_number=self.block_number,
                             block_idx=self.block_idx,
                             main_volume_shape=self.main_volume_shape,
                             total_number_blocks=self.total_number_blocks,
                             main_volume_blocks=self.main_volume_blocks,
                             meta=self.meta)

        return result

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

    def start_cuda(self, device_numbers: list[int], device_memory_limit: str = "30GB", use_rmm_cupy_allocator: bool = False, protocol="ucx"):
        """
        Protocols like tcp possible, or ucx. "ucx" is best for GPU to GPU, GPU to host communication: https://ucx-py.readthedocs.io/en/latest/install.html
        """

        self._config()

        dask.config.set({"distributed.diagnostics.nvml": False})
        #if use_rmm_cupy_allocator:
            #rmm.reinitialize(pool_allocator=True)
            #cp.cuda.set_allocator(rmm_cupy_allocator)

        possible_protocols = ["tcp", "tls", "inproc", "ucx"]
        if protocol not in possible_protocols:
            Console.printf("error", f"Only protocols {possible_protocols} available for Local CUDA cluster, but you set {protocol}. Terminate program!")
            sys.exit()

        # 2) To spawn UCX cluster before any CUDA imports happen. Important!
        self.cluster = LocalCUDACluster(n_workers=len(device_numbers),
                                        device_memory_limit=device_memory_limit,
                                        jit_unspill=True,
                                        CUDA_VISIBLE_DEVICES=device_numbers,
                                        protocol=protocol,
                                        dashboard_address=":55000")
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

        Console.printf("success", f"Started {self.cluster_type} Cluster \n"
                               f" Link to dashboard: {dashboard_url}")



    def start_cuda_test(
        self,
        device_numbers: list[int] = 1,
        threads_per_worker: int = 1,
        memory_limit_per_worker: str = "30GB",
        dashboard_port: int = 55000,
        local_directory: str = "/tmp/dask-worker-space",
        use_rmm_cupy_allocator: bool = False,
    ):
        # 0) Force-disable NVML diagnostics for this cluster, regardless of instance setting
        prev_nvml_diagnostics = self.nvml_diagnostics
        self.nvml_diagnostics = False
    
        self._config()
        self.nvml_diagnostics = prev_nvml_diagnostics
    
        # 1) LocalCluster with *no Nanny* (processes=False or worker_class=Worker)
        self.cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=threads_per_worker,
            processes=False,                         # <- critical: no Nanny
            worker_class=Worker,                    # (optional but explicit)
            memory_limit=memory_limit_per_worker,   # or "auto"
            local_directory=local_directory,
            dashboard_address=f":{dashboard_port}",
        )
        self.cluster_type = "cuda"
        self.__start_client()
    
        # 2) Optional: RMM / CuPy allocator init
        if use_rmm_cupy_allocator:
            def _init_rmm():
                import rmm
                import cupy as cp
                rmm.reinitialize(pool_allocator=True, initial_pool_size="10GB")
                cp.cuda.set_allocator(rmm_cupy_allocator)
    
            self.client.run(_init_rmm)


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


# put estimate space here

class SpaceEstimator:
    """
    To calculate the required space on the disk for a numpy array of a given shape and data type. Usefully, if a big array
    should be created and to check easily if it does not exceed a certain required space.
    """

    @staticmethod
    def for_numpy(data_shape: tuple, data_type: np.dtype, unit: str = "MB") -> np.ndarray:
        """
        For estimating the required space of a numpy array with a desired shape and data type.

        :param data_shape: desired shape of the numpy array
        :param data_type: desired data type of the numpy array (e.g., np.int64)
        :param unit: desired unit the numpy array
        :return: required space on disk as a numpy array (with defined unit: [bytes], [KB], [MB], [GB]). Standard unit is [MB].
        """

        # np.prod...  To get total numpy array size
        # itemsize... bytes required by each element in array
        space_required_bytes = np.prod(data_shape) * data_type.itemsize

        if unit == "byte":
            return space_required_bytes

        elif unit == "KB":
            return space_required_bytes * 1 / 1024

        elif unit == "MB":
            return space_required_bytes * 1 / (1024 ** 2)

        elif unit == "GB":
            return space_required_bytes * 1 / (1024 ** 3)

###    @staticmethod
###    def for_torch(torch_array: torch.Tensor, unit: str = "MB"):
###        """
###        TODO: Implement it.
###        """
###        raise NotImplementedError("This method is not yet implemented")


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

