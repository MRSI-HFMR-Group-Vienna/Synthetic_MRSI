from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any


class ResourceInterface(ABC):
    """
    Use in resources.py

    Interface for defining which methods a class must implement that deals with
    different files that holds paths, metabolite information, bibliograpy details.

    (!) This interface is not for resources such as loading arrays!
    """

    @abstractmethod
    def load(self, *args, **kwargs) -> None:
        """Load data from the underlying file into memory."""
        ...

    @abstractmethod
    def save(self, *args, **kwargs) -> None:
        """Persist the current in-memory data back to the file."""
        ...

    @abstractmethod
    def print_formatted(self, *args, **kwargs) -> None:
        """Print a human-readable representation of the data."""
        ...

    @abstractmethod
    def get_data(self, *args, **kwargs) -> Any:
        """Return the in-memory data structure."""
        ...



WorkingObject = TypeVar("WorkingObject")
class WorkingSourceInterface(ABC, Generic[WorkingObject]):
    """
    Use in module: file

    Interface for loader classes that only read and display data. The actual
    work, such as interpolation, addition, manipulation, etc., is implemented
    in another class in another module. Subclasses must convert themselves into
    that working object via 'to_working'.

    For example: file.FID loads FID data and can plot it for inspection.
    fid.to_working() returns a spectral_spatial_simulation.FID, which provides
    interpolation, addition, manipulation, etc.
    """

    @abstractmethod
    def to_working(self) -> WorkingObject:
        """
        Convert this loader instance into the corresponding working object
        from another module, which provides the actual processing functionality.
        """
        ...


class InterpolationInterface(ABC):
    """
    Use not in: file
    Instead use in e.g., spatial_metabolic_distribution. When e.g., file.ParameterMaps -> spatial_metabolic_distribution.ParameterVolume
    then use in the second one (working volume).
    """
    @abstractmethod
    def interpolate(self, target_size: tuple, order: int, device: str, target_gpu: int = 0, verbose: bool=False):
        """Defines an abstract method. Subclasses must implement a method that interpolates to target size"""
        ...

    def interpolate_volume(self, *args, **kwargs):
        return self.interpolate(*args, **kwargs)

class PlotInterface(ABC):
    """
    Use in: file

    For other classes to be inherited. Visualisation of the loaded volume.
    """

    @abstractmethod
    def plot_jupyter(self, cmap: str):
        """Subclasses must implement this."""
        ...

    @abstractmethod
    def plot(self, cmap: str):
        """Subclasses must implement this."""
        ...