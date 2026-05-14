from abc import ABC, abstractmethod
from typing import Generic, TypeVar


WorkingObject = TypeVar("WorkingObject")

class WorkingSource(ABC, Generic[WorkingObject]):
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


class Interpolation(ABC):
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

class Plot(ABC):
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