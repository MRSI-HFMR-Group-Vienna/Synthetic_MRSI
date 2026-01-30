from abc import ABC, abstractmethod

class WorkingVolume(ABC):
    """
    Use in module: file

    For other classes to be inherited. The need of implementing the working equivalent to the loader class.
    For example: file.ParameterMaps is only for loading the volume, then another class in another module
    implements the other functionalities to 'work' with this volume.
    """

    @abstractmethod
    def to_working_volume(self):
        """Defines an abstract (interface) method. Subclasses must implement it to transform their data into
        a target class in another module."""
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