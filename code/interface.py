from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any, Dict
from typing_extensions import Self
import copy



class ResourceInterface(ABC):
    """
    Use in inputs.py

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


class BackupInterface:
    """
    Use in: any module

    Mixin class (see python documentation) for classes that should be able to
    snapshot themselves and fetch those snapshots later.
    This interface is NOT abstract: the backup logic is exactly the same for
    every class, so it is implemented here once and simply inherited, instead
    of forcing each subclass to rewrite identical code.

    Each concrete class gets its OWN backup registry (see __init_subclass__),
    so e.g. FID backups and ParameterMaps backups never land in the same dict.

    (!) Backups are keyed by name and shared across all instances of a class.
        Reusing a name overwrites the older snapshot.
    (!) deepcopy of large arrays / volumes is memory-heavy, that is inherent
        to snapshotting, not a bug.

    For example:
        volume = ParameterVolume(...)
        volume.create_backup("before_interpolation")
        volume.interpolate(...)
        original = ParameterVolume.get_backup("before_interpolation")
    """

    # One dict per subclass, filled in below. Declared here only for typing;
    # (!) Note: BackupInterface itself intentionally holds no shared dict!
    _backups: Dict[str, "BackupInterface"]

    # If a class inherits from this Interface then it gets its own empty "_backups" dictionary
    def __init_subclass__(cls, **kwargs) -> None:
        """
        Give every subclass its own, independent backup dictionary.
        """
        super().__init_subclass__(**kwargs)  # cooperate with ABC etc.
        cls._backups = {}

    def create_backup(self, name: str) -> None:
        """
        Store a deep copy of the current instance under 'name'.

        A deep copy will be cerated so that later changes to this instance do NOT
        change the stored snapshot, they stay fully independent.

        The backup dict lives on the class which implements this interface, not on
        the instance, so it is not part of the copy. Therefore, backups never nest
        inside other backups.

        :param name: name of the backup as string
        :return:
        """
        type(self)._backups[name] = copy.deepcopy(self)


    @classmethod
    def get_backup(cls, name: str) -> Self:
        """
        Return the snapshot stored under 'name'.
        (!) Note: Raises KeyError if 'name' was never backed up!
        """
        return cls._backups[name]