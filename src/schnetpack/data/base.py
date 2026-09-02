from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Iterable, Union
import torch
import copy
from ase import Atoms
from schnetpack.transform import Transform

class AtomsDataError(Exception):
    pass

class BaseAtomsData(ABC):
    """
    Base mixin class for atomistic data. Use together with PyTorch Dataset or
    IterableDataset to implement concrete data formats.
    """

    def __init__(
        self,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        transforms: Optional[List[Transform]] = None,
        subset_idx: Optional[List[int]] = None,
    ):
        """
        Args:
            load_properties: Set of properties to be loaded and returned.
                If None, all properties in the ASE dB will be returned.
            load_structure: If True, load structure properties.
            transforms: preprocessing transforms (see schnetpack.data.transforms)
            subset: List of data indices.
        """
        self._transform_module = None
        self.load_properties = load_properties
        self.load_structure = load_structure
        self.transforms = transforms
        self.subset_idx = subset_idx

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, value: Optional[List[Transform]]):
        self._transforms = []
        self._transform_module = None

        if value is not None:
            for tf in value:
                self._transforms.append(tf)
            self._transform_module = torch.nn.Sequential(*self._transforms)

    def subset(self, subset_idx: List[int]):
        assert (
            subset_idx is not None
        ), "Indices for creation of the subset need to be provided!"
        ds = copy.copy(self)
        if ds.subset_idx:
            ds.subset_idx = [ds.subset_idx[i] for i in subset_idx]
        else:
            ds.subset_idx = subset_idx
        return ds

    @property
    @abstractmethod
    def available_properties(self) -> List[str]:
        """Available properties in the dataset"""
        pass

    @property
    @abstractmethod
    def units(self) -> Dict[str, str]:
        """Property to unit dict"""
        pass

    @property
    def load_properties(self) -> List[str]:
        """Properties to be loaded"""
        if self._load_properties is None:
            return self.available_properties
        else:
            return self._load_properties

    @load_properties.setter
    def load_properties(self, val: List[str]):
        if val is not None:
            props = self.available_properties
            assert all(
                [p in props for p in val]
            ), "Not all given properties are available in the dataset!"
        self._load_properties = val

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Global metadata"""
        pass

    @property
    @abstractmethod
    def atomrefs(self) -> Dict[str, torch.Tensor]:
        """Single-atom reference values for properties"""
        pass

    @abstractmethod
    def update_metadata(self, **kwargs):
        pass

    @abstractmethod
    def iter_properties(
        self,
        indices: Union[int, Iterable[int]] = None,
        load_properties: List[str] = None,
        load_structure: Optional[bool] = None,
    ):
        pass

    @staticmethod
    @abstractmethod
    def create(
        datapath: str,
        position_unit: str,
        property_unit_dict: Dict[str, str],
        atomrefs: Dict[str, List[float]],
        **kwargs,
    ) -> "BaseAtomsData":
        pass

    @abstractmethod
    def add_systems(
        self,
        property_list: List[Dict[str, Any]],
        atoms_list: Optional[List[Atoms]] = None,
    ):
        pass

    @abstractmethod
    def add_system(self, atoms: Optional[Atoms] = None, **properties):
        pass
