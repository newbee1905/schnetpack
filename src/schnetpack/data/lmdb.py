import logging
import os
import pickle
from typing import Optional, List, Dict, Any, Iterable, Union

import lmdb
import torch
from ase import Atoms

import schnetpack as spk
import schnetpack.properties as structure
from schnetpack.data.base import BaseAtomsData, AtomsDataError


class LMDBAtomsData(BaseAtomsData):
    """
    PyTorch dataset for atomistic data. The raw data is stored in a LMDB database.
    """

    def __init__(
        self,
        datapath: str,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        transforms: Optional[List[torch.nn.Module]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
    ):
        self._env = None
        self._env_pid = None

        self.datapath = datapath
        super().__init__(
            load_properties=load_properties,
            load_structure=load_structure,
            transforms=transforms,
            subset_idx=subset_idx,
        )

        self._check_db()

        # initialize units
        md = self.metadata
        if "_distance_unit" not in md.keys():
            raise AtomsDataError(
                "Dataset does not have a distance unit set. Please add units to the "
                + "dataset using `spkconvert`!"
            )
        if "_property_unit_dict" not in md.keys():
            raise AtomsDataError(
                "Dataset does not have a property units set. Please add units to the "
                + "dataset using `spkconvert`!"
            )

        if distance_unit:
            self.distance_conversion = spk.units.convert_units(
                md["_distance_unit"], distance_unit
            )
            self.distance_unit = distance_unit
        else:
            self.distance_conversion = 1.0
            self.distance_unit = md["_distance_unit"]

        self._units = md["_property_unit_dict"]
        self.conversions = {prop: 1.0 for prop in self._units}
        if property_units is not None:
            for prop, unit in property_units.items():
                self.conversions[prop] = spk.units.convert_units(
                    self._units[prop], unit
                )
                self._units[prop] = unit

    def __getstate__(self):
        # When pickling, don't include the LMDB environment object.
        # It's not fork-safe.
        state = self.__dict__.copy()
        state['_env'] = None
        state['_env_pid'] = None

        return state

    def __setstate__(self, state):
        # When unpickling in the new process, restore the dict.
        # The '_env' will be None, and the 'env' property will
        # correctly re-initialize it upon first access.
        self.__dict__.update(state)


    @property
    def env(self):
        if self._env is not None and self._env_pid != os.getpid():
            self._env = None

        if self._env is None:
            self._env = lmdb.open(
                self.datapath,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=128,
                map_size=10995111627776 * 2,
            )
            self._env_pid = os.getpid()
        return self._env

    def __len__(self) -> int:
        if self.subset_idx is not None:
            return len(self.subset_idx)

        with self.env.begin() as txn:
            return pickle.loads(txn.get(b"__len__"))

    def _check_db(self):
        if not os.path.exists(self.datapath):
            raise AtomsDataError(f"LMDB DB does not exists at {self.datapath}")

    @property
    def metadata(self) -> Dict[str, Any]:
        with self.env.begin() as txn:
            return pickle.loads(txn.get(b"_metadata"))

    def _set_metadata(self, val: Dict[str, Any]):
        with self.env.begin(write=True) as txn:
            txn.put(b"_metadata", pickle.dumps(val))

    def update_metadata(self, **kwargs):
        assert all(
            key[0] != 0 for key in kwargs
        ), "Metadata keys starting with '_' are protected!"

        md = self.metadata
        md.update(kwargs)
        self._set_metadata(md)

    @property
    def available_properties(self) -> List[str]:
        md = self.metadata
        return list(md["_property_unit_dict"].keys())

    @property
    def units(self) -> Dict[str, str]:
        """Dictionary of properties to units"""
        return self._units

    @property
    def atomrefs(self) -> Dict[str, torch.Tensor]:
        md = self.metadata
        arefs = md["atomrefs"]
        arefs = {k: self.conversions[k] * torch.tensor(v) for k, v in arefs.items()}
        return arefs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.subset_idx is not None:
            idx = self.subset_idx[idx]

        props = self._get_properties(
            idx, self.load_properties, self.load_structure
        )
        props = self._apply_transforms(props)

        return props

    def _apply_transforms(self, props):
        if self._transform_module is not None:
            props = self._transform_module(props)
        return props

    def _get_properties(
        self, idx: int, load_properties: List[str], load_structure: bool
    ):
        key = str(idx).encode("ascii")
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(key))

        properties = {}
        properties[structure.idx] = torch.tensor([idx])
        for pname in load_properties:
            np_data = data[pname]
            properties[pname] = (
                torch.tensor(np_data) * self.conversions[pname]
            )

        if load_structure:
            Z = data[structure.Z]
            properties[structure.n_atoms] = torch.tensor([Z.shape[0]])
            properties[structure.Z] = torch.tensor(Z, dtype=torch.long)
            
            R = data[structure.R]
            properties[structure.position] = (
                torch.tensor(R) * self.distance_conversion
            )
            
            cell = data[structure.cell]
            properties[structure.cell] = (
                torch.tensor(cell[None]) * self.distance_conversion
            )
            
            pbc = data[structure.pbc]
            properties[structure.pbc] = torch.tensor(pbc)
        
        return properties

    def iter_properties(
        self,
        indices: Union[int, Iterable[int]] = None,
        load_properties: List[str] = None,
        load_structure: Optional[bool] = None,
    ):
        if load_properties is None:
            load_properties = self.load_properties
        load_structure = load_structure or self.load_structure

        if self.subset_idx:
            if indices is None:
                indices = self.subset_idx
            elif isinstance(indices, int):
                indices = [self.subset_idx[indices]]
            else:
                indices = [self.subset_idx[i] for i in indices]
        else:
            if indices is None:
                indices = range(len(self))
            elif isinstance(indices, int):
                indices = [indices]

        for i in indices:
            yield self._get_properties(
                i,
                load_properties=load_properties,
                load_structure=load_structure,
            )


    @staticmethod
    def create(
        datapath: str,
        distance_unit: str,
        property_unit_dict: Dict[str, str],
        atomrefs: Optional[Dict[str, List[float]]] = None,
        **kwargs,
    ) -> "BaseAtomsData":
        if not datapath.endswith(".lmdb"):
            raise AtomsDataError(
                "Invalid datapath! Please make sure to add the file extension '.lmdb' to "
                "your dbpath."
            )

        if os.path.exists(datapath):
            raise AtomsDataError(f"Dataset already exists: {datapath}")

        env = lmdb.open(
            datapath,
            subdir=False,
            map_size=1099511627776 * 2,  # 2TB
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
        )

        atomrefs = atomrefs or {}
        metadata = {
            "_property_unit_dict": property_unit_dict,
            "_distance_unit": distance_unit,
            "atomrefs": atomrefs,
        }

        with env.begin(write=True) as txn:
            txn.put(b"_metadata", pickle.dumps(metadata))
            txn.put(b"__len__", pickle.dumps(0))
        
        env.close()

        return LMDBAtomsData(datapath, **kwargs)

    def add_system(self, atoms: Optional[Atoms] = None, **properties):
        # This method is not used directly, add_systems is used.
        raise NotImplementedError

    def add_systems(
        self,
        property_list: List[Dict[str, Any]],
        atoms_list: Optional[List[Atoms]] = None,
    ):
        if atoms_list is None:
            atoms_list = [None] * len(property_list)

        env = lmdb.open(
            self.datapath,
            subdir=False,
            map_size=10995111627776 * 2,  # 2TB
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with env.begin(write=True) as txn:
            for at, prop in zip(atoms_list, property_list):
                self._add_system(txn, at, **prop)
        env.close()


    def _add_system(self, txn, atoms: Optional[Atoms] = None, **properties):
        """Add systems to DB"""
        if atoms is None:
            try:
                Z = properties[structure.Z]
                R = properties[structure.R]
                cell = properties[structure.cell]
                pbc = properties[structure.pbc]
                atoms = Atoms(numbers=Z, positions=R, cell=cell, pbc=pbc)
            except KeyError as e:
                raise AtomsDataError(
                    "Property dict does not contain all necessary structure keys"
                ) from e

        # add available properties to database
        metadata = pickle.loads(txn.get(b"_metadata"))
        valid_props = set().union(
            metadata["_property_unit_dict"].keys(),
            [
                structure.Z,
                structure.R,
                structure.cell,
                structure.pbc,
            ],
        )
        for prop in properties:
            if prop not in valid_props:
                logging.warning(
                    f"Property `{prop}` is not a defined property for this dataset and "
                    + f"will be ignored. If it should be included, it has to be "
                    + f"provided together with its unit when calling "
                    + f"LMDBAtomsData.create()."
                )

        data = {}
        for pname in metadata["_property_unit_dict"].keys():
            try:
                data[pname] = properties[pname]
            except:
                raise AtomsDataError("Required property missing:" + pname)
        
        # Add structure properties
        data[structure.Z] = atoms.numbers
        data[structure.R] = atoms.positions
        data[structure.cell] = atoms.cell
        data[structure.pbc] = atoms.pbc

        idx = pickle.loads(txn.get(b"__len__"))
        txn.put(str(idx).encode("ascii"), pickle.dumps(data))
        txn.put(b"__len__", pickle.dumps(idx + 1))

# vi: set ts=4 sw=4 expandtab:
