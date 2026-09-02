import logging
import os
import shutil
import tempfile
from typing import List, Optional, Dict, Union, Any, Iterable, Tuple
import requests
from copy import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import h5py
import fasteners
from tqdm import tqdm

import torch
from schnetpack.nn import scatter_add
from schnetpack.data import *
import schnetpack.properties as properties
import schnetpack.properties as structure
from schnetpack.data import AtomsDataModuleError, AtomsDataModule
from schnetpack.task import ModelOutput
from torchmetrics import Metric

from openbabel import openbabel
from io import StringIO

__all__ = ["RGD1", "RGD1Single"]


def get_qm9_atomrefs_g():
    """
    Returns QM9 atomrefs for Gibbs Free Energy (G) in Hartree.
    Values are taken from QM9 atomrefs.txt (index 5).
    """
    # H (1), C (6), N (7), O (8), F (9)
    atref_g = torch.zeros((100,))
    atref_g[1] = -0.500273
    atref_g[6] = -37.846772
    atref_g[7] = -54.583861
    atref_g[8] = -75.064579
    atref_g[9] = -99.718730
    return atref_g

def _rgd1_smiles_worker(smiles):
    """
    Worker function to perform SMILES geometry generation for a single molecule.
    Ensures fragments are separated.
    """
    from openbabel import openbabel
    import numpy as np
    
    try:
        if not smiles:
            return None, None
            
        smiles_fragments = [f for f in smiles.split('.') if f]
        all_coords = []
        all_numbers = []
        
        for i, frag in enumerate(smiles_fragments):
            f_mol = openbabel.OBMol()
            f_conv = openbabel.OBConversion()
            f_conv.SetInFormat("smi")
            f_conv.ReadString(f_mol, frag)
            f_mol.AddHydrogens()
            builder = openbabel.OBBuilder()
            builder.Build(f_mol)
            
            # Basic optimization to get decent internal coordinates
            ff = openbabel.OBForceField.FindForceField("mmff94") or openbabel.OBForceField.FindForceField("uff")
            if ff:
                ff.Setup(f_mol)
                ff.SteepestDescent(500, 1.0e-4)
                ff.ConjugateGradients(1000, 1.0e-6)
                ff.GetCoordinates(f_mol)
            
            frag_coords = []
            frag_numbers = []
            offset = i * 50.0  # Large separation
            for j in range(1, f_mol.NumAtoms() + 1):
                atom = f_mol.GetAtom(j)
                frag_coords.append([atom.GetX() + offset, atom.GetY(), atom.GetZ()])
                frag_numbers.append(atom.GetAtomicNum())
            
            all_coords.append(np.array(frag_coords))
            all_numbers.append(np.array(frag_numbers))
        
        return np.concatenate(all_coords).astype(np.float32), np.concatenate(all_numbers).astype(np.int64)
    except Exception:
        pass
    return None, None


def _separate_fragments(pos, threshold=2.2):
    """
    Simple distance-based fragment separation to simulate isolated species.
    Preserves atom order.
    """
    import numpy as np
    n = len(pos)
    if n < 2:
        return pos
    
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    adj = dist < threshold
    labels = -np.ones(n, dtype=int)
    comp_id = 0
    for i in range(n):
        if labels[i] == -1:
            stack = [i]
            labels[i] = comp_id
            while stack:
                u = stack.pop()
                for v in np.where(adj[u])[0]:
                    if labels[v] == -1:
                        labels[v] = comp_id
                        stack.append(v)
            comp_id += 1
            
    if comp_id <= 1:
        return pos
        
    new_pos = pos.copy()
    for i in range(1, comp_id):
        new_pos[labels == i] += i * 50.0
    return new_pos


class RGD1Single(AtomsDataModule):
    """
    Flattened version of RGD1 where each reaction is split into separate
    samples based on the provided states (e.g. Reactant, Product, Transition State).
    Each sample contains a standard 'energy' property (Gibbs free energy).
    """

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        num_train: Optional[Union[int, float]] = None,
        num_val: Optional[Union[int, float]] = None,
        num_test: Optional[Union[int, float]] = None,
        split_file: Optional[str] = "split_single.npz",
        format: Optional[AtomsDataFormat] = AtomsDataFormat.LMDB,
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 12,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        states: List[str] = ["reactants", "products", "ts"],
        use_smiles: bool = False,
        optimize_geometries: bool = False,
        **kwargs,
    ):
        self.states = states
        self.use_smiles = use_smiles
        self.optimize_geometries = optimize_geometries
        self.num_workers = num_workers

        # Default to a state-specific split file to avoid stale index errors
        if split_file is not None and ("split_single" in split_file or "single_split" in split_file) and len(states) != 3:
            path, ext = os.path.splitext(split_file)
            split_file = f"{path}_{len(states)}{ext}"

        super().__init__(
            datapath=datapath,
            batch_size=batch_size,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            split_file=split_file,
            format=format,
            load_properties=load_properties,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=num_workers,
            num_val_workers=num_val_workers,
            num_test_workers=num_test_workers,
            property_units=property_units,
            distance_unit=distance_unit,
            data_workdir=data_workdir,
            **kwargs,
        )

    def _load_partitions(self):
        lock = fasteners.InterProcessLock("splitting.lock")

        with lock:
            self._log_with_rank("Enter splitting lock")

            if self.split_file is not None and os.path.exists(self.split_file):
                self._log_with_rank("Load split")

                S = np.load(self.split_file)
                self.train_idx = S["train_idx"].tolist()
                self.val_idx = S["val_idx"].tolist()
                self.test_idx = S["test_idx"].tolist()
            else:
                self._log_with_rank("Create balanced reaction-level split")

                if not self.num_train or not self.num_val:
                    raise AtomsDataModuleError(
                        "If no `split_file` is given, the sizes of the training and"
                        + " validation partitions need to be set!"
                    )

                # Split based on reaction indices
                n_rxns = self.dataset.n_rxns
                n_states = self.dataset.n_states
                
                # Use base dataset for splitting to ensure we get reaction-level partitioning
                # We need to adjust the requested counts to be on reaction level
                rxn_train = int(self.num_train // n_states) if self.num_train > 1 else self.num_train
                rxn_val = int(self.num_val // n_states) if self.num_val > 1 else self.num_val
                rxn_test = int(self.num_test // n_states) if (self.num_test and self.num_test > 1) else self.num_test

                train_rxn, val_rxn, test_rxn = self.splitting.split(
                    self.dataset.base_dataset, rxn_train, rxn_val, rxn_test
                )
                
                # Map back to flattened indices
                self.train_idx = []
                for i in train_rxn:
                    self.train_idx.extend([i * n_states + j for j in range(n_states)])
                
                self.val_idx = []
                for i in val_rxn:
                    self.val_idx.extend([i * n_states + j for j in range(n_states)])
                    
                self.test_idx = []
                for i in test_rxn:
                    self.test_idx.extend([i * n_states + j for j in range(n_states)])

                if self.split_file is not None:
                    self._log_with_rank("Save split")
                    np.savez(
                        self.split_file,
                        train_idx=self.train_idx,
                        val_idx=self.val_idx,
                        test_idx=self.test_idx,
                    )

        self._log_with_rank("Exit splitting lock")

    def prepare_data(self):
        rgd1_dm = RGD1(
            datapath=self.datapath,
            batch_size=self.batch_size,
            format=self.format,
            num_workers=self.num_workers,
            property_units=self.property_units,
            distance_unit=self.distance_unit,
        )
        rgd1_dm.prepare_data()

    def get_stats(
        self, property: str, divide_by_atoms: bool, remove_atomref: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if property == structure.energy and self.train_idx is not None:
            # Check if we already have combined stats
            key = ("combined_energy", divide_by_atoms, remove_atomref)
            if key in self._stats:
                return self._stats[key]

            # In RGD1Single, calculate_stats on the balanced training set 
            # naturally gives the unified mean across all groups.
            stats = super().get_stats(property, divide_by_atoms, remove_atomref)
            self._stats[key] = stats

            return stats

        return super().get_stats(property, divide_by_atoms, remove_atomref)

    def setup(self, stage: Optional[str] = None):
        # check whether data needs to be copied
        if self.data_workdir is None:
            datapath = self.datapath
        else:
            datapath = self._copy_to_workdir()

        if self.dataset is not None:
            return

        # Load the base reaction dataset to get available properties if not set
        if self.load_properties is None:
            # The RGD1Single expects 'energy' and 'state_idx' by default if no properties are set.
            self.load_properties = [structure.energy, "state_idx"]

        # Ensure all internal energy keys are converted if energy unit is specified
        if (
            self.property_units is not None
            and structure.energy in self.property_units
        ):
            energy_unit = self.property_units[structure.energy]
            for p in [
                "reactants_free_energy",
                "products_free_energy",
                "ts_free_energy",
            ]:
                if p not in self.property_units:
                    self.property_units[p] = energy_unit

        # Load the base reaction dataset
        # We need to map unified properties back to the base dataset
        base_load_properties = None
        if self.load_properties is not None:
            base_load_properties = []
            for p in self.load_properties:
                if p == structure.energy:
                    base_load_properties.extend([
                        "reactants_free_energy",
                        "products_free_energy",
                        "ts_free_energy"
                    ])
                elif p != "state_idx":
                    base_load_properties.append(p)

            if self.use_smiles:
                base_load_properties.extend(["reactants_smiles_positions", "products_smiles_positions"])
            else:
                base_load_properties.extend(["reactants_positions", "products_positions"])
            base_load_properties.append("ts_positions")

            base_load_properties = list(set(base_load_properties))

        base_property_units = {}
        if self.property_units is not None:
            # Map unified 'energy' unit to all RGD1 energy keys
            if structure.energy in self.property_units:
                energy_target_unit = self.property_units[structure.energy]
                for k in [
                    "reactants_free_energy", "products_free_energy", "ts_free_energy",
                    "reactants_energy", "products_energy", "ts_energy",
                    "reactants_enthalpy", "products_enthalpy", "ts_enthalpy"
                ]:
                    base_property_units[k] = energy_target_unit

            # Copy other requested units
            for k, v in self.property_units.items():
                if k != structure.energy:
                    base_property_units[k] = v

        base_dataset = load_dataset(
            datapath,
            self.format,
            property_units=base_property_units,
            distance_unit=self.distance_unit,
            load_properties=base_load_properties,
        )

        # Create a wrapper that flattens it
        self.dataset = RGD1FlattenedWrapper(
            base_dataset, states=self.states, use_smiles=self.use_smiles
        )
        self.dataset.load_properties = self.load_properties

        # load and generate partitions
        if self.train_idx is None:
            self._load_partitions()

        # partition dataset
        self._train_dataset = self.dataset.subset(self.train_idx)
        self._val_dataset = self.dataset.subset(self.val_idx)
        self._test_dataset = self.dataset.subset(self.test_idx)

        # Ensure unit-converted atomrefs are available in metadata for transforms
        if self.property_units:
            for p, unit in self.property_units.items():
                if p == structure.energy and self.dataset.atomrefs is not None:
                    # The dataset.atomrefs property already handles conversion 
                    # if property_units were passed to load_dataset.
                    pass

        self._setup_transforms()

    def get_group_stats(
        self, property: str, divide_by_atoms: bool
    ) -> torch.Tensor:
        """
        Calculate mean for each state (reactants, ts, products).
        """
        if property == structure.energy:
            key = ("group_energy_stats", divide_by_atoms)
            if key in self._stats:
                return self._stats[key]

            # We need to calculate stats for each state separately
            # Use self.train_dataloader() to iterate over training data
            n_states = len(self.states)
            sums = torch.zeros(n_states)
            counts = torch.zeros(n_states)

            # We want to remove atomrefs before calculating group means
            # to be consistent with RemoveOffsets(remove_atomrefs=True)
            aref = self.train_dataset.atomrefs

            loader = self.train_dataloader()
            for batch in tqdm(loader, desc="Calculating group stats"):
                energy = batch[structure.energy]
                z = batch[structure.Z]
                idx_m = batch[structure.idx_m]
                state_idx = batch["state_idx"]

                # Subtract atomrefs
                y0i = aref[structure.energy][z]
                maxm = int(idx_m[-1]) + 1
                y0 = scatter_add(y0i, idx_m, dim_size=maxm)
                energy = energy - y0

                if divide_by_atoms:
                    energy = energy / batch[structure.n_atoms]

                for s in range(n_states):
                    mask = (state_idx == s)
                    if mask.any():
                        sums[s] += energy[mask].sum()
                        counts[s] += mask.sum()

            means = sums / counts
            self._stats[key] = means
            return means

        return super().get_stats(property, divide_by_atoms, True)[0].repeat(len(self.states))

    @property
    def atomrefs(self) -> Dict[str, torch.Tensor]:
        if self._train_dataset is not None:
            return self._train_dataset.atomrefs
        return super().atomrefs


class RGD1FlattenedWrapper(BaseAtomsData, torch.utils.data.Dataset):
    """
    Wrapper that transforms a reaction dataset into a structure dataset.
    Returns filtered structures for every reaction.
    """
    def __init__(
        self,
        base_dataset,
        states=["reactants", "products", "ts"],
        use_smiles=False,
        subset_idx=None,
    ):
        self._load_properties = None
        super().__init__(subset_idx=subset_idx)
        self.base_dataset = base_dataset
        self.n_rxns = len(base_dataset)
        self.states = states
        self.n_states = len(states)
        self.use_smiles = use_smiles
        
    def __len__(self):
        if self.subset_idx is not None:
            return len(self.subset_idx)
        return self.n_rxns * self.n_states
        
    def __getitem__(self, idx):
        if self.subset_idx is not None:
            idx = self.subset_idx[idx]

        rxn_idx = idx // self.n_states
        state_name = self.states[idx % self.n_states]
        
        data = self.base_dataset[rxn_idx]
        
        # Remap properties based on state
        if state_name == "reactants":
            pos_key = (
                "reactants_smiles_positions" if self.use_smiles else "reactants_positions"
            )
            z_key = "reactants_Z"
            energy_key = "reactants_free_energy"
        elif state_name == "products":
            pos_key = (
                "products_smiles_positions" if self.use_smiles else "products_positions"
            )
            z_key = "products_Z"
            energy_key = "products_free_energy"
        else:  # ts
            pos_key, z_key, energy_key = "ts_positions", "ts_Z", "ts_free_energy"
            
        suffix = f"_{pos_key}" if pos_key != properties.R else ""
        
        # Safe Z key fallback
        actual_z_key = z_key if z_key in data else structure.Z

        new_data = {
            structure.Z: data[actual_z_key].long(),
            structure.R: data[pos_key].float(),
            structure.cell: data[structure.cell].float(),
            structure.pbc: data[structure.pbc].bool(),
            structure.energy: data[energy_key].float().view(1),
            structure.idx: torch.tensor([idx], dtype=torch.long),
            "state_idx": torch.tensor([idx % self.n_states], dtype=torch.long)
        }
        
        # Remap neighbor list keys
        if f"{properties.idx_i}{suffix}" in data:
            new_data[properties.idx_i] = data[f"{properties.idx_i}{suffix}"]
            new_data[properties.idx_j] = data[f"{properties.idx_j}{suffix}"]
            new_data[properties.offsets] = data[f"{properties.offsets}{suffix}"]
        
        # Add any other properties if they exist and are not specific to other states
        for k, v in data.items():
            if k not in [
                structure.Z, structure.R, structure.cell, structure.pbc, structure.idx,
                structure.energy,
                "reactants_positions", "products_positions", "ts_positions",
                "reactants_free_energy", "products_free_energy", "ts_free_energy",
                properties.idx_i, properties.idx_j, properties.offsets,
                f"{properties.idx_i}_reactants_positions", f"{properties.idx_j}_reactants_positions", f"{properties.offsets}_reactants_positions",
                f"{properties.idx_i}_products_positions", f"{properties.idx_j}_products_positions", f"{properties.offsets}_products_positions",
                f"{properties.idx_i}_ts_positions", f"{properties.idx_j}_ts_positions", f"{properties.offsets}_ts_positions"
            ]:
                new_data[k] = v
        
        # Apply transforms
        new_data = self._apply_transforms(new_data)

        return new_data


    def _apply_transforms(self, props):
        if self._transform_module is not None:
            props = self._transform_module(props)
        return props

    @property
    def metadata(self):
        meta = copy(self.base_dataset.metadata)
        if "_property_unit_dict" in meta:
            # Remap units: find any energy-like unit and assign it to 'energy'
            for k in ["ts_free_energy", "reactants_free_energy", "products_free_energy",
                      "ts_energy", "reactants_energy", "products_energy"]:
                if k in meta["_property_unit_dict"]:
                    meta["_property_unit_dict"][structure.energy] = meta["_property_unit_dict"][k]
                    break
        
        # Use converted atomrefs from the base dataset property
        arefs = self.base_dataset.atomrefs
        if arefs:
            if "atomrefs" not in meta:
                meta["atomrefs"] = {}
            
            # Map converted atomrefs to 'energy'
            for k in ["ts_free_energy", "reactants_free_energy", "products_free_energy",
                      "ts_energy", "reactants_energy", "products_energy"]:
                if k in arefs:
                    # Store as list for metadata compatibility
                    meta["atomrefs"][structure.energy] = arefs[k].tolist()
                    break
        return meta

    @property
    def atoms_data_format(self):
        return self.base_dataset.atoms_data_format

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
            ), f"Not all given properties {val} are available in the dataset {props}!"
        self._load_properties = val

    @property
    def available_properties(self):
        props = self.base_dataset.available_properties
        # Filter out state-specific props and add the unified energy
        props = [p for p in props if p not in ["reactants_free_energy", "products_free_energy", "ts_free_energy", 
                                              "reactants_positions", "products_positions", "ts_positions"]]
        props.append(structure.energy)
        props.append("state_idx")
        return props

    @property
    def atomrefs(self) -> Dict[str, torch.Tensor]:
        # Process atomrefs from metadata and apply base dataset conversions
        meta = self.base_dataset.metadata
        raw_arefs = meta.get("atomrefs", {})
        conversions = getattr(self.base_dataset, "conversions", {})

        # Determine the conversion factor for energy if not explicitly set
        # We need this to scale the raw atomref values from metadata
        energy_conv = 1.0
        for k in [
            "ts_free_energy",
            "reactants_free_energy",
            "products_free_energy",
            "ts_energy",
            "reactants_energy",
            "products_energy",
        ]:
            if k in conversions:
                energy_conv = conversions[k]
                break

        arefs = {}
        # Apply conversions to all raw arefs
        for k, v in raw_arefs.items():
            # If property has a conversion in base dataset, use it
            if k in conversions:
                arefs[k] = conversions[k] * torch.tensor(v)
            # If it's an energy-like property but no conversion in base, use energy_conv
            elif any(x in k for x in ["energy", "free_energy", "enthalpy"]):
                arefs[k] = energy_conv * torch.tensor(v)
            else:
                arefs[k] = torch.tensor(v)

        # Explicitly ensure 'energy' key is present and converted
        if structure.energy not in arefs:
            for k in [
                "ts_free_energy",
                "reactants_free_energy",
                "products_free_energy",
                "ts_energy",
                "reactants_energy",
                "products_energy",
            ]:
                if k in raw_arefs:
                    arefs[structure.energy] = energy_conv * torch.tensor(raw_arefs[k])
                    break
        else:
            # Re-apply conversion factor to raw 'energy' if it was 1.0 (Hartree)
            # but we know we are in eV (energy_conv > 1)
            if energy_conv > 1.0 and torch.abs(arefs[structure.energy][1]) < 1.0:
                arefs[structure.energy] = energy_conv * torch.tensor(
                    raw_arefs.get(structure.energy, raw_arefs.get("ts_free_energy"))
                )

        return arefs

    @property
    def units(self) -> Dict[str, str]:
        units = copy(self.base_dataset.units)
        if "ts_free_energy" in units:
            if structure.energy not in units:
                units[structure.energy] = units["ts_free_energy"]
        return units

    @staticmethod
    def create(datapath, position_unit, property_unit_dict, atomrefs=None, **kwargs):
        raise NotImplementedError("RGD1FlattenedWrapper does not support creation. Create the base RGD1 dataset first.")

    def add_system(self, atoms=None, **properties):
        raise NotImplementedError("RGD1FlattenedWrapper is read-only.")

    def add_systems(self, property_list, atoms_list=None):
        raise NotImplementedError("RGD1FlattenedWrapper is read-only.")

    def update_metadata(self, **kwargs):
        self.base_dataset.update_metadata(**kwargs)

    def iter_properties(
        self,
        indices: Union[int, Iterable[int]] = None,
        load_properties: List[str] = None,
        load_structure: Optional[bool] = None,
    ):
        if indices is None:
            indices = range(len(self))
        if isinstance(indices, int):
            indices = [indices]
            
        for i in indices:
            yield self[i]

class RGD1(AtomsDataModule):
    """
    RGD1 dataset for reaction properties.
    Contains over 170,000 elementary reactions with transition state geometries and energies.

    References:
        .. [#rgd1_1] https://doi.org/10.6084/m9.figshare.21066901
        .. [#rgd1_2] https://doi.org/10.5281/zenodo.7860446
    """

    # Zenodo download links (using the corrected version)
    zenodo_url = "https://zenodo.org/record/7860446/files/"
    files = {
        "reactions": "RGD1_CHNO.h5",
        "molecules": "RGD1_uniqueRs.h5",
        "smiles_dict": "RandP_smiles.txt"
    }

    # QM9 Figshare links for atomrefs
    qm9_base_url = "https://ndownloader.figshare.com/files/"
    qm9_atomrefs_id = "3195395"

    # Properties
    reactants_positions = "reactants_positions"
    products_positions = "products_positions"
    ts_positions = "ts_positions"
    reactants_smiles_positions = "reactants_smiles_positions"
    products_smiles_positions = "products_smiles_positions"
    
    reactants_energy = "reactants_energy"
    products_energy = "products_energy"
    ts_energy = "ts_energy"
    
    reactants_enthalpy = "reactants_enthalpy"
    products_enthalpy = "products_enthalpy"
    ts_enthalpy = "ts_enthalpy"
    
    reactants_free_energy = "reactants_free_energy"
    products_free_energy = "products_free_energy"
    ts_free_energy = "ts_free_energy"

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        num_train: Optional[Union[int, float]] = None,
        num_val: Optional[Union[int, float]] = None,
        num_test: Optional[Union[int, float]] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = AtomsDataFormat.LMDB,
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        use_smiles: bool = False,
        optimize_geometries: bool = False,
        num_workers: int = 12,
        **kwargs,
    ):
        self.use_smiles = use_smiles
        self.optimize_geometries = optimize_geometries
        self.num_workers = num_workers

        super().__init__(
            datapath=datapath,
            batch_size=batch_size,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            split_file=split_file,
            format=format,
            load_properties=load_properties,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=num_workers,
            num_val_workers=num_val_workers,
            num_test_workers=num_test_workers,
            property_units=property_units,
            distance_unit=distance_unit,
            data_workdir=data_workdir,
            **kwargs,
        )

    def _download_file(self, filename: str, destination: str):
        url = f"{self.zenodo_url}{filename}?download=1"
        logging.info(f"Downloading {filename} from {url}...")
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            with open(destination, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    out_file.write(chunk)
        except Exception as e:
            raise AtomsDataModuleError(f"Could not download {filename}. Error: {e}")

    def _download_qm9_atomrefs(self, destination: str):
        url = f"{self.qm9_base_url}{self.qm9_atomrefs_id}"
        logging.info(f"Downloading QM9 atomrefs from {url}...")
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            with open(destination, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    out_file.write(chunk)
        except Exception as e:
            raise AtomsDataModuleError(f"Could not download QM9 atomrefs. Error: {e}")
        logging.info("Done.")


    def prepare_data(self):
        db_path, db_format = resolve_format(self.datapath, self.format)
        self.datapath = db_path
        self.format = db_format

        if not os.path.exists(self.datapath):
            # Try to find local files in common locations
            data_dir_from_path = os.path.dirname(self.datapath)
            # Add 'data' directory relative to project root (cwd)
            search_dirs = [data_dir_from_path, "data", os.path.join(os.getcwd(), "data"), "."]
            
            local_rxn_path = None
            found_dir = None
            for d in search_dirs:
                if d is None: continue
                p = os.path.join(d, self.files["reactions"])
                if os.path.exists(p):
                    local_rxn_path = p
                    found_dir = d
                    break
            
            if local_rxn_path:
                logging.info(f"Found local dataset at {local_rxn_path}. Skipping download.")
                self._create_dataset_from_local(found_dir)
            else:
                logging.warning(f"Could not find local dataset {self.files['reactions']} in {search_dirs}. Attempting download.")
                self._download_and_create_dataset()

    def _create_dataset_from_local(self, data_dir: str):
        reaction_path = os.path.join(data_dir, self.files["reactions"])
        
        # Always download atomrefs from QM9
        tmpdir = tempfile.mkdtemp(prefix="rgd1_atomrefs")
        atomrefs_path = os.path.join(tmpdir, "atomrefs.txt")
        self._download_qm9_atomrefs(atomrefs_path)
        
        atref_g = np.zeros((100,))
        with open(atomrefs_path) as f:
            lines = f.readlines()
            for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
                atref_g[z] = float(l.split()[5])
        shutil.rmtree(tmpdir)
        
        atomrefs = {
            self.reactants_free_energy: atref_g.tolist(),
            self.products_free_energy: atref_g.tolist(),
            self.ts_free_energy: atref_g.tolist(),
            structure.energy: atref_g.tolist(),
        }

        property_unit_dict = {
            self.ts_energy: "Ha",
            self.reactants_energy: "Ha",
            self.products_energy: "Ha",
            self.ts_enthalpy: "Ha",
            self.reactants_enthalpy: "Ha",
            self.products_enthalpy: "Ha",
            self.ts_free_energy: "Ha",
            self.reactants_free_energy: "Ha",
            self.products_free_energy: "Ha",
            structure.energy: "Ha",
            self.ts_positions: "Ang",
            self.reactants_positions: "Ang",
            self.products_positions: "Ang",
            self.reactants_smiles_positions: "Ang",
            self.products_smiles_positions: "Ang",
            "reactants_Z": "None",
            "products_Z": "None",
            "ts_Z": "None",
        }

        dataset = create_dataset(
            datapath=self.datapath,
            format=self.format,
            distance_unit="Ang",
            property_unit_dict=property_unit_dict,
            atomrefs=atomrefs,
        )

        self._parse_and_add_to_dataset(reaction_path, dataset)

    def _download_and_create_dataset(self):
        tmpdir = tempfile.mkdtemp(prefix="rgd1")
        
        reaction_path = os.path.join(tmpdir, self.files["reactions"])
        molecule_path = os.path.join(tmpdir, self.files["molecules"])
        smiles_dict_path = os.path.join(tmpdir, self.files["smiles_dict"])
        atomrefs_path = os.path.join(tmpdir, "atomrefs.txt")

        self._download_file(self.files["reactions"], reaction_path)
        # self._download_file(self.files["molecules"], molecule_path)
        # self._download_file(self.files["smiles_dict"], smiles_dict_path)
        self._download_qm9_atomrefs(atomrefs_path)

        # Parse QM9 atomrefs for Gibbs Free Energy (G is the 5th property in atomrefs.txt)
        atref_g = np.zeros((100,))
        with open(atomrefs_path) as f:
            lines = f.readlines()
            for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
                # QM9 props in atomrefs.txt: zpve, U0, U, H, G, Cv
                # G is index 5 (after atom label)
                atref_g[z] = float(l.split()[5])
        
        atomrefs = {
            self.reactants_free_energy: atref_g.tolist(),
            self.products_free_energy: atref_g.tolist(),
            self.ts_free_energy: atref_g.tolist(),
            structure.energy: atref_g.tolist(),
        }

        property_unit_dict = {
            self.ts_energy: "Ha",
            self.reactants_energy: "Ha",
            self.products_energy: "Ha",
            self.ts_enthalpy: "Ha",
            self.reactants_enthalpy: "Ha",
            self.products_enthalpy: "Ha",
            self.ts_free_energy: "Ha",
            self.reactants_free_energy: "Ha",
            self.products_free_energy: "Ha",
            structure.energy: "Ha",  # Default energy is Gibbs free energy of TS
            self.ts_positions: "Ang",
            self.reactants_positions: "Ang",
            self.products_positions: "Ang",
            self.reactants_smiles_positions: "Ang",
            self.products_smiles_positions: "Ang",
            "reactants_Z": "None",
            "products_Z": "None",
            "ts_Z": "None",
        }

        dataset = create_dataset(
            datapath=self.datapath,
            format=self.format,
            distance_unit="Ang",
            property_unit_dict=property_unit_dict,
            atomrefs=atomrefs,
        )

        self._parse_and_add_to_dataset(reaction_path, dataset)
        shutil.rmtree(tmpdir)

    def get_stats(
        self, property: str, divide_by_atoms: bool, remove_atomref: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Special handling for delta_e which is a virtual property
        if property == "delta_e":
            key = ("delta_e", divide_by_atoms, remove_atomref)
            if key in self._stats:
                return self._stats[key]

            logging.info("Calculating stats for delta_e...")
            # We iterate over the training data to get the exact mean and std for delta_e
            loader = self.train_dataloader()
            
            all_deltas = []
            for batch in tqdm(loader, desc="Calculating delta_e stats"):
                # Get raw components
                r_e = batch[self.reactants_free_energy]
                ts_e = batch[self.ts_free_energy]
                
                # Subtract atomrefs if requested
                if remove_atomref and self.train_dataset.atomrefs is not None:
                    aref = self.train_dataset.atomrefs
                    z = batch[structure.Z]
                    idx_m = batch[structure.idx_m]
                    
                    r0 = scatter_add(aref[self.reactants_free_energy][z], idx_m)
                    ts0 = scatter_add(aref[self.ts_free_energy][z], idx_m)
                    
                    r_e = r_e - r0
                    ts_e = ts_e - ts0
                
                if divide_by_atoms:
                    n_atoms = batch[structure.n_atoms]
                    r_e = r_e / n_atoms
                    ts_e = ts_e / n_atoms
                    
                delta = ts_e - r_e
                all_deltas.append(delta.detach().cpu())
                
            all_deltas = torch.cat(all_deltas)
            mean_delta = torch.mean(all_deltas)
            std_delta = torch.std(all_deltas)
            
            stats = (mean_delta, std_delta)
            self._stats[key] = stats
            return stats

        # For free energy properties, we want consistent stats to ensure a unified potential surface
        free_energy_props = [
            self.reactants_free_energy,
            self.products_free_energy,
            self.ts_free_energy,
        ]
        if property in free_energy_props:
            # Check if we already have combined stats
            key = ("combined_free_energy", divide_by_atoms, remove_atomref)
            if key in self._stats:
                return self._stats[key]

            # Calculate stats over all free energy properties simultaneously
            # We use a custom dataloader that yields all states to get the global mean
            logging.info(f"Calculating combined stats for {free_energy_props}...")
            
            # Use calculate_stats with multiple properties
            all_stats = calculate_stats(
                self.train_dataloader(),
                divide_by_atoms={p: divide_by_atoms for p in free_energy_props},
                atomref=self.train_dataset.atomrefs if remove_atomref else None,
            )
            
            # Average the means and variances (M2) to get the combined stats
            # Since calculate_stats uses Welford's, we have to be careful.
            # However, for simply getting a consistent mean, we can average them if the counts are the same.
            means = torch.stack([all_stats[p][0] for p in free_energy_props])
            stddevs = torch.stack([all_stats[p][1] for p in free_energy_props])
            
            combined_mean = torch.mean(means)
            # For stddev, we take the mean of variances (approximate) or just the mean of stddevs
            combined_stddev = torch.mean(stddevs)
            
            stats = (combined_mean, combined_stddev)
            self._stats[key] = stats
            return stats

        return super().get_stats(property, divide_by_atoms, remove_atomref)

    def _parse_and_add_to_dataset(self, reaction_path: str, dataset: BaseAtomsData):
        logging.info(f"Parsing reactions from {reaction_path} using {self.num_workers} workers...")
        
        with h5py.File(reaction_path, 'r') as rxns:
            r_ids = list(rxns.keys())
            batch_size = 1000
            
            for i in tqdm(range(0, len(r_ids), batch_size), desc="Processing RGD1 reactions"):
                batch_ids = r_ids[i:i+batch_size]
                property_list = []
                
                # Pre-collect data and tasks for the batch
                smiles_tasks = []
                batch_data_info = []
                
                for r_id in batch_ids:
                    rxn = rxns[r_id]
                    required_keys = ['elements', 'TSG', 'RG', 'PG', 'R_F', 'P_F', 'TS_F', 'R_H', 'P_H', 'TS_H', 'R_E', 'P_E', 'TS_E', 'Rsmiles', 'Psmiles']
                    if not all(key in rxn for key in required_keys):
                        continue

                    elements = np.array(rxn.get('elements'))
                    r_smiles = rxn.get('Rsmiles')[()].decode('utf-8')
                    p_smiles = rxn.get('Psmiles')[()].decode('utf-8')
                    
                    # Store info for later reconstruction
                    info = {
                        "r_id": r_id,
                        "elements": elements,
                        "r_smiles": r_smiles,
                        "p_smiles": p_smiles,
                        "ts_g": np.array(rxn.get('TSG')),
                        "r_g": np.array(rxn.get('RG')),
                        "p_g": np.array(rxn.get('PG')),
                        "ts_f": np.array(rxn.get('TS_F')),
                        "r_f": np.array(rxn.get('R_F')),
                        "p_f": np.array(rxn.get('P_F')),
                        "ts_h": np.array(rxn.get('TS_H')),
                        "r_h": np.array(rxn.get('R_H')),
                        "p_h": np.array(rxn.get('P_H')),
                        "ts_e": np.array(rxn.get('TS_E')),
                        "r_e": np.array(rxn.get('R_E')),
                        "p_e": np.array(rxn.get('P_E')),
                    }
                    batch_data_info.append(info)
                    
                    if self.use_smiles:
                        smiles_tasks.append(r_smiles)
                        smiles_tasks.append(p_smiles)
                
                # Parallel SMILES processing
                smiles_results = []
                if self.use_smiles and smiles_tasks:
                    with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                        smiles_results = list(executor.map(_rgd1_smiles_worker, smiles_tasks))

                # Reconstruct properties and add to dataset
                for idx, info in enumerate(batch_data_info):
                    elements = info["elements"]
                    
                    # TS is ALWAYS default geometry
                    ts_pos, ts_z = info["ts_g"], elements
                    
                    if self.use_smiles:
                        r_res = smiles_results[idx * 2]
                        p_res = smiles_results[idx * 2 + 1]
                        
                        if r_res[0] is not None:
                            r_pos, r_z = r_res
                        else:
                            # Fallback to separated H5
                            r_pos, r_z = _separate_fragments(info["r_g"]), elements
                            
                        if p_res[0] is not None:
                            p_pos, p_z = p_res
                        else:
                            # Fallback to separated H5
                            p_pos, p_z = _separate_fragments(info["p_g"]), elements
                    else:
                        # Use default H5 geometry but FIX LOSS by separating fragments
                        r_pos, r_z = _separate_fragments(info["r_g"]), elements
                        p_pos, p_z = _separate_fragments(info["p_g"]), elements

                    props = {
                        structure.Z: ts_z.astype(np.int64),
                        "reactants_Z": r_z.astype(np.int64),
                        "products_Z": p_z.astype(np.int64),
                        "ts_Z": ts_z.astype(np.int64),
                        structure.R: ts_pos.astype(np.float32),
                        self.ts_positions: ts_pos.astype(np.float32),
                        self.reactants_positions: r_pos.astype(np.float32),
                        self.products_positions: p_pos.astype(np.float32),
                        self.ts_free_energy: info["ts_f"].astype(np.float32),
                        self.reactants_free_energy: info["r_f"].astype(np.float32),
                        self.products_free_energy: info["p_f"].astype(np.float32),
                        self.ts_enthalpy: info["ts_h"].astype(np.float32),
                        self.reactants_enthalpy: info["r_h"].astype(np.float32),
                        self.products_enthalpy: info["p_h"].astype(np.float32),
                        self.ts_energy: info["ts_e"].astype(np.float32),
                        self.reactants_energy: info["r_e"].astype(np.float32),
                        self.products_energy: info["p_e"].astype(np.float32),
                        structure.energy: info["ts_f"].astype(np.float32),
                        structure.cell: np.zeros((3, 3), dtype=np.float32),
                        structure.pbc: np.zeros(3, dtype=bool),
                        self.reactants_smiles_positions: r_pos.astype(np.float32),
                        self.products_smiles_positions: p_pos.astype(np.float32)
                    }
                    property_list.append(props)

                
                if property_list:
                    dataset.add_systems(property_list=property_list)
