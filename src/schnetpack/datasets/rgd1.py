import logging
import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import h5py
import torch
from tqdm import tqdm
from ase import Atoms
from ase.db import connect

from schnetpack.data import AtomsDataModule, AtomsDataFormat

log = logging.getLogger(__name__)

class RGD1(AtomsDataModule):
    """
    RGD1 Dataset for Conditional Flow Matching.
    
    Data Source:
        - RGD1_CHNO.h5: Contains Reaction data (TS geometry, energies, SMILES keys).
        - RGD1_RPs.h5: Contains Optimized Molecule data (Reactant/Product geometries).
        - RandP_smiles.txt / RGD1CHNO_AMsmiles.csv: Mapping from reaction SMILES keys to Molecule H5 keys.
    
    Structure:
        - Position (R): Transition State (TSG from RGD1_CHNO)
        - Reactant (R_react): Aggregated DFT Geometries of reactants (from RGD1_RPs)
        - Product (R_prod): Aggregated DFT Geometries of products (from RGD1_RPs)
        - Property: Calculated from Energy/Enthalpy/Gibbs (TS - Reactant)

    References:
        .. [#rgd1] https://figshare.com/articles/dataset/model_reaction_database/21066901
    """

    # Mapping user request 'target_property' to (TS_key, Reactant_key) in H5
    # to calculate Delta (Activation).
    # E = Energy, H = Enthalpy, G = Gibbs Free Energy
    PROPERTY_MAP = {
        "E": ("TS_E", "R_E"), 
        "H": ("TS_H", "R_H"), 
        "G": ("TS_F", "R_F") 
    }

    FIGSHARE_URL = "https://figshare.com/ndownloader/articles/21066901/versions/9"
    ZIP_NAME = "RGD1_data.zip"
    
    # Expected filenames inside the zip
    FILE_RXNS = "RGD1_CHNO.h5"
    FILE_MOLS = "RGD1_RPs.h5"

    FILE_MAP_TXT = "RandP_smiles.txt" 
    FILE_MAP_CSV = "RGD1CHNO_AMsmiles.csv"

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        target_property: str = "E",
        num_train: int = None,
        num_val: int = None,
        num_test: int = None,
        split_file: str = "split.npz",
        format: AtomsDataFormat = AtomsDataFormat.ASE,
        load_properties: list = None,
        val_batch_size: int = None,
        test_batch_size: int = None,
        transforms: list = None,
        train_transforms: list = None,
        val_transforms: list = None,
        test_transforms: list = None,
        num_workers: int = 2,
        num_val_workers: int = None,
        num_test_workers: int = None,
        property_units: dict = None,
        distance_unit: str = None,
        **kwargs,
    ):
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
            **kwargs,
        )
        
        if target_property not in self.PROPERTY_MAP:
            raise ValueError(f"Invalid target_property '{target_property}'. Choose 'E', 'H', or 'G'.")
            
        self.target_prop_config = self.PROPERTY_MAP[target_property]
        self.target_key = target_property # The key we will save in ASE db
        
        # We need these for Flow Matching
        self.extra_props = ["R_react", "R_prod", self.target_key]
        
        if self.load_properties is None:
            self.load_properties = self.extra_props
        else:
            for p in self.extra_props:
                if p not in self.load_properties:
                    self.load_properties.append(p)

    def _download_data(self, raw_dir: str):
        """Downloads single zip from Figshare."""
        os.makedirs(raw_dir, exist_ok=True)
        zip_path = os.path.join(raw_dir, self.ZIP_NAME)
        
        if not os.path.exists(zip_path):
            log.info(f"Downloading {self.ZIP_NAME}...")
            try:
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=self.ZIP_NAME) as t:
                    def reporthook(blocknum, blocksize, totalsize):
                        t.total = totalsize
                        t.update(blocksize)
                    urllib.request.urlretrieve(self.FIGSHARE_URL, zip_path, reporthook=reporthook)
            except Exception as e:
                if os.path.exists(zip_path): os.remove(zip_path)
                raise e

        log.info(f"Extracting {self.ZIP_NAME}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
        except zipfile.BadZipFile:
            raise ValueError("Downloaded file is not a valid zip file.")

    def _load_mapping(self, raw_dir):
        """
        Loads the dictionary mapping from Reaction SMILES components 
        to Molecule HDF5 keys.
        """
        rp_dict = {}
        
        txt_path = os.path.join(raw_dir, self.FILE_MAP_TXT)
        csv_path = os.path.join(raw_dir, self.FILE_MAP_CSV)
        
        lines = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding="utf-8") as f:
                lines = f.readlines()

            # Skip header if present
            if len(lines) > 0 and "smiles" in lines[0].lower():
                lines = lines[1:]
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    rp_dict[parts[0]] = parts[1]
                    
        elif os.path.exists(csv_path):
            # Fallback to CSV if TXT is missing
            df = pd.read_csv(csv_path)

            # Assuming columns [0] and [1] correspond to map
            # Adjust based on actual CSV content if needed
            for _, row in df.iterrows():
                rp_dict[str(row[0])] = str(row[1])
        else:
            log.warning("No mapping file (txt/csv) found. Assuming SMILES strings are keys directly.")
            return None

        return rp_dict

    def prepare_data(self):
        if os.path.exists(self.datapath):
            return

        raw_dir = os.path.dirname(self.datapath)
        
        # Check for files
        path_rxns = os.path.join(raw_dir, self.FILE_RXNS)
        path_mols = os.path.join(raw_dir, self.FILE_MOLS)

        if not (os.path.exists(path_rxns) and os.path.exists(path_mols)):
            self._download_data(raw_dir)

        log.info(f"Converting RGD1 data to {self.datapath}...")
        
        # Load Map
        rp_dict = self._load_mapping(raw_dir)

        try:
            with connect(self.datapath) as db, \
                 h5py.File(path_rxns, "r") as f_rxns, \
                 h5py.File(path_mols, "r") as f_mols:
                
                # Iterate over reactions in RGD1_CHNO.h5
                # The guide uses `for Rind, Rxn in rxns.items()`
                for rxn_id, rxn_grp in tqdm(f_rxns.items(), desc="Parsing Reactions"):
                    
                    try:
                        # Basic Info & Atomic Numbers
                        elements = np.array(rxn_grp.get('elements'))
                        
                        # Transition State Geometry (Target)
                        ts_g = np.array(rxn_grp.get('TSG'))
                        
                        # Properties (Calculate Activation)
                        # Keys: TS_E, R_E, etc.
                        ts_prop_key, r_prop_key = self.target_prop_config
                        
                        val_ts = np.array(rxn_grp.get(ts_prop_key))
                        val_r = np.array(rxn_grp.get(r_prop_key))
                        
                        # Calculate Delta (e.g., Ea = TS - R)
                        # These are often arrays of size 1, take item()
                        scalar_prop = float(val_ts.item() - val_r.item())

                        # Reconstruct Reactants (Source)
                        r_smiles = rxn_grp.get('Rsmiles')[()].decode('utf-8')
                        r_parts = r_smiles.split('.')
                        
                        r_coords_list = []
                        
                        for part in r_parts:
                            # Map snippet to H5 key
                            mol_key = part
                            if rp_dict is not None:
                                mol_key = rp_dict.get(part, part)
                            
                            if mol_key not in f_mols:
                                raise KeyError(f"Molecule {mol_key} (from {part}) not found in RPs file.")
                                
                            mol_grp = f_mols[mol_key]
                            # Guide says use 'DFT_G' (Optimized Geometry)
                            dft_g = np.array(mol_grp.get('DFTG')) # specific molecule geom
                            r_coords_list.append(dft_g)

                        # Concatenate reactants to form one system
                        # Assumes order in 'elements' matches 'Rsmiles' split order
                        r_g_combined = np.concatenate(r_coords_list, axis=0)

                        # Reconstruct Products (Conditioning/Aux)
                        p_smiles = rxn_grp.get('Psmiles')[()].decode('utf-8')
                        p_parts = p_smiles.split('.')
                        
                        p_coords_list = []
                        for part in p_parts:
                            mol_key = part
                            if rp_dict is not None:
                                mol_key = rp_dict.get(part, part)
                            
                            mol_grp = f_mols[mol_key]
                            dft_g = np.array(mol_grp.get('DFTG'))
                            p_coords_list.append(dft_g)
                            
                        p_g_combined = np.concatenate(p_coords_list, axis=0)

                        # Ensure atom counts match across TS, Reactant, Product
                        n_atoms = len(elements)
                        if len(ts_g) != n_atoms or len(r_g_combined) != n_atoms or len(p_g_combined) != n_atoms:
                            # Sometimes atom ordering/count differs due to explicit H or mapping errors
                            # Skip these to avoid training crashes
                            continue

                        atoms = Atoms(numbers=elements, positions=ts_g)
                        
                        data_dict = {
                            "R_react": r_g_combined,
                            "R_prod": p_g_combined,
                            self.target_key: scalar_prop
                        }
                        
                        db.write(atoms, data=data_dict)

                    except Exception as e:
                        continue

        except Exception as e:
            if os.path.exists(self.datapath): os.remove(self.datapath)
            raise e
            
        log.info("Conversion complete.")

# vim:ts=4 sw=4 et
