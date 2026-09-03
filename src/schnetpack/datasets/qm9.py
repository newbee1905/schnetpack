import io
import logging
import multiprocessing as mp
import os
import re
import shutil
import tarfile
import tempfile
import time
from typing import List, Optional, Dict
import requests

import numpy as np
from ase import Atoms
from ase.io.extxyz import read_xyz
from tqdm import tqdm

import torch
from schnetpack.data import *
from schnetpack.data import LMDBAtomsData 
import schnetpack.properties as structure
from schnetpack.data import AtomsDataModuleError, AtomsDataModule

from rdkit import Chem
from openbabel import openbabel
from io import StringIO

from ase import Atoms
from ase.optimize import LBFGS
from ase.io import write as ase_write_io
from ase.io.extxyz import read_xyz

__all__ = ["QM9"]

def _optimize_molecule_worker(task_data):
    """
    Worker function to perform TBLite optimization for a single molecule.
    """
    from openbabel import openbabel
    from io import StringIO
    from ase.io import write as ase_write_io
    from ase.io.extxyz import read_xyz
    from ase import Atoms
    import schnetpack.properties as structure

    import numpy as np
    
    (
        idx,
        xyzfile_lines,
        available_properties,
        optimize_geometries_flag,
        use_smiles_flag,
        distance_threshold,
        store_both_flag,
    ) = task_data

    properties = {}
    # Parse properties from the second line of the XYZ file
    l = xyzfile_lines[1].split()[2:]
    for pn, p in zip(available_properties, l):
        properties[pn] = np.array([float(p)])

    tmp_str = "".join(xyzfile_lines).replace("*^", "e")
    tmp = StringIO(tmp_str)
    tmp.seek(0)

    original_ats: Atoms = list(read_xyz(tmp, 0))[0]
    final_ats = original_ats.copy()
    num_atoms = len(original_ats)

    if use_smiles_flag:
        try:
            smiles = ""
            # Extract SMILES directly from standard QM9 file footer
            if len(xyzfile_lines) > num_atoms + 3:
                smiles_line = xyzfile_lines[num_atoms + 3].strip()
                if smiles_line and not smiles_line.replace(".", "").replace("-", "").isdigit():
                    smiles = smiles_line.split()[0]

            # Setup OpenBabel for SMILES conversion and mapping
            tmp_xyz_io = StringIO()
            ase_write_io(tmp_xyz_io, original_ats, format="xyz")
            xyz_str = tmp_xyz_io.getvalue()

            ob_conversion = openbabel.OBConversion()
            ob_conversion.SetInFormat("xyz")
            ob_mol = openbabel.OBMol()
            ob_conversion.ReadString(ob_mol, xyz_str)

            if not smiles:
                tmp_xyz_io = StringIO()
                ase_write_io(tmp_xyz_io, original_ats, format="xyz")
                xyz_str = tmp_xyz_io.getvalue()

                ob_conversion = openbabel.OBConversion()
                ob_conversion.SetInFormat("xyz")
                ob_mol = openbabel.OBMol()
                ob_conversion.ReadString(ob_mol, xyz_str)

                ob_conversion.SetOutFormat("can")
                smiles = ob_conversion.WriteString(ob_mol).strip()

            # Generate 3D structure from SMILES using Open Babel
            ob_mol_smi = openbabel.OBMol()
            ob_conversion.SetInFormat("smi")
            ob_conversion.ReadString(ob_mol_smi, smiles)

            ob_mol_smi.AddHydrogens()

            builder = openbabel.OBBuilder()
            builder.Build(ob_mol_smi)
            
            ff = openbabel.OBForceField.FindForceField("mmff94")
            if ff is None:
                raise ValueError("Could not find MMFF94 force field. Check Open Babel installation.")

            ff.Setup(ob_mol_smi)
            ff.SteepestDescent(500)
            ff.GetCoordinates(ob_mol_smi)

            # Convert Open Babel mol to RDKit mol to ensure consistent indexing
            # for the isomorphism check while keeping OB coordinates.
            ob_conversion.SetOutFormat("mol")
            mol_block = ob_conversion.WriteString(ob_mol_smi)
            mol_smi = Chem.MolFromMolBlock(mol_block, removeHs=False)
            
            if mol_smi is None:
                raise ValueError(f"RDKit failed to import Open Babel mol block for SMILES: {smiles}")

            # Extract final atoms from the RDKit-wrapped OB geometry
            conf = mol_smi.GetConformer()
            new_pos = conf.GetPositions()
            new_z = [a.GetAtomicNum() for a in mol_smi.GetAtoms()]
            final_ats = Atoms(numbers=new_z, positions=new_pos)

            # Map original positions to new atom order if requested
            if store_both_flag:
                from rdkit.Chem import rdDetermineBonds

                try:
                    # Load the original XYZ into RDKit and perceive bonds
                    tmp_io = StringIO()
                    ase_write_io(tmp_io, original_ats, format="xyz")
                    clean_xyz_block = tmp_io.getvalue()
                    
                    raw_mol_xyz = Chem.MolFromXYZBlock(clean_xyz_block)
                    if raw_mol_xyz is None:
                        raise ValueError("RDKit failed to parse the clean XYZ block.")

                    rdDetermineBonds.DetermineConnectivity(raw_mol_xyz)
                    rdDetermineBonds.DetermineBondOrders(raw_mol_xyz, charge=0)

                    # Find mapping: SMILES indices (OB-ordered) -> XYZ indices
                    matches = raw_mol_xyz.GetSubstructMatches(mol_smi, uniquify=False)

                    if matches:
                        orig_pos = original_ats.positions
                        best_match = None
                        min_rmsd = float('inf')
                        
                        # Center the SMILES conformer for alignment
                        new_pos_centered = new_pos - np.mean(new_pos, axis=0)
                        
                        for match in matches:
                            reordered_orig = orig_pos[list(match)]
                            reordered_orig_centered = reordered_orig - np.mean(reordered_orig, axis=0)
                            
                            # Optimal rotation (Kabsch algorithm)
                            cov = reordered_orig_centered.T @ new_pos_centered
                            u, s, vh = np.linalg.svd(cov)
                            d = np.linalg.det(u @ vh)
                            if d < 0:
                                u[:, -1] *= -1
                            rot = u @ vh
                            
                            # Rotate original coordinates to match SMILES orientation
                            aligned_orig = reordered_orig_centered @ rot
                            
                            rmsd = np.sqrt(np.mean((aligned_orig - new_pos_centered)**2))
                            if rmsd < min_rmsd:
                                min_rmsd = rmsd
                                best_match = reordered_orig
                                
                        properties["R_real"] = best_match.astype(np.float32)
                    else:
                        raise ValueError(f"No graph isomorphism found between SMILES and original XYZ.")

                except Exception as e:
                    logging.warning(f"RDKit Alignment failed for molecule {idx}: {e}. Skipping molecule.")
                    return idx, original_ats, properties, "alignment_failed"

        except Exception as e:
            logging.warning(
                f"SMILES conversion failed for molecule {idx}. Error: {e}. Skipping molecule."
            )
            return idx, original_ats, properties, "smiles_failed"

    if optimize_geometries_flag:
        try:
            # Convert current ASE Atoms (either original or SMILES-generated) to XYZ string
            tmp_xyz_io = StringIO()
            ase_write_io(tmp_xyz_io, final_ats, format="xyz")
            xyz_str = tmp_xyz_io.getvalue()

            # Convert XYZ string to OpenBabel OBMol
            ob_conversion = openbabel.OBConversion()
            ob_conversion.SetInFormat("xyz")
            ob_mol = openbabel.OBMol()
            ob_conversion.ReadString(ob_mol, xyz_str)

            # Use MMFF94 force field for optimization via OpenBabel
            ff = openbabel.OBForceField.FindForceField("mmff94")
            if ff is None:
                ff = openbabel.OBForceField.FindForceField("uff")
                
            if ff is None:
                raise RuntimeError("Could not find MMFF94 or UFF force field in OpenBabel.")
                
            ff.Setup(ob_mol)
            ff.SteepestDescent(1000, 1.0e-4)
            ff.WeightedRotorSearch(50, 20)
            ff.ConjugateGradients(3000, 1.0e-6)
            ff.GetCoordinates(ob_mol)

            # Convert optimized OBMol back to XYZ string for ASE
            ob_conversion.SetOutFormat("xyz")
            optimized_xyz_str = ob_conversion.WriteString(ob_mol)

            # Read into ASE Atoms object
            optimized_ats = list(read_xyz(StringIO(optimized_xyz_str), 0))[0]
            final_ats = optimized_ats

        except Exception as e:
            logging.warning(
                f"OpenBabel optimization failed for molecule {idx}. Error: {e}. Skipping molecule."
            )
            return idx, original_ats, properties, "opt_failed"

    if np.isnan(final_ats.positions).any():
        logging.warning(f"Molecule {idx} has NaN positions; flagging geometry invalid.")
        return idx, original_ats, properties, "nan"

    if distance_threshold > 0:
        dm = final_ats.get_all_distances()
        np.fill_diagonal(dm, np.inf)
        if np.any(dm < distance_threshold):
            logging.warning(
                f"Molecule {idx} has atoms closer than {distance_threshold} A; "
                f"flagging geometry invalid."
            )
            return idx, original_ats, properties, "distance"

    return idx, final_ats, properties, None




class QM9(AtomsDataModule):
    """QM9 benchmark database for organic molecules.

    The QM9 database contains small organic molecules with up to nine non-hydrogen atoms
    from including C, O, N, F. This class adds convenient functions to download QM9 from
    figshare and load the data into pytorch.

    References:

        .. [#qm9_1] https://ndownloader.figshare.com/files/3195404
    """

    base_urls = [
        "https://ndownloader.figshare.com/files/",
        "https://api.figshare.com/v2/file/download/",
        "https://springernature.figshare.com/ndownloader/files/",
    ]
    file_ids = {
        "data": "3195389",
        "atomrefs": "3195395",
        "uncharacterized": "3195404",
    }

    # marks whether the stored geometry is usable (SMILES builds only)
    geometry_valid = "geometry_valid"
    # metadata key holding row indices whose geometry is unusable
    invalid_geometry_key = "invalid_geometry_idx"

    # properties
    A = "rotational_constant_A"
    B = "rotational_constant_B"
    C = "rotational_constant_C"
    mu = "dipole_moment"
    alpha = "isotropic_polarizability"
    homo = "homo"
    lumo = "lumo"
    gap = "gap"
    r2 = "electronic_spatial_extent"
    zpve = "zpve"
    U0 = "energy_U0"
    U = "energy_U"
    H = "enthalpy_H"
    G = "free_energy"
    Cv = "heat_capacity"

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        num_train: Optional[int] = None,
        num_val: Optional[int] = None,
        num_test: Optional[int] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = AtomsDataFormat.LMDB,
        load_properties: Optional[List[str]] = None,
        remove_uncharacterized: bool = False,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 2,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        optimize_geometries: bool = False,
        use_smiles: bool = False,
        distance_threshold: float = 0.15,
        store_both_geometries: bool = False,
        **kwargs,
    ):
        """

        Args:
            datapath: path to dataset
            batch_size: (train) batch size
            num_train: number of training examples
            num_val: number of validation examples
            num_test: number of test examples
            split_file: path to npz file with data partitions
            format: dataset format
            load_properties: subset of properties to load
            remove_uncharacterized: do not include uncharacterized molecules.
            val_batch_size: validation batch size. If None, use test_batch_size, then batch_size.
            test_batch_size: test batch size. If None, use val_batch_size, then batch_size.
            transforms: Transform applied to each system separately before batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers (overrides num_workers).
            num_test_workers: Number of test data loader workers (overrides num_workers).
            property_units: Dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
            data_workdir: Copy data here as part of setup, e.g. cluster scratch for faster performance.
            distance_threshold: Threshold for filtering molecules with atoms too close.
            store_both_geometries: If True, store both SMILES-generated and original geometries.
        """
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

        self.remove_uncharacterized = remove_uncharacterized
        self.optimize_geometries = optimize_geometries
        self.use_smiles = use_smiles
        self.distance_threshold = distance_threshold
        self.store_both_geometries = store_both_geometries

    def _download_file(
        self,
        file_id: str,
        destination: str,
        n_retries: int = 8,
        timeout: tuple = (10, 120),
    ):
        """Fetch a figshare file id, trying each mirror in `base_urls` in turn.

        figshare answers a request it cannot serve immediately with a bare 202
        (accepted, still being prepared) and an empty body, which
        `urllib.request.urlretrieve` happily writes out as a 0-byte file. Poll
        past the 202 before falling through to the next mirror.

        The gdb9 archive is ~86 MB and figshare redirects to S3, which stalls
        mid-body on a slow link. Body reads are therefore retried too, resuming
        with a Range request so a stall does not discard what already landed.

        Note that figshare returns 202 indefinitely for requests carrying a
        browser-like User-Agent, so this deliberately sends no custom headers.
        """
        session = requests.Session()

        for base_url in self.base_urls:
            url = f"{base_url}{file_id}"
            logging.info(f"Attempting to download from {url}...")

            for attempt in range(n_retries):
                resume_at = (
                    os.path.getsize(destination) if os.path.exists(destination) else 0
                )
                headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}

                try:
                    response = session.get(
                        url, stream=True, timeout=timeout, headers=headers
                    )

                    if response.status_code == 202:
                        delay = 2 * (attempt + 1)
                        logging.warning(
                            f"Got 202 from {url} (file not ready), retrying in "
                            f"{delay}s ({attempt + 1}/{n_retries})..."
                        )
                        time.sleep(delay)
                        continue

                    if response.status_code not in (200, 206):
                        logging.warning(
                            f"Download from {url} failed with status "
                            f"{response.status_code}"
                        )
                        break

                    # 206 means the server honoured the Range and we can append;
                    # a 200 in reply to a Range request means it ignored it, so
                    # start the file over.
                    appending = response.status_code == 206 and resume_at > 0
                    if resume_at and not appending:
                        logging.info(f"{url} ignored Range, restarting download")
                        resume_at = 0

                    with open(destination, "ab" if appending else "wb") as out_file:
                        for chunk in response.iter_content(chunk_size=1 << 16):
                            out_file.write(chunk)

                    logging.info(
                        f"Downloaded {file_id} to {destination} "
                        f"({os.path.getsize(destination)} bytes)"
                    )
                    return

                except Exception as e:
                    got = (
                        os.path.getsize(destination)
                        if os.path.exists(destination)
                        else 0
                    )
                    delay = 2 * (attempt + 1)
                    logging.warning(
                        f"Transfer from {url} interrupted after {got} bytes "
                        f"({type(e).__name__}: {e}); resuming in {delay}s "
                        f"({attempt + 1}/{n_retries})..."
                    )
                    time.sleep(delay)
                    continue

        raise AtomsDataModuleError(
            f"Could not download file with id {file_id} from any of: "
            + ", ".join(self.base_urls)
        )

    # def prepare_data(self):
    #     if not os.path.exists(self.datapath):
    #         property_unit_dict = {
    #             QM9.A: "GHz",
    #             QM9.B: "GHz",
    #             QM9.C: "GHz",
    #             QM9.mu: "Debye",
    #             QM9.alpha: "a0 a0 a0",
    #             QM9.homo: "Ha",
    #             QM9.lumo: "Ha",
    #             QM9.gap: "Ha",
    #             QM9.r2: "a0 a0",
    #             QM9.zpve: "Ha",
    #             QM9.U0: "Ha",
    #             QM9.U: "Ha",
    #             QM9.H: "Ha",
    #             QM9.G: "Ha",
    #             QM9.Cv: "cal/mol/K",
    #         }

    def _convert_ase_to_lmdb(self, ase_db_path: str, lmdb_path: str):
        ase_dataset = load_dataset(ase_db_path, AtomsDataFormat.ASE)
        property_unit_dict = ase_dataset.metadata["_property_unit_dict"]
        distance_unit = ase_dataset.metadata["_distance_unit"]
        atomrefs = ase_dataset.metadata["atomrefs"]

        lmdb_dataset = create_dataset(
            datapath=lmdb_path,
            format=AtomsDataFormat.LMDB,
            distance_unit=distance_unit,
            property_unit_dict=property_unit_dict,
            atomrefs=atomrefs,
        )

        all_properties = []
        for i in tqdm(range(len(ase_dataset)), desc="Converting ASE to LMDB"):
            data = ase_dataset[i]
            properties = {k: v.numpy() for k, v in data.items() if k not in [structure.Z, structure.R, structure.cell, structure.pbc, structure.idx, structure.n_atoms]}
            
            # Extract structure properties and ensure they are numpy arrays
            properties[structure.Z] = data[structure.Z].numpy()
            properties[structure.R] = data[structure.R].numpy()
            properties[structure.cell] = data[structure.cell].numpy().squeeze()
            properties[structure.pbc] = data[structure.pbc].numpy()
            all_properties.append(properties)

        lmdb_dataset.add_systems(property_list=all_properties)
        del ase_dataset # release the connection to the ase_db
        

    def _download_and_create_dataset(self, target_datapath: str, target_format: AtomsDataFormat):
        property_unit_dict = {
            QM9.A: "GHz",
            QM9.B: "GHz",
            QM9.C: "GHz",
            QM9.mu: "Debye",
            QM9.alpha: "a0 a0 a0",
            QM9.homo: "Ha",
            QM9.lumo: "Ha",
            QM9.gap: "Ha",
            QM9.r2: "a0 a0",
            QM9.zpve: "Ha",
            QM9.U0: "Ha",
            QM9.U: "Ha",
            QM9.H: "Ha",
            QM9.G: "Ha",
            QM9.Cv: "cal/mol/K",
        }
        if self.store_both_geometries:
            property_unit_dict["R_real"] = "Ang"
        # Rows whose SMILES geometry could not be built are still written, so
        # this dataset stays index-parallel with the plain QM9 build. The flag
        # marks which geometries are usable; the split drops the rest.
        property_unit_dict[QM9.geometry_valid] = ""

        tmpdir = tempfile.mkdtemp("qm9")
        atomrefs = self._download_atomrefs(tmpdir)

        dataset = create_dataset(
            datapath=target_datapath,
            format=target_format,
            distance_unit="Ang",
            property_unit_dict=property_unit_dict,
            atomrefs=atomrefs,
        )

        if self.remove_uncharacterized:
            uncharacterized = self._download_uncharacterized(tmpdir)
        else:
            uncharacterized = None
        self._download_data(tmpdir, dataset, uncharacterized=uncharacterized)
        shutil.rmtree(tmpdir)
        
    def prepare_data(self):
        # Resolve the data path and format
        original_datapath = self.datapath
        db_path, db_format = resolve_format(self.datapath, self.format)
        self.datapath = db_path
        self.format = db_format

        if not os.path.exists(self.datapath):
            if self.format == AtomsDataFormat.LMDB:
                ase_db_path = os.path.splitext(self.datapath)[0] + ".db"
                if os.path.exists(ase_db_path):
                    logging.info(
                        f"Converting ASE DB at {ase_db_path} to LMDB at {self.datapath}..."
                    )
                    self._convert_ase_to_lmdb(ase_db_path, self.datapath)
                    logging.info("Conversion complete.")
                else:
                    logging.info(
                        f"Neither LMDB nor ASE DB found at {self.datapath} or {ase_db_path}. Downloading and creating new dataset."
                    )
                    self._download_and_create_dataset(self.datapath, self.format)
            elif self.format == AtomsDataFormat.ASE:
                logging.info(
                    f"ASE DB not found at {self.datapath}. Downloading and creating new dataset."
                )
                self._download_and_create_dataset(self.datapath, self.format)
            else:
                raise AtomsDataModuleError(f"Unsupported format: {self.format}")
        
        # After ensuring the database exists, perform checks for uncharacterized molecules
        dataset = load_dataset(self.datapath, self.format)

        if self.remove_uncharacterized and len(dataset) == 133885:
            raise AtomsDataModuleError(
                "The dataset at the chosen location contains the uncharacterized 3054 molecules. "
                + "Choose a different location to reload the data or set `remove_uncharacterized=False`!"
            )
        elif not self.remove_uncharacterized and len(dataset) < 133885:
            raise AtomsDataModuleError(
                "The dataset at the chosen location does NOT contain the uncharacterized 3054 molecules. "
                + "Choose a different location to reload the data or set `remove_uncharacterized=True`!"
            )

    def _download_uncharacterized(self, tmpdir):
        logging.info("Downloading list of uncharacterized molecules...")
        tmp_path = os.path.join(tmpdir, "uncharacterized.txt")
        self._download_file(self.file_ids["uncharacterized"], tmp_path)
        logging.info("Done.")

        uncharacterized = []
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines[9:-1]:
                uncharacterized.append(int(line.split()[0]))
        return uncharacterized

    def _download_atomrefs(self, tmpdir):
        logging.info("Downloading GDB-9 atom references...")
        tmp_path = os.path.join(tmpdir, "atomrefs.txt")
        self._download_file(self.file_ids["atomrefs"], tmp_path)
        logging.info("Done.")

        props = [QM9.zpve, QM9.U0, QM9.U, QM9.H, QM9.G, QM9.Cv]
        atref = {p: np.zeros((100,)) for p in props}
        with open(tmp_path) as f:
            lines = f.readlines()
            for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
                for i, p in enumerate(props):
                    atref[p][z] = float(l.split()[i + 1])
        atref = {k: v.tolist() for k, v in atref.items()}
        return atref

    def _download_data(
        self, tmpdir, dataset: BaseAtomsData, uncharacterized: List[int]
    ):
        logging.info("Starting _download_data()...")
        logging.info("Downloading GDB-9 data...")
        tar_path = os.path.join(tmpdir, "gdb9.tar.gz")
        raw_path = os.path.join(tmpdir, "gdb9_xyz")
        logging.info("Calling _download_file for data...")
        self._download_file(self.file_ids["data"], tar_path)
        logging.info("Done downloading GDB-9 data.")

        logging.info("Extracting files...")
        tar = tarfile.open(tar_path)
        tar.extractall(raw_path)
        tar.close()
        logging.info("Done.")

        logging.info("Parse xyz files...")
        ordered_files = sorted(
            os.listdir(raw_path), key=lambda x: (int(re.sub(r"\D", "", x)), x)
        )

        # Prepare tasks for the multiprocessing pool
        tasks = []
        irange = np.arange(len(ordered_files), dtype=int)
        if uncharacterized is not None:
            irange = np.setdiff1d(irange, np.array(uncharacterized, dtype=int) - 1)

        # Collect all file contents to avoid workers reading files (might cause contention)
        # This is a trade-off: more memory usage, but potentially faster I/O if files are small
        file_contents_for_tasks = []
        for i in irange: # Use irange to iterate over valid indices
            xyzfile = os.path.join(raw_path, ordered_files[i])
            with open(xyzfile, "r") as f:
                file_contents_for_tasks.append((i, f.readlines())) # Store (original_idx, lines)

        # Create tasks for the worker pool
        tasks = []
        for original_idx, lines in file_contents_for_tasks:
            tasks.append(
                (
                    original_idx, 
                    lines, 
                    dataset.available_properties, 
                    self.optimize_geometries, 
                    self.use_smiles,
                    self.distance_threshold,
                    self.store_both_geometries,
                )
            )

        collected_properties = [] # Collect all properties here
        skip_stats = {
            "nan": 0,
            "distance": 0,
            "smiles_failed": 0,
            "alignment_failed": 0,
            "opt_failed": 0,
            "worker_error": 0,
        }

        # The SMILES/optimisation path is CPU-bound (OpenBabel build + MMFF94
        # relaxation + an RDKit isomorphism search per molecule), so fan it out
        # across processes when more than one worker is configured. The worker
        # is a module-level function taking and returning plain tuples, so it
        # pickles cleanly.
        n_proc = max(1, int(self.num_workers or 1))
        needs_work = self.use_smiles or self.optimize_geometries

        if n_proc > 1 and needs_work:
            logging.info(f"Processing {len(tasks)} molecules across {n_proc} processes")
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_proc) as pool:
                results = list(
                    tqdm(
                        pool.imap(_optimize_molecule_worker, tasks, chunksize=64),
                        total=len(tasks),
                        desc="Processing molecules",
                    )
                )
        else:
            results = [
                _optimize_molecule_worker(task)
                for task in tqdm(tasks, desc="Processing molecules")
            ]

        invalid_idx = []
        for row, (original_idx, final_ats, props, skip_reason) in enumerate(results):
            if final_ats is None or props is None:
                # Dropping a row here would break index parity with the plain
                # QM9 build, which is the whole point of keeping them.
                skip_stats["worker_error"] += 1
                raise AtomsDataModuleError(
                    f"Worker returned nothing for molecule {original_idx}; cannot "
                    f"keep this dataset index-parallel with the plain QM9 build."
                )

            if skip_reason:
                skip_stats[skip_reason] += 1
                invalid_idx.append(row)

            properties = props
            properties[QM9.geometry_valid] = np.array(
                [0.0 if skip_reason else 1.0], dtype=np.float64
            )
            properties[structure.Z] = final_ats.numbers
            properties[structure.R] = final_ats.positions
            
            # If both are stored, props already contains R_real from the worker
            # and it is aligned with final_ats.numbers

            # Explicitly cast the ASE Cell object to a numpy array to prevent LMDB serialization corruption
            properties[structure.cell] = np.array(final_ats.cell.array) if hasattr(final_ats.cell, 'array') else np.array(final_ats.cell)
            properties[structure.pbc] = final_ats.pbc
            collected_properties.append(properties)

        property_list = collected_properties

        if sum(skip_stats.values()) > 0:
            logging.info("QM9 geometry report:")
            for reason, count in skip_stats.items():
                if count > 0:
                    logging.info(f"  - {reason}: {count} molecules flagged invalid")
            logging.info(
                f"Total flagged invalid: {sum(skip_stats.values())} "
                f"(kept in the dataset, excluded at split time)"
            )

        logging.info("Write atoms to db...")
        dataset.add_systems(property_list=property_list)
        # Record unusable rows in dataset metadata so a split can exclude them
        # without a side-car mapping file.
        dataset.update_metadata(**{QM9.invalid_geometry_key: invalid_idx})
        logging.info(f"Done. {len(invalid_idx)} rows flagged invalid.")
