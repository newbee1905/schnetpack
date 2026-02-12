#!/usr/bin/env python

"""
This script provides an example of how to use a deployed SchNetPack model
in TorchScript format ('model.script').

It performs the following steps:
1.  Loads the TorchScript model.
2.  Generates a molecule using RDKit (by picking a random SMILES from a list).
3.  Generates 3D coordinates for the molecule using the openbabel API.
4.  Prepares the atomic inputs (atomic numbers and coordinates) as PyTorch tensors.
5.  Runs inference with the loaded model.
6.  Prints the resulting predictions.
"""

import torch
from rdkit import Chem
from openbabel import openbabel
import numpy as np
import random
import sys
from io import StringIO
import logging

try:
	from tblite.ase import TBLite
	from ase.optimize import LBFGS
	from ase.io import read as ase_read
	from ase import Atoms

	TBLITE_AVAILABLE = True
except ImportError as e:
	TBLITE_AVAILABLE = False
	TBLITE_IMPORT_ERROR = e
	TBLite, LBFGS, ase_read, Atoms = None, None, None, None


def get_molecule_data_from_rdkit_mol(mol, optimize_with_tblite=False):
	"""
	Takes an RDKit mol object, converts it to an Open Babel OBMol object, 
	and generates 3D coordinates. Optionally optimizes with TBLite.
	
	Args:
		mol (rdkit.Chem.Mol): An RDKit molecule object.
		optimize_with_tblite (bool): If True, optimizes the generated structure with TBLite.

	Returns:
		(torch.Tensor, torch.Tensor): A tuple containing a tensor of atomic numbers (Z)
									  and a tensor of 3D coordinates (R).
	"""
	mol_with_h = Chem.AddHs(mol)
	mol_block = Chem.MolToMolBlock(mol_with_h)
	
	ob_conversion = openbabel.OBConversion()
	ob_conversion.SetInFormat("mol")
	ob_mol = openbabel.OBMol()
	ob_conversion.ReadString(ob_mol, mol_block)

	ff = openbabel.OBForceField.FindForceField("mmff94")
	if ff is None:
		raise RuntimeError("Could not find MMFF94 force field. Check Open Babel installation.")
		
	ff.Setup(ob_mol)
	ff.SteepestDescent(500)  
	ff.GetCoordinates(ob_mol)

	if optimize_with_tblite:
		if not TBLITE_AVAILABLE:
			error_message = f"""
TBLite optimization is enabled, but the required packages (tblite, ase) were not found.
Original import error: {TBLITE_IMPORT_ERROR}
"""
			raise ImportError(error_message)
		try:
			ob_conversion.SetOutFormat("xyz")
			xyz_str = ob_conversion.WriteString(ob_mol)
			string_io = StringIO(xyz_str)
			
			atoms = ase_read(string_io, format="xyz")
			
			# Set TBLite calculator and optimize
			atoms.calc = TBLite()
			dyn = LBFGS(atoms)
			dyn.run(fmax=0.05)

			# Extract optimized coordinates and atomic numbers
			atomic_numbers = atoms.get_atomic_numbers()
			coords = atoms.get_positions()
		except Exception as e:
			logging.warning(f"TBLite optimization failed. Using unoptimized Open Babel geometry. Error: {e}")

			# Fallback to Open Babel generated coordinates
			atomic_numbers = [atom.GetAtomicNum() for atom in openbabel.OBMolAtomIter(ob_mol)]
			coords = np.array([[atom.GetX(), atom.GetY(), atom.GetZ()] for atom in openbabel.OBMolAtomIter(ob_mol)])
	else:
		# Extract atomic numbers and coordinates from Open Babel OBMol
		atomic_numbers = [atom.GetAtomicNum() for atom in openbabel.OBMolAtomIter(ob_mol)]
		coords = np.array([[atom.GetX(), atom.GetY(), atom.GetZ()] for atom in openbabel.OBMolAtomIter(ob_mol)])
	
	Z = torch.tensor(atomic_numbers, dtype=torch.long)
	R = torch.tensor(coords, dtype=torch.float32)
	
	return Z, R

def calculate_dynamic_cell_and_pbc(coords: torch.Tensor, buffer_angstroms: float = 10.0):
	"""
	Dynamically calculates a cubic cell matrix and periodic boundary conditions
	based on the molecule's coordinates.

	Args:
		coords (torch.Tensor): Tensor of 3D coordinates (N_atoms, 3).
		buffer_angstroms (float): Additional space to add around the molecule
								  on each side to define the cell size.

	Returns:
		(torch.Tensor, torch.Tensor): A tuple containing the cell matrix and pbc tensor.
	"""
	min_coords = coords.min(dim=0).values
	max_coords = coords.max(dim=0).values
	
	# Determine the largest dimension of the molecule
	molecule_extent = max_coords - min_coords
	max_dimension = molecule_extent.max()
	
	# Add buffer to ensure isolation in the periodic cell
	# 2*buffer because it's on each side
	cell_side_length = max_dimension + 2 * buffer_angstroms 
	
	# Create a cubic cell matrix
	cell = torch.diag(torch.tensor([cell_side_length, cell_side_length, cell_side_length], dtype=torch.float32))
	
	# For an isolated molecule, PBC should be True in all directions
	pbc = torch.tensor([True, True, True])
	
	return cell, pbc

def main():
	"""
	Main function to run the example.
	"""
	model_path = 'model.script'
	
	try:
		print(f"Loading TorchScript model from: {model_path}")
		model = torch.jit.load(model_path)
		print("-> Successfully loaded model.")
	except Exception as e:
		print(f"Error: Could not load '{model_path}'.", file=sys.stderr)
		print(f"Details: {e}", file=sys.stderr)
		print("\nPlease ensure that a valid 'model.script' file exists in the current directory.", file=sys.stderr)
		sys.exit(1)

	# Generate a random molecule with RDKit
	# For a simple and runnable example, we pick a random SMILES from a predefined list.
	# Generating chemically valid random molecules is a complex topic.
	smiles_list = [
		"CC1C2C3C1C(C)(C)N23",
	]
	random_smiles = random.choice(smiles_list)
	print(f"\nUsing RDKit to create molecule from SMILES: '{random_smiles}'")
	
	rdkit_mol = Chem.MolFromSmiles(random_smiles)
	if rdkit_mol is None:
		print(f"Error: RDKit could not parse the SMILES string '{random_smiles}'.", file=sys.stderr)
		sys.exit(1)

	use_tblite_optimization = True
	if use_tblite_optimization:
		print("Attempting geometry optimization with TBLite...")
	else:
		print("Using geometry optimized with Open Babel MMFF94 (TBLite optimization skipped).")

	print("Generating 3D coordinates with Open Babel...")
	try:
		Z, R = get_molecule_data_from_rdkit_mol(rdkit_mol, optimize_with_tblite=use_tblite_optimization)
		print("-> Generated molecule data:")
		print(f"  - Atomic numbers (Z): {Z.tolist()}")
		print(f"  - Coordinates (R) shape: {R.shape}")
	except ImportError as e:
		print(f"Error: {e}", file=sys.stderr)
		print("\nExiting because TBLite optimization was requested but packages are missing.", file=sys.stderr)
		sys.exit(1)
	except Exception as e:
		print(f"Error: Failed to generate 3D coordinates.", file=sys.stderr)
		print(f"Details: {e}", file=sys.stderr)
		print("\nPlease ensure that RDKit and Open Babel are installed correctly.", file=sys.stderr)
		sys.exit(1)
		
	print("\nDynamically defining periodic cell based on molecule extent...")
	# For a single molecule in a box, it's good practice to center it
	R = R - R.mean(dim=0) # Center the molecule
	
	cell, pbc = calculate_dynamic_cell_and_pbc(R, buffer_angstroms=5.0) # Use a 5.0 A buffer
	
	print(f"  - Dynamically calculated cell matrix:\n{cell.numpy()}")
	print(f"  - PBC enabled: {pbc.tolist()}")

	# Run inference with the model
	print("\nRunning inference with the loaded model on the periodic system...")
	try:
		# Set model to evaluation mode
		model.eval()

		with torch.no_grad():
			predictions = model(Z, R, cell=cell, pbc=pbc)
		
		print("\n--- Model Predictions ---")
		for key, value in predictions.items():
			print(f"  - {key}: {value.numpy().squeeze()}")
		print("-------------------------")
			
	except Exception as e:
		print(f"Error: Model inference failed.", file=sys.stderr)
		print(f"Details: {e}", file=sys.stderr)
		sys.exit(1)

	print("\nExample script finished successfully.")

if __name__ == "__main__":
	main()
