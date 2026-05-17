#!/usr/bin/env python

"""
Predicts the Gibbs free energy of a molecule from its SMILES string
using the SchNetPack TorchScript model.
Uses OpenBabel for 3D geometry generation and optimization.
"""

import torch
import sys
import argparse
from openbabel import openbabel
import numpy as np


def smiles_to_tensors(smiles):
	"""
	Converts a SMILES string to the tensors required by the model using OpenBabel.
	"""
	# Setup OpenBabel conversion
	ob_conversion = openbabel.OBConversion()
	ob_conversion.SetInFormat("can")  # Canonical SMILES
	ob_mol = openbabel.OBMol()

	if not ob_conversion.ReadString(ob_mol, smiles):
		raise ValueError(f"Invalid SMILES string: {smiles}")

	# Add Hydrogens and generate 3D coordinates
	ob_mol.AddHydrogens()
	builder = openbabel.OBBuilder()
	builder.Build(ob_mol)

	# Optimize with MMFF94
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

	# Extract atomic numbers and positions
	num_atoms = ob_mol.NumAtoms()
	atomic_numbers = []
	positions = []

	for i in range(1, num_atoms + 1):
		atom = ob_mol.GetAtom(i)
		atomic_numbers.append(atom.GetAtomicNum())
		positions.append([atom.GetX(), atom.GetY(), atom.GetZ()])

	Z = torch.tensor(atomic_numbers, dtype=torch.long)
	R = torch.tensor(positions, dtype=torch.float32)

	# Centering positions
	R = R - R.mean(dim=0)

	# Cell and PBC (not used for isolated molecules like QM9)
	cell = torch.eye(3, dtype=torch.float32)
	pbc = torch.zeros(3, dtype=torch.bool)

	return Z, R, cell, pbc


def main():
	parser = argparse.ArgumentParser(description="Predict Gibbs free energy from SMILES using OpenBabel.")
	parser.add_argument("smiles", help="SMILES string of the molecule.")
	parser.add_argument("--model", default="model.script", help="Path to the TorchScript model.")
	args = parser.parse_args()

	try:
		model = torch.jit.load(args.model)
		model.eval()
	except Exception as e:
		print(f"Error: Could not load model '{args.model}': {e}")
		sys.exit(1)

	# Convert SMILES to input tensors
	try:
		Z, R, cell, pbc = smiles_to_tensors(args.smiles)
	except Exception as e:
		print(f"Error: Could not process SMILES '{args.smiles}': {e}")
		sys.exit(1)

	with torch.no_grad():
		try:
			# The PortableModel takes (Z, R, cell, pbc)
			outputs = model(Z, R, cell=cell, pbc=pbc)

			print(f"\nSMILES: {args.smiles}")
			print("-" * (len(args.smiles) + 8))
			for key, val in outputs.items():
				print(f"{key}: {val.item():.6f} eV")
			print("-" * (len(args.smiles) + 8))

		except Exception as e:
			print(f"Error during inference: {e}")
			sys.exit(1)


if __name__ == "__main__":
	main()
