#!/usr/bin/env python

"""
This script provides an example of how to use a deployed SchNetPack model
in TorchScript format ('model.script') and validates its predictions against
the original LMDB database.

It performs the following steps:
1.  Loads the TorchScript model.
2.  Loads the QM9 SMILES dataset from the LMDB database.
3.  Picks a random molecule from the database.
4.  Prepares the atomic inputs (atomic numbers, coordinates, cell, pbc) as PyTorch tensors.
5.  Runs inference with the loaded model.
6.  Compares the model's prediction with the ground truth value from the database.
"""

import torch
import numpy as np
import random
import sys
import os
import logging

# SchNetPack imports
from schnetpack.data import LMDBAtomsData
import schnetpack.properties as properties


def main():
	"""
	Main function to run the example.
	"""
	model_path = "model.script"
	datapath = "data/qm9_smiles.lmdb"

	try:
		print(f"Loading TorchScript model from: {model_path}")
		model = torch.jit.load(model_path)
		print("-> Successfully loaded model.")
	except Exception as e:
		print(f"Error: Could not load '{model_path}'.", file=sys.stderr)
		print(f"Details: {e}", file=sys.stderr)
		sys.exit(1)

	if not os.path.exists(datapath):
		print(f"Error: Dataset not found at {datapath}")
		sys.exit(1)

	print(f"Loading dataset from: {datapath}")
	property_units = {
		"energy_U0": "eV",
		"energy_U": "eV",
		"enthalpy_H": "eV",
		"free_energy": "eV",
		"homo": "eV",
		"lumo": "eV",
		"gap": "eV",
		"zpve": "eV",
	}
	dataset = LMDBAtomsData(datapath, property_units=property_units)
	num_entries = len(dataset)
	print(f"-> Dataset contains {num_entries} entries.")

	# Pick a random molecule
	idx = random.randint(0, num_entries - 1)
	print(f"\nRetrieving molecule at index: {idx}")
	data = dataset[idx]

	# Prepare inputs
	Z = data[properties.Z].long()
	R = data[properties.R].float()
	cell = data[properties.cell].float()
	pbc = data[properties.pbc].bool()

	# Centering R 
	R = R - R.mean(dim=0)

	print("-> Molecule data retrieved:")
	print(f"  - Atomic numbers (Z): {Z.tolist()}")
	print(f"  - Coordinates (R) shape: {R.shape}")
	print(f"  - Cell matrix shape: {cell.shape}")
	print(f"  - PBC: {pbc.tolist()}")

	print("\nRunning inference with the loaded model...")
	try:
		model.eval()
		with torch.no_grad():
			# The portable model takes (Z, R, cell, pbc) as arguments
			predictions = model(Z, R, cell=cell.squeeze(0), pbc=pbc.squeeze(0))

		print("\n--- Comparison (Model vs. Database) ---")
		for key, pred_val in predictions.items():
			pred_val = pred_val.cpu().numpy().squeeze()
			if key in data:
				true_val = data[key].numpy().squeeze()
				diff = pred_val - true_val
				print(f"  - {key}:")
				print(f"    Predicted: {pred_val:.6f}")
				print(f"    Actual:    {true_val:.6f}")
				print(f"    Difference: {diff:.6e}")
			else:
				print(f"  - {key} (Predicted only): {pred_val:.6f}")
		print("---------------------------------------")

	except Exception as e:
		print(f"Error: Model inference failed.", file=sys.stderr)
		import traceback

		traceback.print_exc()
		sys.exit(1)

	print("\nExample script finished successfully.")


if __name__ == "__main__":
	main()
