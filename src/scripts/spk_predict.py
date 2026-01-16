import argparse
import torch
import numpy as np
import sys


def check_dependencies():
    """Checks for required libraries and provides installation instructions."""
    try:
        from rdkit import Chem  
        from tblite.ase import TBLite
        from ase import Atoms 
    except ImportError as e:
        print(f"Error: Missing dependency -> {e.name}", file=sys.stderr)
        print(
            "\nPlease install the required libraries for structure generation:\n" 
            "pip install rdkit-pypi 'tblite[ase]'",
            file=sys.stderr,
        )
        sys.exit(1)


def generate_optimized_structure(smiles: str) -> "Atoms":
    """
    Generates a 3D structure from a SMILES string, optimizes it with GFN2-xTB
    via the tblite library, and returns an ASE Atoms object.

    Args:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        ase.Atoms: The final, optimized ASE Atoms object.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from ase import Atoms
    from ase.optimize import BFGS
    from tblite.ase import TBLite

    print("Step 1: Generating initial 3D conformer with RDKit...")
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Could not parse SMILES string: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 1
    AllChem.EmbedMolecule(mol, params)

    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        pass
    print("Initial conformer generated.")

    # Convert to ASE Atoms
    positions = mol.GetConformer().GetPositions()
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atoms = Atoms(numbers=atomic_numbers, positions=positions)

    # Optimize with tblite (GFN2-xTB)
    print("\nStep 2: Optimizing geometry with GFN2-xTB (using tblite)...")
    calculator = TBLite(method="GFN2-xTB", verbosity=0)
    atoms.calc = calculator
    optimizer = BFGS(atoms, logfile=None)
    optimizer.run(fmax=0.01)
    print("Geometry optimization finished.")

    return atoms


def main():
    check_dependencies()

    parser = argparse.ArgumentParser(
        description=(
            "Generate a molecular structure from SMILES, optimize it with tblite, and "
            "predict its properties using a portable SchNetPack model."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "model_path",
        help="Path to the portable TorchScript model created by 'spk_general_deploy_model.py'.",
    )
    parser.add_argument(
        "smiles",
        help="SMILES string of the molecule to predict (e.g., 'CCO' for ethanol).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run the prediction on ('cpu' or 'cuda').",
    )
    args = parser.parse_args()

    try:
        model = torch.jit.load(args.model_path, map_location=args.device)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{args.model_path}'", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return

    print(f"\nPortable model loaded from {args.model_path}")

    try:
        atoms = generate_optimized_structure(args.smiles)
    except Exception as e:
        print(f"\nAn error occurred during structure generation: {e}", file=sys.stderr)
        return

    # . Extract basic tensors from the final ASE Atoms object
    Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=args.device)
    R = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=args.device)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float32, device=args.device)
    pbc = torch.tensor(atoms.pbc, dtype=torch.bool, device=args.device)

    # Run prediction
    model.eval()
    with torch.no_grad():
        result = model(Z, R, cell, pbc)

    # Print results
    print("\nPrediction Results:")
    for key, value in result.items():
        value_squeezed = value.squeeze().cpu().numpy()
        print(f"  - {key}: {value_squeezed}")


if __name__ == "__main__":
    main()

# vim: set ft=python ts=4 sw=4 et tw=88:
