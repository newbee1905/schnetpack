import os
import pickle
import lmdb
import tqdm
from schnetpack.data import LMDBAtomsData
import schnetpack.properties as structure


def extract_smiles_dataset(source_path, target_path):
	if not os.path.exists(source_path):
		print(f"Source {source_path} not found.")
		return

	if os.path.exists(target_path):
		print(f"Target {target_path} already exists. Skipping.")
		return

	# Open source
	src_env = lmdb.open(
		source_path,
		subdir=False,
		readonly=True,
		lock=False,
		map_size=10995111627776 * 2,
	)

	with src_env.begin() as txn:
		metadata = pickle.loads(txn.get(b"_metadata"))
		length = pickle.loads(txn.get(b"__len__"))

	# Create target
	# Strip R_real and related from metadata if they exist
	if "_property_unit_dict" in metadata:
		metadata["_property_unit_dict"] = {
			k: v
			for k, v in metadata["_property_unit_dict"].items()
			if k not in ["R_real", "_idx_i_real", "_idx_j_real", "_offsets_real"]
		}

	dst_env = lmdb.open(
		target_path,
		subdir=False,
		map_size=10995111627776 * 2,
		readonly=False,
		lock=False,
	)

	with dst_env.begin(write=True) as txn:
		txn.put(b"_metadata", pickle.dumps(metadata))
		txn.put(b"__len__", pickle.dumps(0))

	print(f"Extracting {length} molecules...")
	with src_env.begin() as src_txn:
		for i in tqdm.tqdm(range(length)):
			key = str(i).encode("ascii")
			data = pickle.loads(src_txn.get(key))

			# Data in LMDB already contains R as the SMILES geometry
			# if it was generated with use_smiles=True

			# Strip unwanted keys
			clean_data = {
				k: v for k, v in data.items() if k not in ["R_real", "_idx_i_real", "_idx_j_real", "_offsets_real"]
			}

			with dst_env.begin(write=True) as dst_txn:
				idx = pickle.loads(dst_txn.get(b"__len__"))
				dst_txn.put(str(idx).encode("ascii"), pickle.dumps(clean_data))
				dst_txn.put(b"__len__", pickle.dumps(idx + 1))

	src_env.close()
	dst_env.close()
	print("Done.")


if __name__ == "__main__":
	extract_smiles_dataset("data/qm9_both.lmdb", "data/qm9_smiles.lmdb")
