import torch
import hydra
from omegaconf import DictConfig
import logging

# SchNetPack imports
from schnetpack.utils import load_model
from schnetpack.transform import MatScipyNeighborList, ASENeighborList
import schnetpack.properties as properties

log = logging.getLogger(__name__)


class PortableModel(torch.nn.Module):
    """
    A wrapper that bundles preprocessing transforms and a trained model into a single
    scriptable module.

    Args:
        transforms (list): A list of SchNetPack transform objects.
        model (schnetpack.model.AtomisticModel): The trained SchNetPack model.
    """

    def __init__(self, transforms: list, model: torch.nn.Module):
        super().__init__()

        # Sanity check for non-scriptable transforms that rely on external libraries
        unscriptable = []
        for t in transforms:
            if isinstance(t, (MatScipyNeighborList, ASENeighborList)):
                unscriptable.append(type(t).__name__)
        if unscriptable:
            raise TypeError(
                f"The following transforms in your config are not scriptable: {', '.join(unscriptable)}. "
                f"They rely on external libraries (SciPy/ASE) for neighbor list computation. "
                f"To create a portable model, please use `schnetpack.transform.TorchNeighborList`."
            )

        self.preprocessing = torch.nn.Sequential(*transforms)
        self.model = model

    def forward(
        self,
        Z: torch.Tensor,
        R: torch.Tensor,
        cell: torch.Tensor = None,
        pbc: torch.Tensor = None,
    ):
        """
        Takes basic atomic data as tensors and returns model predictions.

        Args:
            Z (torch.Tensor): Atomic numbers, shape (n_atoms,).
            R (torch.Tensor): Atomic positions, shape (n_atoms, 3).
            cell (torch.Tensor, optional): Simulation cell, shape (3, 3).
            pbc (torch.Tensor, optional): Periodic boundary conditions, shape (3,).

        Returns:
            dict[str, torch.Tensor]: A dictionary of predicted properties.
        """

        # Create the initial input dictionary required by SchNetPack transforms,
        # replicating what AtomsConverter + _atoms_collate_fn do for a single molecule.
        inputs = {
            properties.Z: Z,
            properties.R: R,
            properties.n_atoms: torch.tensor(
                [Z.shape[0]], device=Z.device, dtype=torch.long
            ),
            properties.idx: torch.tensor([0], device=Z.device, dtype=torch.long),
            properties.idx_m: torch.zeros(
                Z.shape[0], dtype=torch.long, device=Z.device
            ),
        }
        if cell is not None:
            inputs[properties.cell] = cell.view(1, 3, 3)
        if pbc is not None:
            inputs[properties.pbc] = pbc.view(1, 3)

        # Apply all preprocessing transforms
        processed_inputs = self.preprocessing(inputs)

        return self.model(processed_inputs)


@hydra.main(config_path="../schnetpack/configs", config_name="deploy", version_base=None)
def main(cfg: DictConfig):
    """
    Create a portable TorchScript model by bundling a trained SchNetPack model
    with its preprocessing transforms, configured via Hydra.
    """

    log.info(f"Loading model from: {cfg.model_path}")
    try:
        model = load_model(cfg.model_path)
    except Exception as e:
        log.error(f"Error loading model file: {e}")
        return

    log.info("Instantiating preprocessing transforms from configuration...")
    try:
        if "data" not in cfg or "transforms" not in cfg.data:
            log.error("Could not find 'data.transforms' in the provided configuration.")
            return
        transforms = [hydra.utils.instantiate(tconf) for tconf in cfg.data.transforms]
    except Exception as e:
        log.error(f"Error instantiating transforms from config: {e}")
        return

    log.info("Wrapping model and preprocessing transforms...")
    try:
        portable_model = PortableModel(transforms, model)
        scripted_model = torch.jit.script(portable_model)
    except Exception as e:
        log.error(
            f"Could not create a portable model. This is likely due to an "
            f"unscriptable operation in your model or transforms.\nDetails: {e}"
        )
        return

    torch.jit.save(scripted_model, cfg.output_path)
    log.info(f"\nSuccess! Portable model saved to: {cfg.output_path}")


if __name__ == "__main__":
    main()

# vim: set ft=python ts=4 sw=4 et tw=88:
