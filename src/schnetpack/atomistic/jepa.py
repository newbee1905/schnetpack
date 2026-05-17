import torch
import torch.nn as nn
from typing import Dict, Optional, List, Type, Any, Union, Callable
import torch.distributed as dist
import schnetpack.properties as properties
import schnetpack.nn as snn
from schnetpack.model.base import AtomisticModel, NeuralNetworkPotential
from schnetpack.task import ModelOutput, UnsupervisedModelOutput
from schnetpack.atomistic import PairwiseDistances
from schnetpack.representation.painn import PaiNN
import copy
import logging
import torch.nn.functional as F

@torch.jit.ignore
def all_reduce(x: torch.Tensor, op: str) -> torch.Tensor:
    """All-reduce operation for distributed training."""
    if dist.is_available() and dist.is_initialized():
        op_obj = dist.ReduceOp.__dict__[op]
        dist.all_reduce(x, op=op_obj)
        return x
    else:
        return x


def epps_pulley(x, t_min: float = -3.0, t_max: float = 3.0, n_points: int = 10):
    """Epps-Pulley test statistic for Gaussianity."""
    # integration points
    t = torch.linspace(t_min, t_max, n_points, device=x.device)
    # theoretical CF for N(0, 1)
    exp_f = torch.exp(-0.5 * t**2)
    # ECF
    x_t = x.unsqueeze(2) * t  # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)
    ecf = all_reduce(ecf, op="AVG")
    # weighted L2 distance
    err = exp_f * (ecf - exp_f).abs() ** 2
    T = torch.trapz(err, t, dim=1)
    return T


class Projector(nn.Module):
    """MLP projector built from a spec string like '256-512-128'."""
    def __init__(self, mlp_spec):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]  # Store output dimension as attribute

    def forward(self, x):
        return self.net(x)


class BCS(nn.Module):
    """BCS (Batched Characteristic Slicing) loss for SIGReg."""

    def __init__(self, num_slices=256, lmbd=10.0):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0
        self.lmbd = lmbd

    def forward(self, z1, z2):
        with torch.no_grad():
            dev = z1.device
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            proj_shape = (z1.size(1), self.num_slices)
            A = torch.randn(proj_shape, device=dev, generator=g)
            A /= A.norm(p=2, dim=0)
        view1 = z1 @ A
        view2 = z2 @ A

        self.step += 1
        bcs = (epps_pulley(view1).mean() + epps_pulley(view2).mean()) / 2
        invariance_loss = F.mse_loss(z1, z2).mean()
        total_loss = invariance_loss + self.lmbd * bcs
        return {"loss": total_loss, "bcs_loss": bcs, "invariance_loss": invariance_loss}


class PaiNNLatentPredictor(nn.Module):
    def __init__(
        self,
        n_atom_basis: int,
        n_layers: int = 3,
        kan_degree: Optional[int] = None,
        use_glu_variant: bool = False,
        activation: Callable = nn.SiLU(),
    ):
        super().__init__()
        input_dim = 2 * n_atom_basis  # q + ||mu||

        # Initial projection to hidden space
        if kan_degree is not None:
            self.pre_lin = nn.Sequential(
                snn.ShiftedChebyKANLayer(input_dim, n_atom_basis * 2, kan_degree),
                activation,
            )
        else:
            self.pre_lin = snn.Dense(
                input_dim,
                n_atom_basis * 2,
                activation=activation,
                use_glu_variant=use_glu_variant,
            )

        # Residual Denoising Blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(n_layers):
            if kan_degree is not None:
                block = nn.Sequential(
                    snn.ShiftedChebyKANLayer(
                        n_atom_basis * 2, n_atom_basis * 2, kan_degree
                    ),
                    nn.LayerNorm(n_atom_basis * 2),
                    activation,
                    snn.ShiftedChebyKANLayer(
                        n_atom_basis * 2, n_atom_basis * 2, kan_degree
                    ),
                )
            else:
                block = nn.Sequential(
                    snn.Dense(
                        n_atom_basis * 2,
                        n_atom_basis * 2,
                        activation=activation,
                        use_glu_variant=use_glu_variant,
                    ),
                    nn.LayerNorm(n_atom_basis * 2),
                    snn.Dense(n_atom_basis * 2, n_atom_basis * 2, activation=None),
                )
            self.res_blocks.append(block)

    def forward(self, q: torch.Tensor, mu: Optional[torch.Tensor]):
        if mu is None:
            raise ValueError("mu must be provided to PaiNNLatentPredictor")
            
        # Compute Invariant Norms from PaiNN vectors (N, 3, F) -> (N, F)
        mu_norms = torch.norm(mu, dim=1)

        # Concatenate with scalars
        x = torch.cat([q, mu_norms], dim=-1)

        # Process through residual layers
        x = self.pre_lin(x)
        for block in self.res_blocks:
            x = x + block(x)  # Learning the geometric correction

        # Split state into predicted scalar and vector scale
        z_pred, mu_scale = torch.chunk(x, 2, dim=-1)

        # Apply scale to original vectors to get predicted vectors
        # mu: (N, 3, F), mu_scale: (N, F) -> (N, 1, F)
        mu_pred = mu_scale.unsqueeze(1) * mu

        return z_pred, mu_pred


class JepaModelOutput(ModelOutput):
    """ModelOutput that returns 0.0 loss if key is missing in pred."""
    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[torch.nn.Module]] = None,
        target_property: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            loss_fn=loss_fn,
            loss_weight=loss_weight,
            metrics=metrics or {},
            constraints=constraints,
            target_property=target_property,
        )

    def calculate_loss(self, pred, target):
        if self.name not in pred or self.target_property not in target:
            return torch.tensor(
                0.0, device=pred[list(pred.keys())[0]].device, requires_grad=True
            )
        return super().calculate_loss(pred, target)

    def update_metrics(self, pred, target, subset):
        if self.name not in pred or self.target_property not in target:
            return
        super().update_metrics(pred, target, subset)


class JEPAGeometryOutput(ModelOutput):
    def __init__(
        self,
        name: str,
        target_property: str = "R_real",
        loss_weight: float = float(1.0),
        detach_embeddings: bool = False,
        lmbd: float = float(10.0),
        num_slices: int = 256,
        predict_embeddings: bool = True,
        aggregation_mode: str = "sum",
        use_vector_loss: bool = False,
        n_atom_basis: int = 128,
        mlp_spec: Optional[str] = None,
    ):
        loss_fn = BCS(lmbd=lmbd, num_slices=num_slices)
        super().__init__(
            name=name,
            target_property=target_property,
            loss_fn=loss_fn,
            loss_weight=loss_weight,
            metrics={},
        )
        self.detach_embeddings = detach_embeddings
        self.predict_embeddings = predict_embeddings
        self.aggregation_mode = aggregation_mode
        self.use_vector_loss = use_vector_loss
        self.model_outputs = [
            name,
            "z_pred",
            "z_real",
            properties.idx,
            properties.idx_m,
            properties.n_atoms,
        ]
        if self.use_vector_loss:
            self.model_outputs.extend(["mu_pred", "mu_real"])
        if predict_embeddings:
            self.model_outputs.append("z_smiles")
        self._step_count = 0

        # Use LayerNorm for fixed unit-scale normalization
        self.norm = nn.LayerNorm(n_atom_basis)
        self.vnorm = snn.EquivariantLayerNorm(n_atom_basis, add_unit_offset=True)

        if mlp_spec:
            self.projector = Projector(mlp_spec)
            if self.use_vector_loss:
                self.v_projector = Projector(mlp_spec)
        else:
            self.projector = nn.Identity()
            self.v_projector = nn.Identity()

    def _aggregate(self, z, inputs, mode=None):
        mode = mode or self.aggregation_mode
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1
        z_agg = snn.scatter_add(z, idx_m, dim_size=maxm)
        if mode == "avg":
            z_agg = z_agg / inputs[properties.n_atoms].view(-1, 1).to(z_agg.dtype)
        return z_agg

    def calculate_loss(self, pred, target):
        if not self.training:
            return torch.tensor(
                0.0, device=pred[list(pred.keys())[0]].device, requires_grad=True
            )

        if "z_real" not in pred or "z_pred" not in pred:
            return torch.tensor(
                0.0, device=pred[list(pred.keys())[0]].device, requires_grad=True
            )

        z_pred = pred["z_pred"]
        z_real = pred["z_real"]

        # Normalize embeddings
        z_pred = self.norm(z_pred)
        z_real = self.norm(z_real)

        # Aggregate to molecule level
        z_pred_mol = self._aggregate(z_pred, pred)
        z_real_mol = self._aggregate(z_real, pred)

        # Stabilize the summed representation for the JEPA loss
        z_pred_mol = F.layer_norm(z_pred_mol, (z_pred_mol.size(-1),))
        z_real_mol = F.layer_norm(z_real_mol, (z_real_mol.size(-1),))

        # Project
        z_pred_mol = self.projector(z_pred_mol)
        z_real_mol = self.projector(z_real_mol)

        loss_dict = self.loss_fn(z_pred_mol, z_real_mol)

        # Optional vector norm loss
        if self.use_vector_loss and "mu_pred" in pred and "mu_real" in pred:
            mu_pred = pred["mu_pred"]
            mu_real = pred["mu_real"]

            mu_pred = self.vnorm(mu_pred)
            mu_real = self.vnorm(mu_real)

            # Compute Invariant Norms (N, 3, F) -> (N, F)
            mu_pred_n = torch.sqrt(torch.sum(mu_pred**2, dim=1) + 1e-8)
            mu_real_n = torch.sqrt(torch.sum(mu_real**2, dim=1) + 1e-8)

            # Aggregate to molecule level
            mu_pred_mol = self._aggregate(mu_pred_n, pred)
            mu_real_mol = self._aggregate(mu_real_n, pred)

            # Stabilize
            mu_pred_mol = F.layer_norm(mu_pred_mol, (mu_pred_mol.size(-1),))
            mu_real_mol = F.layer_norm(mu_real_mol, (mu_real_mol.size(-1),))

            # Project Norms
            mu_pred_mol = self.v_projector(mu_pred_mol)
            mu_real_mol = self.v_projector(mu_real_mol)

            vloss_dict = self.loss_fn(mu_pred_mol, mu_real_mol)

            for k, v in vloss_dict.items():
                loss_dict[f"v_{k}"] = v
            loss_dict["loss"] = loss_dict["loss"] + vloss_dict["loss"]

        for k, v in loss_dict.items():
            pred[f"{self.name}_{k}"] = v.detach()

        loss = self.loss_weight * loss_dict["loss"]
        return loss

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self._step_count += 1
        z_pred = inputs["z_pred"]
        inputs[self.name] = z_pred

        # If not training, energy prediction should use predicted ATOM-WISE embeddings
        if not self.training and self.predict_embeddings:
            inputs["scalar_representation"] = z_pred
            mu_pred = inputs.get("mu_pred")
            if mu_pred is not None:
                inputs["vector_representation"] = mu_pred

        return inputs


class JEPATask(NeuralNetworkPotential):
    def __init__(
        self,
        representation: nn.Module,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,
        postprocessors: Optional[List[Any]] = None,
        input_dtype_str: str = "float32",
        do_postprocessing: bool = True,
        aggregation_mode: str = "sum",
        train_on_pred: bool = False,
        detach_scalar_representation: bool = False,
        predictor: Optional[nn.Module] = None,
    ):
        super().__init__(
            representation=representation,
            input_modules=input_modules,
            output_modules=output_modules,
            postprocessors=postprocessors,
            input_dtype_str=input_dtype_str,
            do_postprocessing=do_postprocessing,
        )
        self.aggregation_mode = aggregation_mode
        self.train_on_pred = train_on_pred
        self.detach_scalar_representation = detach_scalar_representation
        self.distances = None
        if self.input_modules:
            for m in self.input_modules:
                if isinstance(m, PairwiseDistances):
                    self.distances = m
                    break
        if self.distances is None:
            self.distances = PairwiseDistances()

        self.predictor = predictor
        self.collect_outputs()
        self._step_count = 0

    @torch.jit.ignore
    def _log_validation_outputs(
        self,
        inputs: Dict[str, torch.Tensor],
        real_results: Dict[str, torch.Tensor],
        z_pred: torch.Tensor,
        mu_pred: Optional[torch.Tensor],
        z_real: torch.Tensor,
        R_orig: torch.Tensor,
        idx_i_orig: torch.Tensor,
        idx_j_orig: torch.Tensor,
        offsets_orig: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Save REAL outputs
        # To avoid confusion, we first run output modules on REAL representation
        # and store them with _real suffix
        inputs[properties.R] = inputs["R_real"]
        inputs[properties.idx_i] = inputs["_idx_i_real"]
        inputs[properties.idx_j] = inputs["_idx_j_real"]
        inputs[properties.offsets] = inputs["_offsets_real"]
        inputs = self.distances(inputs)

        inputs["scalar_representation"] = z_real
        mu_real = real_results.get("vector_representation")
        if mu_real is not None:
            inputs["vector_representation"] = mu_real

        for m in self.output_modules:
            inputs = m(inputs)

        for m in self.output_modules:
            if hasattr(m, "output_key"):
                inputs[m.output_key + "_real"] = inputs[m.output_key]
            elif hasattr(m, "name") and not isinstance(m, JEPAGeometryOutput):
                inputs[m.name + "_real"] = inputs[m.name]

        # SMILES Energy (Predicted)
        # Restore original geometry and use PREDICTED embeddings
        inputs[properties.R] = R_orig
        inputs[properties.idx_i] = idx_i_orig
        inputs[properties.idx_j] = idx_j_orig
        inputs[properties.offsets] = offsets_orig
        inputs = self.distances(inputs)

        inputs["scalar_representation"] = z_pred
        if mu_pred is not None:
            inputs["vector_representation"] = mu_pred

        for m in self.output_modules:
            inputs = m(inputs)

        # Save SMILES outputs
        for m in self.output_modules:
            if hasattr(m, "output_key"):
                inputs[m.output_key + "_smiles"] = inputs[m.output_key]
            elif hasattr(m, "name") and not isinstance(m, JEPAGeometryOutput):
                inputs[m.name + "_smiles"] = inputs[m.name]

        return inputs

    @torch.jit.ignore
    def _run_output_modules(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for m in self.output_modules:
            inputs = m(inputs)
        return inputs

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.training:
            self._step_count += 1

        inputs = self.initialize_derivatives(inputs)

        for m in self.input_modules:
            inputs = m(inputs)

        # Store original geometry for restoration
        R_orig = inputs[properties.R]
        idx_i_orig = inputs[properties.idx_i]
        idx_j_orig = inputs[properties.idx_j]
        offsets_orig = inputs[properties.offsets]

        # 1. SMILES pass
        inputs = self.distances(inputs)
        inputs = self.representation(inputs)
        z_smiles = inputs["scalar_representation"]
        inputs["z_smiles"] = z_smiles
        mu_smiles = inputs.get("vector_representation")
        if mu_smiles is not None:
            inputs["mu_smiles"] = mu_smiles
        else:
            mu_smiles = None

        # 2. Prediction pass
        if self.predictor is not None:
            z_pred, mu_pred = self.predictor(z_smiles, mu_smiles)
        else:
            if self.detach_scalar_representation:
                z_pred = z_smiles.detach()
                mu_pred = mu_smiles.detach() if mu_smiles is not None else None
            else:
                z_pred = z_smiles
                mu_pred = mu_smiles

        inputs["z_pred"] = z_pred
        if mu_pred is not None:
            inputs["mu_pred"] = mu_pred

        # 3. REAL pass (if present)
        r_real = inputs.get("R_real")
        idx_i_real = inputs.get("_idx_i_real")
        if r_real is not None and idx_i_real is not None:
            # Create a manual copy for the REAL representation pass to avoid polluting core geometry keys.
            # copy.copy is not supported in TorchScript.
            real_inputs: Dict[str, torch.Tensor] = {}
            for k, v in inputs.items():
                real_inputs[k] = v

            real_inputs[properties.R] = r_real
            real_inputs[properties.idx_i] = idx_i_real
            idx_j_real = inputs.get("_idx_j_real")
            if idx_j_real is not None:
                real_inputs[properties.idx_j] = idx_j_real
            offsets_real = inputs.get("_offsets_real")
            if offsets_real is not None:
                real_inputs[properties.offsets] = offsets_real

            real_inputs = self.distances(real_inputs)
            real_results = self.representation(real_inputs)
            z_real = real_results["scalar_representation"]

            if torch.isnan(z_real).any():
                # print is supported in TorchScript, logging.warning is not.
                print("Step ", self._step_count, ": NaN in z_real. Falling back.")
                z_real = z_smiles.detach()

            inputs["z_real"] = z_real
            mu_real = real_results.get("vector_representation")
            if mu_real is not None:
                inputs["mu_real"] = mu_real

            if self.training:
                # Restore original geometry before output modules
                inputs[properties.R] = R_orig
                inputs[properties.idx_i] = idx_i_orig
                inputs[properties.idx_j] = idx_j_orig
                inputs[properties.offsets] = offsets_orig
                inputs = self.distances(inputs)

                # TRAINING: Always use PREDICTED embeddings (from SMILES) to predict energy
                inputs["scalar_representation"] = z_pred
                if mu_pred is not None:
                    inputs["vector_representation"] = mu_pred

                inputs = self._run_output_modules(inputs)
            else:
                # VALIDATION: Calculate for both to check alignment
                inputs = self._log_validation_outputs(
                    inputs,
                    real_results,
                    z_pred,
                    mu_pred,
                    z_real,
                    R_orig,
                    idx_i_orig,
                    idx_j_orig,
                    offsets_orig,
                )
        else:
            # Fallback for inference or missing R_real
            inputs["scalar_representation"] = z_pred
            if mu_pred is not None:
                inputs["vector_representation"] = mu_pred
            
            inputs = self._run_output_modules(inputs)

        inputs = self.postprocess(inputs)
        results = self.extract_outputs(inputs)
        return results


class JepaLatentOutput(UnsupervisedModelOutput):
    def __init__(
        self,
        name: str,
        latent_key: str = "latent",
        loss_weight: float = float(1.0),
        lmbd: float = float(10.0),
        num_slices: int = 256,
        n_latents: int = 128,
        mlp_spec: Optional[str] = None,
    ):
        loss_fn = BCS(lmbd=lmbd, num_slices=num_slices)
        super().__init__(
            name=name,
            loss_fn=loss_fn,
            loss_weight=loss_weight,
            metrics={},
        )
        self.latent_key = latent_key
        self.model_outputs = [
            name,
            f"{latent_key}_pred_projected",
            f"{latent_key}_real",
        ]
        self._step_count = 0

        # Use LayerNorm for fixed unit-scale normalization
        self.norm = nn.LayerNorm(n_latents)

        if mlp_spec:
            self.projector = Projector(mlp_spec)
        else:
            self.projector = nn.Identity()

    def calculate_loss(self, pred, target):
        if not self.training:
            return torch.tensor(
                0.0, device=pred[list(pred.keys())[0]].device, requires_grad=True
            )

        key_pred = f"{self.latent_key}_pred_projected"
        key_real = f"{self.latent_key}_real"

        if key_real not in pred or key_pred not in pred:
            return torch.tensor(
                0.0, device=pred[list(pred.keys())[0]].device, requires_grad=True
            )

        z_pred = pred[key_pred]
        z_real = pred[key_real]

        # Normalize latents
        z_pred = self.norm(z_pred)
        z_real = self.norm(z_real)

        # Project
        z_pred = self.projector(z_pred)
        z_real = self.projector(z_real)

        loss_dict = self.loss_fn(z_pred, z_real)

        for k, v in loss_dict.items():
            pred[f"{self.name}_{k}"] = v.detach()

        loss = self.loss_weight * loss_dict["loss"]
        return loss

    def forward(self, inputs):
        self._step_count += 1
        inputs[self.name] = inputs.get(f"{self.latent_key}_pred_projected", None)
        return inputs
