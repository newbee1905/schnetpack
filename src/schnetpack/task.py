import warnings
from typing import Optional, Dict, List, Type, Any

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torchmetrics import Metric

from schnetpack.model.base import AtomisticModel
import schnetpack.properties as properties

__all__ = ["ModelOutput", "AtomisticTask"]


class ModelOutput(nn.Module):
    """
    Defines an output of a model, including mappings to a loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        constraints: Optional[List[torch.nn.Module]] = None,
        target_property: Optional[str] = None,
    ):
        """
        Args:
            name: name of output in results dict
            target_property: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of metrics with names as keys
            constraints:
                constraint class for specifying the usage of model output in the loss function and logged metrics,
                while not changing the model output itself. Essentially, constraints represent postprocessing transforms
                that do not affect the model output but only change the loss value. For example, constraints can be used
                to neglect or weight some atomic forces in the loss function. This may be useful when training on
                systems, where only some forces are crucial for its dynamics.
        """
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.train_metrics = nn.ModuleDict(metrics)
        self.val_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
        self.test_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }
        self.constraints = constraints or []

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property]
        )
        return loss

    def update_metrics(self, pred, target, subset):
        for metric in self.metrics[subset].values():
            metric(pred[self.name], target[self.target_property])


class UnsupervisedModelOutput(ModelOutput):
    """
    Defines an unsupervised output of a model, i.e. an unsupervised loss or a regularizer
    that do not depend on label data. It includes mappings to the loss function,
    a weight for training and metrics to be logged.
    """

    def calculate_loss(self, pred, target=None):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0
        loss = self.loss_weight * self.loss_fn(pred[self.name])
        return loss

    def update_metrics(self, pred, target, subset):
        for metric in self.metrics[subset].values():
            metric(pred[self.name])


class GroupedEnergyOutput(ModelOutput):
    def __init__(
        self,
        name: str,
        target_property: str,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        loss_fn: Optional[torch.nn.Module] = None,
    ):
        super().__init__(
            name=name,
            target_property=target_property,
            loss_fn=loss_fn or torch.nn.MSELoss(),
            loss_weight=loss_weight,
            metrics=metrics,
        )

    def calculate_loss(self, pred, target):
        if self.name not in pred or self.target_property not in target:
            return torch.tensor(0.0, device=pred[list(pred.keys())[0]].device, requires_grad=True)

        loss = self.loss_fn(pred[self.name], target[self.target_property])
        return self.loss_weight * loss

    def update_metrics(self, pred, target, subset):
        for metric in self.metrics[subset].values():
            if getattr(metric, "is_delta_e", False):
                # Calculate Delta E (Barrier) performance
                # This requires samples from the same reaction to be in the same batch.
                
                # Check for index and state_idx in prediction or target
                idxs = pred.get(properties.idx)
                if idxs is None:
                    idxs = target.get(properties.idx)
                
                state_idxs = target.get("state_idx")
                
                if idxs is None or state_idxs is None:
                    logging.warning("Could not find properties.idx or state_idx in pred or target. Skipping Delta E metric.")
                    continue

                idxs = idxs.squeeze()
                state_idxs = state_idxs.squeeze()
                if idxs.ndim == 0 or state_idxs.ndim == 0:
                    continue
                
                # We need to find the number of states to map back to reaction index
                # In RGD1Single, n_states is len(states)
                # However, since we might not know n_states here, we use the property that 
                # for RGD1, reactions are contiguous in the flattened dataset.
                # Actually, a safer way is to use the fact that state_idx 0 is reactant and 2 is TS
                # if we follow the [reactant, products, ts] or [reactant, ts, products] convention.
                # For RGD1, state_idx 0 is always reactant, 1 product, 2 TS in the 3-state case.
                # If only 2 states [reactants, ts] are used, state_idx 0 is reactant, 1 is ts.
                
                # Find reactants and TS based on their names if available, or assume 0 and max
                # A more general way: reactant is the FIRST state, TS is usually the last or specifically tagged.
                # Given the RGD1Single implementation:
                # 0: reactants, 1: products, 2: ts (default)
                # If only [reactants, ts], then 0: reactants, 1: ts.
                
                # Find reactant (0) and find TS (highest index or specifically 2)
                r_mask = state_idxs == 0
                ts_mask = (state_idxs == 2) | ((state_idxs == 1) & (state_idxs.max() == 1))
                
                if r_mask.any() and ts_mask.any():
                    # Map to reaction index. In RGD1Single, rxn_idx = idx // n_states
                    # We can find n_states from state_idxs.max() + 1
                    n_states = int(state_idxs.max().item()) + 1
                    rxn_idxs = idxs // n_states
                    
                    r_rxns = rxn_idxs[r_mask]
                    ts_rxns = rxn_idxs[ts_mask]
                    
                    # Find common reactions
                    common_rxns = np.intersect1d(r_rxns.cpu().numpy(), ts_rxns.cpu().numpy())
                    
                    if len(common_rxns) > 0:
                        p_delta_e = []
                        t_delta_e = []
                        
                        for rxn in common_rxns:
                            # Use actual masks to find the correct samples in the batch
                            r_match = (rxn_idxs == rxn) & r_mask
                            ts_match = (rxn_idxs == rxn) & ts_mask
                            
                            if r_match.any() and ts_match.any():
                                r_idx_in_batch = torch.where(r_match)[0][0]
                                ts_idx_in_batch = torch.where(ts_match)[0][0]
                                
                                p_delta_e.append(pred[self.name][ts_idx_in_batch] - pred[self.name][r_idx_in_batch])
                                t_delta_e.append(target[self.target_property][ts_idx_in_batch] - target[self.target_property][r_idx_in_batch])
                        
                        if len(p_delta_e) > 0:
                            metric(torch.stack(p_delta_e).detach().cpu(), torch.stack(t_delta_e).detach().cpu())
            else:
                metric(
                    pred[self.name].detach().cpu(),
                    target[self.target_property].detach().cpu(),
                )


class AtomisticTask(pl.LightningModule):
    """
    The basic learning task in SchNetPack, which ties model, loss and optimizer together.

    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs: List[ModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        """
        Args:
            model: the neural network model
            outputs: list of outputs an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            warmup_steps: number of steps used to increase the learning rate from zero
              linearly to the target learning rate at the beginning of training
        """
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.lr = optimizer_args["lr"]
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.model.initialize_transforms(self.trainer.datamodule)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        results = self.model(inputs)
        return results

    def loss_fn(self, pred, batch):
        loss = 0.0
        for output in self.outputs:
            loss += output.calculate_loss(pred, batch)
        return loss

    def log_metrics(self, pred, targets, subset):
        for output in self.outputs:
            output.update_metrics(pred, targets, subset)
            for metric_name, metric in output.metrics[subset].items():
                self.log(
                    f"{subset}_{output.name}_{metric_name}",
                    metric,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                )

    def apply_constraints(self, pred, targets):
        for output in self.outputs:
            for constraint in output.constraints:
                pred, targets = constraint(pred, targets, output)
        return pred, targets

    def training_step(self, batch, batch_idx):

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        for k in [properties.idx, properties.idx_m, properties.n_atoms]:
            if k in batch:
                targets[k] = batch[k]
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log_metrics(pred, targets, "train")

        # Log extra scalar losses from pred
        for k, v in pred.items():
            if k.endswith("_loss") and isinstance(v, torch.Tensor) and v.ndim == 0:
                self.log(f"train_{k}", v, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        for k in [properties.idx, properties.idx_m, properties.n_atoms]:
            if k in batch:
                targets[k] = batch[k]
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["_idx"]),
        )
        self.log_metrics(pred, targets, "val")

        # Log extra scalar losses from pred
        for k, v in pred.items():
            if k.endswith("_loss") and isinstance(v, torch.Tensor) and v.ndim == 0:
                self.log(f"val_{k}", v, on_step=False, on_epoch=True, prog_bar=False)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        for k in [properties.idx, properties.idx_m, properties.n_atoms]:
            if k in batch:
                targets[k] = batch[k]
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["_idx"]),
        )
        self.log_metrics(pred, targets, "test")

        # Log extra scalar losses from pred
        for k, v in pred.items():
            if k.endswith("_loss") and isinstance(v, torch.Tensor) and v.ndim == 0:
                self.log(f"test_{k}", v, on_step=False, on_epoch=True, prog_bar=False)

        return {"test_loss": loss}

    def predict_without_postprocessing(self, batch):
        pp = self.model.do_postprocessing
        self.model.do_postprocessing = False
        pred = self(batch)
        self.model.do_postprocessing = pp
        return pred

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedulers = []
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)
            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            # incase model is validated before epoch end (not recommended use of val_check_interval)
            if self.trainer.val_check_interval < 1.0:
                warnings.warn(
                    "Learning rate is scheduled after epoch end. To enable scheduling before epoch end, "
                    "please specify val_check_interval by the number of training epochs after which the "
                    "model is validated."
                )
            # incase model is validated before epoch end (recommended use of val_check_interval)
            if self.trainer.val_check_interval > 1.0:
                optimconf["interval"] = "step"
                optimconf["frequency"] = self.trainer.val_check_interval
            schedulers.append(optimconf)
            return [optimizer], schedulers
        else:
            return optimizer

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_closure=None,
    ):
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def save_model(self, path: str, do_postprocessing: Optional[bool] = None):
        if self.global_rank == 0:
            pp_status = self.model.do_postprocessing
            if do_postprocessing is not None:
                self.model.do_postprocessing = do_postprocessing
            torch.save(self.model, path)
            self.model.do_postprocessing = pp_status


class ConsiderOnlySelectedAtoms(nn.Module):
    """
    Constraint that allows to neglect some atomic targets (e.g. forces of some specified atoms) for model optimization,
    while not affecting the actual model output. The indices of the atoms, which targets to consider in the loss
    function, must be provided in the dataset for each sample in form of a torch tensor of type boolean
    (True: considered, False: neglected).
    """

    def __init__(self, selection_name):
        """
        Args:
            selection_name: string associated with the list of considered atoms in the dataset
        """
        super().__init__()
        self.selection_name = selection_name

    def forward(self, pred, targets, output_module):
        """
        A torch tensor is loaded from the dataset, which specifies the considered atoms. Only the
        predictions of those atoms are considered for training, validation, and testing.

        :param pred: python dictionary containing model outputs
        :param targets: python dictionary containing targets
        :param output_module: torch.nn.Module class of a particular property (e.g. forces)
        :return: model outputs and targets of considered atoms only
        """

        considered_atoms = targets[self.selection_name].nonzero()[:, 0]

        # drop neglected atoms
        pred[output_module.name] = pred[output_module.name][considered_atoms]
        targets[output_module.target_property] = targets[output_module.target_property][
            considered_atoms
        ]

        return pred, targets
