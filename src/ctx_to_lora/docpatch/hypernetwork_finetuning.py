from dataclasses import dataclass
from typing import Iterable

from torch import nn


@dataclass
class TrainableParameterSummary:
    trainable: int
    frozen: int

    @property
    def total(self) -> int:
        return self.trainable + self.frozen


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_base_model(modulated_model: nn.Module) -> None:
    if hasattr(modulated_model, "base_model"):
        set_requires_grad(modulated_model.base_model, False)


def freeze_ctx_encoder(modulated_model: nn.Module) -> None:
    if hasattr(modulated_model, "ctx_encoder"):
        set_requires_grad(modulated_model.ctx_encoder, False)


def freeze_hypernetwork(modulated_model: nn.Module) -> None:
    if hasattr(modulated_model, "hypernet"):
        set_requires_grad(modulated_model.hypernet, False)


def unfreeze_hypernetwork(modulated_model: nn.Module) -> None:
    if hasattr(modulated_model, "hypernet"):
        set_requires_grad(modulated_model.hypernet, True)


def freeze_hypernetwork_early_blocks(
    modulated_model: nn.Module,
    num_blocks: int,
) -> None:
    """Freeze early aggregator/perceiver blocks when those names exist."""

    if num_blocks <= 0 or not hasattr(modulated_model, "hypernet"):
        return
    frozen = 0
    for name, param in modulated_model.hypernet.named_parameters():
        if ".layers." not in name:
            continue
        parts = name.split(".layers.", 1)[1].split(".", 1)
        try:
            layer_idx = int(parts[0])
        except ValueError:
            continue
        if layer_idx < num_blocks:
            param.requires_grad = False
            frozen += 1
    if frozen == 0:
        # Fallback: freeze the first N direct children.
        for idx, child in enumerate(modulated_model.hypernet.children()):
            if idx >= num_blocks:
                break
            set_requires_grad(child, False)


def trainable_parameters(modules: Iterable[nn.Module]):
    for module in modules:
        for param in module.parameters():
            if param.requires_grad:
                yield param


def parameter_summary(module: nn.Module) -> TrainableParameterSummary:
    trainable = 0
    frozen = 0
    for param in module.parameters():
        n = param.numel()
        if param.requires_grad:
            trainable += n
        else:
            frozen += n
    return TrainableParameterSummary(trainable=trainable, frozen=frozen)


def prepare_docpatch_finetuning(
    modulated_model: nn.Module,
    train_hypernetwork: bool = True,
    freeze_encoder: bool = True,
    freeze_base: bool = True,
    freeze_early_hyper_blocks: int = 0,
) -> TrainableParameterSummary:
    if freeze_base:
        freeze_base_model(modulated_model)
    if freeze_encoder:
        freeze_ctx_encoder(modulated_model)
    if train_hypernetwork:
        unfreeze_hypernetwork(modulated_model)
    else:
        freeze_hypernetwork(modulated_model)
    freeze_hypernetwork_early_blocks(modulated_model, freeze_early_hyper_blocks)
    return parameter_summary(modulated_model)
