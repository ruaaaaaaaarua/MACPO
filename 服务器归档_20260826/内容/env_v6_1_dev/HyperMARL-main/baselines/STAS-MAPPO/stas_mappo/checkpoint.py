"""Complete resumable state for STAS credit assigners."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = {1, CHECKPOINT_SCHEMA_VERSION}
_VERSION_ONE_OPTIONAL_CONFIG_FIELDS = {
    "eval_mask_seed",
    "eval_mask_count",
}
_VERSIONLESS_LEGACY_CONFIG_FIELDS = {
    "causal",
    "eval_mask_seed",
    "eval_mask_count",
}
_DETERMINISTIC_CONFIG_FIELDS = (
    "obs_dim",
    "action_dim",
    "n_agents",
    "seq_length",
    "emb_dim",
    "n_heads",
    "n_layers",
    "sample_num",
    "causal",
    "eval_mask_seed",
    "eval_mask_count",
)
_OPTIONAL_ASSIGNER_STATE_FIELDS = (
    "episodes_seen",
    "last_explained_variance",
    "last_conservation_error",
    "last_mix_coef",
    "last_target_error",
    "last_gate_phase",
    "last_gate_active",
    "last_gate_reason",
    "reward_model_updates",
    "next_reward_model_update_episode",
    "last_reconstruction_rmse",
    "last_agent_credit_variance",
    "last_time_credit_variance",
)


def _validate_checkpoint_compatibility(assigner, state: Dict[str, Any]) -> int:
    versionless_legacy = "version" not in state
    if versionless_legacy:
        version = 0
    else:
        raw_version = state["version"]
        if isinstance(raw_version, bool) or not isinstance(
            raw_version, (int, np.integer)
        ):
            raise ValueError("invalid STAS credit checkpoint schema version")
        version = int(raw_version)
        if version not in _SUPPORTED_SCHEMA_VERSIONS:
            supported = ", ".join(
                str(item) for item in sorted(_SUPPORTED_SCHEMA_VERSIONS)
            )
            raise ValueError(
                f"unsupported STAS credit checkpoint schema version {version}; "
                f"supported versions: {supported}"
            )

    saved_config = state.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("STAS credit checkpoint is missing its configuration")
    for field in _DETERMINISTIC_CONFIG_FIELDS:
        if field not in saved_config:
            if (
                versionless_legacy and field in _VERSIONLESS_LEGACY_CONFIG_FIELDS
            ) or (version == 1 and field in _VERSION_ONE_OPTIONAL_CONFIG_FIELDS):
                continue
            raise ValueError(
                f"STAS credit checkpoint configuration is missing {field!r}"
            )
        expected = getattr(assigner.config, field)
        actual = saved_config[field]
        if actual != expected:
            raise ValueError(
                f"STAS credit checkpoint configuration mismatch for {field!r}: "
                f"checkpoint={actual!r}, assigner={expected!r}"
            )
    return version


def credit_assigner_state_dict(assigner) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "version": CHECKPOINT_SCHEMA_VERSION,
        "config": copy.deepcopy(dict(assigner.config.__dict__)),
        "model_state_dict": copy.deepcopy(assigner.model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(assigner.optimizer.state_dict()),
        "buffer_capacity": int(assigner.buffer.capacity),
        "buffer_storage": copy.deepcopy(list(assigner.buffer.storage)),
        "rollouts_seen": int(assigner.rollouts_seen),
        "last_loss": float(assigner.last_loss),
        "python_random_state": copy.deepcopy(random.getstate()),
        "numpy_random_state": copy.deepcopy(np.random.get_state()),
        "torch_rng_state": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = [
            rng_state.clone() for rng_state in torch.cuda.get_rng_state_all()
        ]
    for name in _OPTIONAL_ASSIGNER_STATE_FIELDS:
        if hasattr(assigner, name):
            state[name] = copy.deepcopy(getattr(assigner, name))
    if hasattr(assigner, "holdout_buffer"):
        state["holdout_buffer_capacity"] = int(assigner.holdout_buffer.capacity)
        state["holdout_buffer_storage"] = copy.deepcopy(
            list(assigner.holdout_buffer.storage)
        )
    if hasattr(assigner, "normalizer"):
        state["normalizer"] = {
            "count": int(assigner.normalizer.count),
            "mean": float(assigner.normalizer.mean),
            "m2": float(assigner.normalizer.m2),
        }
    if hasattr(assigner, "gate"):
        state["gate"] = {
            "negative_streak": int(assigner.gate.negative_streak),
            "disabled": bool(assigner.gate.disabled),
        }
    return state


def load_credit_assigner_state(assigner, state: Dict[str, Any]) -> None:
    _validate_checkpoint_compatibility(assigner, state)
    assigner.model.load_state_dict(state["model_state_dict"])
    assigner.optimizer.load_state_dict(state["optimizer_state_dict"])
    if "buffer_capacity" in state:
        assigner.buffer.capacity = int(state["buffer_capacity"])
    assigner.buffer.storage = copy.deepcopy(list(state.get("buffer_storage", [])))[
        -assigner.buffer.capacity :
    ]
    assigner.rollouts_seen = int(state.get("rollouts_seen", 0))
    assigner.last_loss = float(state.get("last_loss", float("nan")))
    for name in _OPTIONAL_ASSIGNER_STATE_FIELDS:
        if name in state:
            setattr(assigner, name, copy.deepcopy(state[name]))
    if hasattr(assigner, "holdout_buffer"):
        if "holdout_buffer_capacity" in state:
            assigner.holdout_buffer.capacity = int(state["holdout_buffer_capacity"])
        assigner.holdout_buffer.storage = copy.deepcopy(
            list(state.get("holdout_buffer_storage", []))
        )[-assigner.holdout_buffer.capacity :]
    if hasattr(assigner, "normalizer") and "normalizer" in state:
        normalizer = state["normalizer"]
        assigner.normalizer.count = int(normalizer["count"])
        assigner.normalizer.mean = float(normalizer["mean"])
        assigner.normalizer.m2 = float(normalizer["m2"])
    if hasattr(assigner, "gate") and "gate" in state:
        assigner.gate.negative_streak = int(state["gate"]["negative_streak"])
        assigner.gate.disabled = bool(state["gate"]["disabled"])
    if "numpy_random_state" in state:
        np.random.set_state(state["numpy_random_state"])
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "torch_cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(
            [rng_state.cpu() for rng_state in state["torch_cuda_rng_state_all"]]
        )


def save_credit_assigner_checkpoint(
    path,
    assigner,
    *,
    update: int,
    episode: int,
    global_step: int,
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    state = credit_assigner_state_dict(assigner)
    state.update(
        {
            "update": int(update),
            "episode": int(episode),
            "global_step": int(global_step),
        }
    )
    try:
        torch.save(state, temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_credit_assigner_checkpoint(path, assigner) -> Dict[str, int]:
    state = torch.load(
        Path(path).expanduser(),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(state, dict):
        raise ValueError("STAS credit checkpoint payload must be a dictionary")
    try:
        metadata = {
            name: int(state[name]) for name in ("update", "episode", "global_step")
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("STAS credit checkpoint metadata is incomplete") from error
    load_credit_assigner_state(assigner, state)
    return metadata
