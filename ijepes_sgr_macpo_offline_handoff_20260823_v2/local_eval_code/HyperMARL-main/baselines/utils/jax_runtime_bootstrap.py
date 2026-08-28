"""Shared JAX/XLA runtime tuning for training scripts."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TMP = PROJECT_ROOT / ".tmp"
WANDB_LOCAL_DIR = PROJECT_ROOT / "wandb"
JAX_COMPILATION_CACHE_DIR = Path(
    os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        str(PROJECT_ROOT / ".jax_compilation_cache_stable"),
    )
)

for _path in (
    LOCAL_TMP,
    WANDB_LOCAL_DIR,
    WANDB_LOCAL_DIR / "cache",
    WANDB_LOCAL_DIR / "data",
    JAX_COMPILATION_CACHE_DIR,
):
    _path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TMP", str(LOCAL_TMP))
os.environ.setdefault("TEMP", str(LOCAL_TMP))
os.environ.setdefault("TMPDIR", str(LOCAL_TMP))
os.environ.setdefault("WANDB_DIR", str(WANDB_LOCAL_DIR))
os.environ.setdefault("WANDB_CACHE_DIR", str(WANDB_LOCAL_DIR / "cache"))
os.environ.setdefault("WANDB_DATA_DIR", str(WANDB_LOCAL_DIR / "data"))
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(JAX_COMPILATION_CACHE_DIR))
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "true")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import jax  # noqa: E402

jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])
jax.config.update(
    "jax_persistent_cache_min_compile_time_secs",
    float(os.environ.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")),
)
jax.config.update(
    "jax_persistent_cache_min_entry_size_bytes",
    int(os.environ.get("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")),
)
print(f"JAX persistent compilation cache: {os.environ['JAX_COMPILATION_CACHE_DIR']}")
