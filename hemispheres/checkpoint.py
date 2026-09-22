"""Run directories and checkpoints.

  <run>/config.json                  training config, including the model config and arm
  <run>/metrics.jsonl                one line per log/eval event
  <run>/checkpoints/<tag>/model.safetensors
  <run>/checkpoints/<tag>/optimizer.safetensors
  <run>/checkpoints/<tag>/state.pkl  step and data-sampler RNG state, for resuming
  <run>/evals/                       evaluation outputs
"""

import json
import pickle
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .latent import LatentConfig, LatentGPT
from .model import GPT, GPTConfig

MODEL_TYPES = {"gpt": (GPT, GPTConfig), "latent": (LatentGPT, LatentConfig)}


def save(run: Path, tag: str, model: GPT, optimizer=None, state: dict | None = None) -> Path:
    d = run / "checkpoints" / tag
    d.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(d / "model.safetensors"))
    if optimizer is not None:
        mx.save_safetensors(str(d / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    if state is not None:
        (d / "state.pkl").write_bytes(pickle.dumps(state))
    return d


def build_model(model_type: str, cfg: dict):
    cls, cfg_cls = MODEL_TYPES[model_type]
    return cls(cfg_cls(**cfg))


def load_model(run: str | Path, tag: str = "final") -> tuple:
    """The model saved under `tag`, and the run's config."""
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    model = build_model(config.get("model_type", "gpt"), config["model"])
    model.load_weights(str(run / "checkpoints" / tag / "model.safetensors"))
    model.set_dtype(getattr(mx, config.get("dtype", "float32")))
    mx.eval(model.parameters())
    return model, config


def load_optimizer_state(run: Path, tag: str, optimizer) -> dict:
    d = run / "checkpoints" / tag
    optimizer.state = tree_unflatten(list(mx.load(str(d / "optimizer.safetensors")).items()))
    return pickle.loads((d / "state.pkl").read_bytes())
