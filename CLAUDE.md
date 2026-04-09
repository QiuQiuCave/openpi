# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenPI is Physical Intelligence's open-source robotics library containing vision-language-action (VLA) models: π₀ (flow matching), π₀-FAST (autoregressive with FAST action tokenizer), and π₀.5. It provides both JAX (primary, Flax NNX) and PyTorch implementations, with support for DROID, ALOHA, LIBERO, and other robot platforms.

## Common Commands

```bash
# Install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Run all tests (CI command)
uv run pytest --strict-markers -m "not manual"

# Run a single test file
uv run pytest src/openpi/models/model_test.py

# Lint and format
ruff check . && ruff format .

# Pre-commit hooks
pre-commit install
pre-commit run --all-files

# Serve a policy (inference server on port 8000)
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<path>

# Train (JAX) - requires XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py <config> --exp-name=<name>

# Train (PyTorch, single GPU)
uv run scripts/train_pytorch.py <config> --exp_name <name>

# Train (PyTorch, multi-GPU)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<n> scripts/train_pytorch.py <config>

# Compute normalization statistics for a dataset
uv run scripts/compute_norm_stats.py --config-name <config>
```

## Architecture

### Core Layers (src/openpi/)

- **models/**: JAX model implementations using Flax NNX. `pi0.py` (flow matching), `pi0_fast.py` (autoregressive), `model.py` (base types: `Observation`, `Actions`, `BaseModel`). Uses `jaxtyping` + `beartype` via `@typecheck` decorator for array shape validation.

- **models_pytorch/**: PyTorch mirror of models. `transformers_replace/` contains patched HuggingFace transformers files (excluded from ruff).

- **policies/**: Wraps models with input/output transforms into `Policy` objects. Robot-specific policies (ALOHA, DROID, LIBERO) handle platform-specific data formats. `policy_config.py` is the factory.

- **training/**: `config.py` defines `TrainConfig` and `DataConfig` as frozen dataclasses with many pre-built configurations. `weight_loaders.py` handles loading from JAX/PyTorch checkpoints. `sharding.py` for FSDP.

- **transforms.py**: Composable data transform pipeline. Key transforms: `Normalize`, `ResizeImages`, `TokenizePrompt`, `RepackTransform`. Transforms compose into `Group` (input + output transforms).

- **shared/**: Utilities — `normalize.py` (NormStats), `download.py` (GCS asset caching to `~/.cache/openpi`), `array_typing.py`, `image_tools.py`.

- **serving/**: WebSocket-based policy server for remote inference.

### Client Library (packages/openpi-client/)

Separate package in `uv` workspace. Provides `BasePolicy` interface and `WebSocketClientPolicy` for interacting with served models.

### Key Patterns

- Configs are frozen dataclasses, often with many pre-built instances in the same file.
- `Protocol`-based interfaces for `Dataset`, `IterableDataset`, `DataLoader`, `DataTransformFn`.
- Assets downloaded from GCS (`gs://openpi-assets`), cached locally. Override with `OPENPI_DATA_HOME` env var.
- LoRA fine-tuning supported via `get_freeze_filter()` on model configs.
- Tests are co-located with source (`*_test.py`). The `manual` pytest marker excludes GPU-requiring tests from CI.

## Code Style

- Python 3.11, line length 120
- Ruff for linting and formatting with isort (force-single-line imports)
- `ruff check . && ruff format .` before submitting
- `third_party/` and `docker/` are excluded from linting
