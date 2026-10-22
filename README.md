# 圣水回响 | ElixirEcho

*Learning to echo human elixir decisions.*

圣水回响 is a personal research project about learning game decisions from complete
Clash Royale replay trajectories. Its emphasis is the learning pipeline: actor-fair
semantics, episode-safe data, temporal inputs, and imitation-learning integration.

It is not an official Clash Royale project, a production bot, or a fork that bundles
FirstLight. The current policy integration reuses FirstLight V4 components through a
read-only adapter. ElixirEcho owns its domain contracts, observation profiles, cache
boundaries, temporal context, and integration layer; the long-term goal is to use this
foundation to investigate and train independent policies.

## What is here

- Canonical game-agent concepts: `Observation`, `ActionMask`, `DecisionStep`,
  `Trajectory`, `History`, `PublicStateTracker`, `PolicyInput`, and `RecurrentState`.
- Actor-fair minimum and rich observation selections exposed as
  `ObservationProfile.MIN` and `ObservationProfile.RICH`. `CORE` and `FULL` remain
  compatibility aliases, and the serialized values stay `core` and `full`.
- Atomic, hash-verified episode storage and a training-safe reader that does not expose
  episode decks or provider diagnostics.
- Causal public history and deterministic tracking, with unknown facts kept unknown.
- Optional read-only FirstLight V4 adapters for tensorization, policy inference, and
  imitation-learning compatibility checks.
- A read-only command for validating local trajectory caches.

The core package has no runtime dependencies. Replay data, checkpoints, game assets, and
machine-specific FirstLight paths are intentionally not included.

## Quick start

Requirements: Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --extra dev
uv run pytest -q
uv run elixir-echo --help
```

The test suite exercises the local contracts and compatibility boundaries. Tests that
need PyTorch, a FirstLight checkout, or a private replay cache are optional and skip when
those resources are not configured. The FirstLight optimizer smoke is separately gated
by `ELIXIR_ECHO_RUN_TRAINING_SMOKE=1`. CLI training is dry-run by default and starts
only when `train-firstlight` is explicitly passed `--run` with a required episode cap.

## Verify a local trajectory cache

Point the command at an ElixirEcho trajectory-store root (the directory containing
`episodes/`). It verifies committed episode and record hashes, ordering, identity, and
full-episode structure, then prints a summary through the training-safe reader.

```powershell
uv run elixir-echo verify-trajectory "D:\path\to\trajectory-cache"
```

The report intentionally excludes episode context, complete decks, provider diagnostics,
and other audit-only metadata.

## Plan or run a capped FirstLight training pass

The training command first verifies the trajectory store and builds a stable fingerprint
from the safe training projection only. It selects only terminated episodes in the
`train` split; validation, test, and truncated episodes never enter this first pass.
Without `--run`, the command only prints a plan and does not import FirstLight/PyTorch,
create a model, or write checkpoints.

```powershell
uv run elixir-echo train-firstlight `
  "D:\path\to\trajectory-cache" `
  --firstlight-root "D:\path\to\FirstLight_CR" `
  --checkpoint-directory "D:\path\to\elixirecho-checkpoints" `
  --max-episodes 2
```

After reviewing the plan, add `--run` to execute that capped pass. To continue from an
immutable checkpoint, pass its directory with `--resume-from`; the dataset fingerprint
and episode prefix are checked before model/optimizer state is restored. Choose `--device
cuda` only after confirming the installed PyTorch build supports the intended GPU.

```powershell
uv run elixir-echo train-firstlight `
  "D:\path\to\trajectory-cache" `
  --firstlight-root "D:\path\to\FirstLight_CR" `
  --checkpoint-directory "D:\path\to\elixirecho-checkpoints" `
  --max-episodes 2 `
  --checkpoint-every 1 `
  --device cuda `
  --run
```

Training uses the RICH profile and the existing recurrent FirstLight V4 imitation-learning
wrapper. This command is a capped research entry point, not a claim of model quality or a
recommendation to start large-scale training.

## Optional FirstLight compatibility checks

FirstLight remains an external, read-only checkout. Install a PyTorch build appropriate
for the local CPU/CUDA environment separately, then set the paths for the current shell:

```powershell
$env:ELIXIR_ECHO_FIRSTLIGHT_ROOT = "D:\path\to\FirstLight_CR"
$env:ELIXIR_ECHO_FIRSTLIGHT_CHECKPOINT = "D:\path\to\checkpoint-v1.pt"
uv run pytest -q tests/compatibility/test_firstlight_checkpoint_logits.py
```

That compatibility test loads the checkpoint and checks model outputs; it does not
optimize weights or save a checkpoint. The one-step optimizer smoke is opt-in and is not
run by normal setup:

```powershell
$env:ELIXIR_ECHO_RUN_TRAINING_SMOKE = "1"
uv run pytest -q tests/integration/test_firstlight_il_smoke.py
```

The runtime adapter disables bytecode writes into the configured FirstLight checkout.

The FirstLight integration is based on the external
[FirstLight CR project](https://github.com/Jaasssooooonnnn/FirstLight_CR), which is not
vendored here. Check the upstream repository's current license and asset terms before
redistributing upstream code, weights, or game resources. This repository does not yet
declare a distribution license.

## Current scope

This repository packages and validates the data-to-policy path; it does not claim a
measured competitive win rate or a production-ready standalone model. The current V4
integration is a compatibility baseline. New training runs and model-quality comparisons
are separate research steps, not automatic side effects of installing or testing the
package.

For the domain vocabulary and ownership boundaries, see [CONTEXT.md](CONTEXT.md). For
source and migration provenance, see [docs/provenance](docs/provenance/).
