# ElixirEcho | 圣水回响

[简体中文](README.md) · [English](README.en.md)

*Learning to echo human elixir decisions.*

## Project overview

ElixirEcho is a personal imitation-learning research project built around complete Clash Royale replay trajectories. Its focus is constructing fair, traceable sequential examples from full matches and connecting them to recurrent policies and imitation-learning training.

It is not an official Clash Royale project or a production-ready online bot. The project reuses FirstLight's Native runner, BattleEnv, V4 model, and training components where possible. ElixirEcho owns its semantic contracts, trajectory boundaries, public history, and integration code. FirstLight source, game assets, replay data, and model weights are not bundled here.

## Core technical contributions

- Defines actor-fair `Observation`, `ActionMask`, `DecisionStep`, and complete-match `Trajectory` contracts, aligning current state, legal actions, and expert labels by episode/tick/actor.
- Stores complete trajectories with atomic writes, per-record hashes, and an episode aggregate hash; the training reader does not expose episode decks or provider diagnostics.
- Uses causal `History` and a deterministic `PublicStateTracker` for public events, revealed opponent cards, and previous actions; unknown facts remain unknown.
- Provides `MIN` / `RICH` observation profiles and a FirstLight V4 adapter. Historical formal checkpoints use their compatible legacy input mode.
- Reuses existing FirstLight V4 components for single-match capture, RICH input compilation, and training integration. G1 runtime smoke, G2 fixed-replay capture, G3 single-episode training, and G4 fresh-process restore passed. G2 used Native deployment resources already present on this host; a clean-machine installation or batch capture is not established.

## Data-flow architecture

```mermaid
flowchart LR
  Replay[Replay record] --> Native[FirstLight Native / BattleEnv]
  Native --> Capture[ElixirEcho capture adapter]
  Capture --> Frame[Observation + ActionMask]
  Frame --> Trajectory[Atomic full-episode Trajectory]
  Trajectory --> Temporal[History + deterministic public Tracker]
  Temporal --> Tensorizer[RICH V4 Tensorizer]
  Tensorizer --> Policy[FirstLight V4 recurrent policy / IL loss]
  Policy --> Checkpoint[Separate training and checkpoint integration]
```

FirstLight supplies the Native runner, BattleEnv, V4 encoders, recurrent policy, action heads, and core IL loss. ElixirEcho calls these components through read-only adapters and supplies its own data contracts and training orchestration. The FirstLight checkout remains external and unchanged.

## Experimental results

- **Formal checkpoint from the earlier research project:** its historical sealed-test evaluation covered 38 matches, 76 actor sequences, and 67,894 decision frames, with teacher-forced total IL loss 6.0717. This is an earlier-project result under its fingerprinted input contract, not an ElixirEcho retraining result or a win rate.
- **G1 runtime acceptance:** the specified V4/IL/checkpoint smoke test passed in the isolated environment with the public raw FirstLight Provider and PyTorch 2.5.1+cu124 on an RTX 4070 Laptop (1 test, 14.63 seconds). This verifies a bounded runtime training/checkpoint path, not Native capture.
- **G3/G4 single-episode training acceptance:** using the read-only archived fixed replay `86582cdb2f0f4e8c94e9d9540d75e4cb` (terminated, train split; dataset fingerprint `f95d10b96731134a93597e79548c202902886288fe53984182363ba6047f499b`), ElixirEcho trained on 1,406 actor records with RICH input for 44 optimizer updates in 6m 23s. A separate checkpoint was saved, then a fresh process restored the model, AdamW, and training progress and ran a finite forward pass on the same RICH input. Weight SHA-256: `925055abb124804dc901f200739bafcb41c42f1b98e68f53233c3a691ea41d80`. This is a complete single-match engineering run, not formal large-scale training or a model-quality evaluation.
- **G2 isolated Native capture on this host: PASS (one replay).** The isolated ElixirEcho checkout reconstructed fixed replay `86582cdb2f0f4e8c94e9d9540d75e4cb` to `terminated`, producing 703 continuous decision ticks (90–3600, every five ticks) and 1,406 paired actor records. Trajectory verification, tick/actor alignment, and all 1,406 label-leakage assertions passed. The replay came from user-provided local IL_Replay data; the Provider was the original-source FirstLight checkout, while Native was started from an existing local deployment. This is not proof of a fresh-machine setup or reliable arbitrary/batch capture. There were 63 expert actions: 57 card plays and 6 abilities. Under the contract, 1,344 empty-action records mean WAIT; 62 non-WAIT records include one two-action window. All 57 card labels matched their hand-slot masks. One ability label at tick 830 has `execute_offset_ticks=+3` while that frame's ability mask is false; tick 833 was not sampled separately, so immediate legality for every ability action is not claimed. See [Demo/Results](docs/demo-results.md).
- **Historical checkpoint prediction demo:** the checkpoint's `legacy_checkpoint` input mode was used for ordered inference over all 1,406 records. Twenty comparisons were emitted at 10 ticks × 2 actors. Every sampled model prediction was WAIT; 12 of 16 comparable WAIT/single-card labels matched, while four single-card labels were predicted as WAIT. Four ability/multi-action labels are outside this checkpoint's supported action-head comparison. This is a fixed-sample demonstration, not a win-rate or held-out metric.

The fixed replay appears in the earlier Pilot20 `train` split. Its membership in the historical formal checkpoint's train/validation/test splits could not be confirmed by matching the same replay identity, so this comparison is **not described as held out**. The historical checkpoint and sealed data are not included in this repository.

See [Demo and Results](docs/demo-results.md) and the [model card](docs/model-card.md) for run details, checkpoint hashes, action comparisons, and limitations.

## Demo

This repository does not distribute a GIF derived from the real replay. See [Demo and Results](docs/demo-results.md) for verified prediction records, run metrics, and reproduction details.

## Quick start

The core package requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --extra dev
uv run pytest -q
uv run elixir-echo --help
```

The base package has no runtime dependencies. Compatibility checks that require PyTorch, a FirstLight checkout, or a private replay cache skip when those resources are not configured. The FirstLight optimizer smoke runs only when `ELIXIR_ECHO_RUN_TRAINING_SMOKE=1` is explicitly set. Training commands print a plan by default and start only when `--run` is explicitly supplied with an episode cap.

## Verify a local trajectory

Point the command at an ElixirEcho Trajectory store root (the directory containing `episodes/`). It verifies episode and record hashes, ordering, identity, and full-match structure, then prints a summary through the training-safe reader.

```powershell
uv run elixir-echo verify-trajectory "D:\path\to\trajectory-cache"
```

The summary excludes episode context, full decks, provider diagnostics, and other audit-only metadata.

## Capture one fixed replay

Capture reuses the Native runner and the replay preparation, calibration, BattleEnv reset, and expert-command scheduling sequence already validated in the earlier research project. The Native service must already be running and reachable; this command does not launch MuMu.

Provide your own IL_Replay Parquet data, compatible FirstLight checkout, and Native game environment. First run the path preflight, which does not capture a replay:

```powershell
uv sync --extra dev --extra firstlight-capture
uv run elixir-echo capture `
  --replay-dataset "<local-IL_Replay-root>" `
  --replay-tag "<exact-replay-tag>" `
  --firstlight-root "<local-FirstLight-checkout>" `
  --output "<new-trajectory-cache-root>"
```

The preflight should report `ready: true` and `not_checked` for Native reachability. Add `--run` to connect to Native and process the requested replay. A missing identity or upstream reconstruction failure does not trigger a substitute replay. A complete match captures both actors from tick 90 every five ticks until a trusted terminated/truncated outcome and verifies the episode before atomic commit. The output must be outside the input dataset and FirstLight checkout.

On 2026-10-25, one fixed replay was captured from an isolated ElixirEcho checkout using Native deployment resources already present on this host. This does not establish that a new machine can launch Native from public source alone, nor that arbitrary replays will reconstruct successfully. Batch capture is unverified. With `--debug-train`, keep output under an isolated `runs/real-replay-chain/<run-id>` directory. Replay data, game assets, and Native setup are user-supplied and not distributed.

## Capped training entry point

The training command first verifies the trajectory store and fingerprints only the training-safe projection. The first pass selects only terminated full episodes in the `train` split; validation, test, and truncated episodes are excluded. Without `--run`, it only prints a plan and does not import FirstLight/PyTorch, create a model, or write checkpoints.

```powershell
uv run elixir-echo train-firstlight `
  "D:\path\to\trajectory-cache" `
  --firstlight-root "<local-FirstLight-checkout>" `
  --checkpoint-directory "D:\path\to\elixirecho-checkpoints" `
  --max-episodes 2
```

Add `--run` after reviewing the plan. To resume from a saved checkpoint, pass `--resume-from`; the dataset fingerprint and episode prefix are checked first. Use `--device cuda` only after confirming that the local PyTorch build supports the target GPU. This entry point reuses the external FirstLight V4 model and loss. It is a capped research entry point, not evidence of model quality.

## Dependencies, provenance, and limitations

- **Core:** Python 3.12+; `uv` installs development dependencies. The optional `firstlight-capture` extra installs a Parquet reader; `firstlight-cache` installs safetensors and zstandard. Install the PyTorch build required by the local FirstLight CPU/CUDA setup separately.
- **FirstLight is external:** [FirstLight CR](https://github.com/Jaasssoooonnnnn/FirstLight_CR). Its source, weights, and game resources are not vendored here; check upstream code and asset terms before use or redistribution.
- The historical formal checkpoint, sealed data, source IL_Replay data, and Native game resources are not included. The real-replay prediction script requires a compatible local checkpoint, trajectory, and FirstLight checkout.
- Verified capabilities include the G1 runtime smoke, G2 capture of one fixed replay on this host, G3 training on one archived complete match for 44 updates, G4 fresh-process restore, historical-checkpoint loading compatibility, and a small teacher-forced prediction display. Clean-machine Native installation, batch production, multi-replay formal training, autonomous simulator play, and win rate are unverified. G3 is an engineering acceptance, not evidence of model quality; the historical checkpoint predicted only WAIT at the sampled points.
- ElixirEcho-authored implementation code under `src/`, `tests/`, and `tools/` is licensed under Apache-2.0; see the root [LICENSE](LICENSE). This license covers only code the project owner is entitled to license. It does not grant rights to FirstLight, IL_Replay or replay-derived materials, model weights, or game/Native resources; see [Third-party Notices](THIRD_PARTY_NOTICES.md). This is not an official game tool and makes no production deployment or online-safety claim.

See [CONTEXT.md](CONTEXT.md) for domain vocabulary and ownership boundaries, [architecture](docs/architecture.md) for system structure, and [`docs/provenance`](docs/provenance/) for source and migration records.
