# Reproducibility

## What can be reproduced without private assets

The package, domain tests, trajectory fixture, and command preflight run with
Python 3.12 or newer. `elixir-echo fixture` creates a synthetic two-actor
episode and performs one update of a small fake policy. It is a wiring check,
not evidence that a Clash Royale model learned to play.

## Capture and training inputs

Real replay capture requires a user-supplied IL_Replay Parquet dataset, a
compatible FirstLight checkout, the game's assets, and an already-running
Native control service. `capture` without `--run` reads no replay and creates
no output. `capture --run` requires an exact replay tag, records both actors
from the first decision tick, and rejects source/Native/semantic/cache errors
without substituting another replay. A successful episode has per-record and
aggregate hashes and can be checked with `verify-trajectory`.

The earlier local project produced a sealed full-episode cache and formal
recurrent V4 imitation-learning checkpoint. ElixirEcho's bridge checks
selected old episode, sequence shard, tensor and logit parity against those
artifacts. Those private assets are not in this repository; users must supply
their own compatible inputs to repeat the real-asset check. The checkpoint
identity and selected data hashes are checked before any comparison.

Training is explicit and capped. `train-firstlight` without `--run` reports
the selected terminated train episodes and input fingerprint. With `--run`,
it uses RICH, public temporal context, a FirstLight V4 recurrent policy, and
imitation loss. Train, validation, and test are separated by whole replay.
Ability, multi-action, delay, and value heads require their declared
supervision masks; unsupported labels are never silently converted to WAIT.

The historical checkpoint must be evaluated with its fingerprinted legacy
input mode. The current richer input mode is a separate experiment and cannot
be substituted silently. A training loss or single replay demonstration does
not establish competitive gameplay quality. See the [model card](model-card.md).

## Read-only evaluation of a historical checkpoint

The formal checkpoint uses the earlier project's streaming checkpoint format.
`tools/evaluate_legacy_checkpoint.py` is a thin entry point to that project's
existing checkpoint loader, test stream, FirstLight V4 policy, and recurrent IL
loss. It does not convert the checkpoint or train the model. Supply local paths
to assets you are permitted to use:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$legacy = "D:\path\to\earlier-project"
$provider = "D:\path\to\FirstLight-checkout"
& "$legacy\.venv\Scripts\python.exe" -B .\tools\evaluate_legacy_checkpoint.py `
  --legacy-root $legacy `
  --firstlight-root $provider `
  --cache-manifest "$legacy\data\training\recurrent-v4-cache\cache_manifest.json" `
  --checkpoint "$legacy\runs\formal-checkpoint\checkpoint-v1.pt" `
  --device cuda `
  --report ".\runs\heldout-test.json"
```

Add `--max-batches 1` for a bounded connection check; omit it for the full
sealed test split. The loader verifies the checkpoint's training and execution
fingerprints against the cache manifest. The report is ignored by Git. Its
teacher-forced loss and label counts are offline imitation metrics, not a win
rate or an evaluation of newly trained ElixirEcho weights.
