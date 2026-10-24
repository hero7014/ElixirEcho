# Read-only integration with the existing project

ElixirEcho is a separate, provider-neutral package. It does not copy itself into
the active project, change its Python environment, rewrite its cache, or replace
its checkpoint. The bridge below is an explicit verification tool; it reads a
selected legacy episode, one sealed Layer 2 shard, and a checkpoint, then prints
an equivalence report.

## Prepare an isolated environment

Use a dedicated Python 3.12 environment for the bridge. Do not install
ElixirEcho into the active project's existing virtual environment or run
installation commands from that environment. Install the same compatible
PyTorch and optional cache dependencies that the selected FirstLight checkout
requires, followed by the package and its test extra:

```powershell
py -3.12 -m venv .venv-bridge
.\.venv-bridge\Scripts\python.exe -m pip install --upgrade pip
# Install the compatible torch build and the provider's required Python packages.
.\.venv-bridge\Scripts\python.exe -m pip install torch safetensors zstandard pytest
.\.venv-bridge\Scripts\python.exe -m pip install -e .
```

The bridge uses CPU inference for old/new checkpoint-logit comparisons except
for the independent raw V4 wrapper smoke test, which selects CUDA when
available. It does not train or capture a replay. If a different Python or
dependency set is needed, create another isolated environment rather than
changing the active project's environment.

## Run the bridge

Pass every asset explicitly. Select a complete, terminated episode represented
in the chosen sealed Layer 2 shard. Before importing the project or running any
parity check, the tool verifies the episode ID against the shard's hashed
manifest and rejects a mismatched shard instead of silently skipping a check.

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv-bridge\Scripts\python.exe .\tools\verify_active_project_bridge.py `
  --baseline-root "<read-only active project root>" `
  --legacy-episode "<selected episode directory>" `
  --firstlight-root "<read-only FirstLight checkout>" `
  --checkpoint "<formal checkpoint file>" `
  --layer2-summary "<one sealed Layer 2 shard summary>"
```

The report goes to stdout by default. Add `--report <new path outside all
read-only input directories>` only if a persistent report is wanted; existing
files are never overwritten. The script inventories baseline and provider
source files and hashes the selected episode, checkpoint, Layer 2 summary, and
shard payload before and after the checks. It fails if those inputs change, if
bytecode/source inventories change, if any required check skips, or if the
bridge cannot restore the process import path after the checks. Any external
paths added during checks are listed in the report before the runner restores
its original `sys.path`.

The checks cover:

- legacy episode hash/order verification and the deck-free ElixirEcho reader;
- six actor-0 temporal RICH input and action-logit parity points against the
  legacy entry point using the selected formal checkpoint;
- one selected old Layer 2 shard versus the ElixirEcho compiler output;
- formal checkpoint loading and ElixirEcho policy-wrapper versus raw V4 output;
- FirstLight source inventory preservation and temporary import-path cleanup.

`EpisodeContext` is read only by the legacy integrity verifier. It is not part
of ElixirEcho's training projection or policy input. No model is trained, and no
Native runner or game assets are launched.

## Rollback

The bridge does not install into or write to either read-only project. To stop
using it, close the isolated environment and, if no longer needed, uninstall
`elixir-echo` from that isolated environment or remove that environment. The
active project and its checkpoint/cache remain as they were. The report tool
does not edit project files, package metadata, or imports in the active project.
