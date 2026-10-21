# ElixirEcho migration map

This repository is extracted from the validated local research workspace known as
`legacy_source`. The legacy workspace remains the read-only behavioral oracle during
migration and is not copied wholesale into this repository.

## Baseline scope

The immutable inventory covers publishable candidates from four source areas:

- `src/`: implementation modules;
- `tests/`: executable behavior and regression evidence;
- `tools/`: reproducible command-line workflows;
- `adb/`: two legacy device-inspection utilities retained for migration audit;
- `docs/`: approved technical specifications, plans, and audit notes.

Generated caches, replay datasets, checkpoints, logs, secrets, virtual environments,
game resources, and local executables are excluded. The formal checkpoint is represented
only by its identity, byte size, SHA-256 digest, and training-report digest.

## Migration order

| Legacy area | ElixirEcho destination | Compatibility gate |
| --- | --- | --- |
| Semantic contracts | `elixir_echo.domain` | Domain parity |
| Observation profiles | `elixir_echo.profiles` | Field-selection parity |
| FirstLight adapters | `elixir_echo.integrations.firstlight` | Read-only and tensor parity |
| Full-episode cache | `elixir_echo.datasets` | Hash and reader-boundary parity |
| History and tracker | `elixir_echo.temporal` | Causality and leakage tests |
| Recurrent policy boundary | `elixir_echo.policies` | Logit and recurrent-reset parity |
| Sequence compilation and IL | `elixir_echo.training` | Training-smoke parity |
| Operational scripts | `elixir_echo.cli` | Clean-clone CLI tests |

The detailed public rename map and ownership boundaries are defined in the approved
repository design. FirstLight remains an attributed external integration rather than
vendored project code.
