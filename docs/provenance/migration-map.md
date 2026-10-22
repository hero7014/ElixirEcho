# Implementation and verification map

This file summarizes the extracted ElixirEcho implementation and its compatibility
evidence. It is a project map, not a record of private conversations or a claim that the
independent-policy research goal is complete.

| Area | ElixirEcho module | Verification boundary |
| --- | --- | --- |
| Game-agent contracts | `elixir_echo.domain` | Identity, validation, and serialization tests |
| Observation selection | `elixir_echo.profiles` | `ObservationProfile.MIN/RICH` field-selection tests |
| FirstLight boundary | `elixir_echo.integrations.firstlight` | Read-only import, tensor and checkpoint-output checks |
| Episode dataset | `elixir_echo.datasets` | Atomic write, hashes, ordering, and deck-free reader tests |
| Public temporal context | `elixir_echo.temporal` | Causality, reset, and leakage tests |
| Policy interface | `elixir_echo.policies` | Recurrent-state identity and reset tests |
| Imitation-learning path | `elixir_echo.training` | Sequence/label checks and optional in-memory smoke |
| Local operations | `elixir_echo.cli` | Clean-install help and trajectory-cache verification |

The FirstLight adapter loads provider code from a user-configured checkout rather than
copying its implementation into this repository. The current policy integration is a
reusable compatibility baseline; it is not evidence of a new, independently trained
policy. Normal tests and CLI commands do not start model training.

`source-inventory.json` is a byte-level inventory from the earlier validated local
research workspace used during extraction. It excludes generated caches, replay data,
checkpoints, secrets, and local paths. It is historical migration evidence, not a
license grant or an inventory of the current ElixirEcho files.
