# Third-party material and attribution

ElixirEcho connects to [FirstLight CR](https://github.com/Jaasssooooonnnn/FirstLight_CR)
as a user-supplied, read-only runtime. FirstLight's repository declares the
Apache License 2.0 and includes a NOTICE file. ElixirEcho does not bundle its
Native implementation, V4 weights, game binaries, or game resources. The
`integrations.firstlight.capture` call order was adapted from this project's
earlier local research implementation; the FirstLight classes it invokes are
loaded from the external checkout.

The optional [IL_Replay dataset](https://huggingface.co/datasets/VanguardX101/IL_Replay)
is obtained separately. The local dataset card reviewed for this repository
does not provide an explicit redistribution license. No replay payloads or
Parquet files are included.

Clash Royale and related game assets belong to their respective rights
holders. ElixirEcho is an independent research project and does not claim
affiliation or endorsement.

This file records dependencies and provenance. It does not grant a license
for the external dataset, game assets, FirstLight checkout, or historical
checkpoint. The repository owner must choose a license for ElixirEcho's own
code before a public release.
