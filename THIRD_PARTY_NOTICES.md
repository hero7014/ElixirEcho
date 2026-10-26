# Third-party notices and licensing boundaries

## FirstLight CR

[FirstLight CR](https://github.com/Jaasssoooonnnnn/FirstLight_CR) is the
external provider used for Native replay reconstruction and the V4 model. Its
public repository identifies [Apache License 2.0](https://github.com/Jaasssoooonnnnn/FirstLight_CR/blob/main/LICENSE)
and includes a [`NOTICE`](https://github.com/Jaasssoooonnnnn/FirstLight_CR/blob/main/NOTICE).
The upstream NOTICE attributes FirstLight CR to its contributors and says the
repository does not distribute game binaries, resources, or modified clients.
It also clarifies that locally supplied APKs and resources remain subject to
their owners' rights.

ElixirEcho's FirstLight integration loads `native_runner.*` from a separately
configured checkout at runtime. This repository does not include the upstream
Native source tree, V4 implementation, formal checkpoint, game binaries, or
game resources. The capture call order is ported from the author's earlier
local capture implementation. The tracked source tree contains ElixirEcho
integration modules rather than a vendored `native_runner` implementation.
This is a provenance statement, not a grant of rights to the external checkout
or any game assets.

If a distributed ElixirEcho file contains copied or derivative FirstLight
source, Apache-2.0 redistribution conditions apply to that material. They
include providing the license text, retaining applicable source notices,
marking modified files, and reproducing applicable upstream NOTICE attribution
in an allowed location. The existence of a runtime import alone does not copy
or relicense the external provider. This inventory is not a legal conclusion
about every possible source-level similarity.

The current tracked tree and migration map were checked for source headers and
FirstLight code files. No vendored FirstLight implementation file or
file-specific FirstLight copyright header was identified, so no FirstLight
file modification notice was added to ElixirEcho source in this pass. If a
copied or derivative upstream file is later identified or added, retain its
applicable copyright and license notices, mark modifications, and include the
applicable NOTICE attribution as required by Apache-2.0. The historical
`source-inventory.json` records paths, sizes, and hashes; it is not a record of
copyright ownership. This review cannot determine authorship beyond the
project owner's authorization for ElixirEcho-owned code.

## ElixirEcho code license

ElixirEcho's original implementation code under `src/`, `tests/`, and `tools/`
is licensed under Apache-2.0 by the project owner, as stated in the root
[`LICENSE`](LICENSE). This applies only to code the project owner is
entitled to license. It does not relicense the separately obtained FirstLight
Provider or grant rights to the IL_Replay dataset, replay-derived materials,
historical checkpoints or model weights, game assets, or Native resources.
The root code license does not grant blanket rights to other repository
materials unless they carry a separate license notice.
The README includes `docs/assets/trajectory-montage-3x3.gif`, a semantic
visualization rather than game footage. It does not include a source replay
payload. The GIF is a display asset, not code covered by the root Apache-2.0
license.

## Dataset and game assets

The optional [IL_Replay dataset](https://huggingface.co/datasets/VanguardX101/IL_Replay)
is obtained separately. The dataset page reviewed on 2026-10-25 does not show
an explicit redistribution license. No replay payloads or Parquet files are
included in this repository.

Clash Royale and related game assets belong to their respective rights holders.
ElixirEcho is an independent research project and does not claim affiliation
or endorsement.
