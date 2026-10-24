"""Evaluate an existing research checkpoint with its original training contract.

This is a thin, read-only entry point. The cache loader, checkpoint validation,
V4 policy, and recurrent loss all come from the earlier local project.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--firstlight-root", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, help="bounded smoke only; omit for the full test split")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    legacy_src = args.legacy_root.resolve() / "src"
    if not (legacy_src / "training" / "recurrent_il.py").is_file():
        parser.error("--legacy-root does not contain the original recurrent trainer")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(legacy_src))

    import torch
    from training.firstlight_v4 import load_firstlight_v4
    from training.recurrent_execution import TrainingExecutionConfigV1
    from training.recurrent_il import evaluate_recurrent_batches, load_checkpoint_v2
    from training.recurrent_labels import TrainingConfigV1
    from training.recurrent_streaming import StreamingRecurrentDatasetV1

    manifest_path = args.cache_manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = TrainingConfigV1()
    execution = TrainingExecutionConfigV1()
    runtime = load_firstlight_v4(args.firstlight_root)
    device = torch.device(args.device)
    model = runtime.build_model(device=device)
    restored = load_checkpoint_v2(
        runtime,
        checkpoint_path,
        model,
        expected_config=config,
        expected_execution_config=execution,
        expected_cache_manifest=manifest,
        map_location=device,
    )
    dataset = StreamingRecurrentDatasetV1(
        runtime, manifest_path, config, execution, split="test", epoch=0
    )
    def test_batches():
        for index, item in enumerate(dataset.iter_batches(), start=1):
            print(f"evaluating test batch {index}", file=sys.stderr, flush=True)
            yield item.batch

    batches = test_batches()
    if args.max_batches is not None:
        batches = islice(batches, args.max_batches)
    metrics = evaluate_recurrent_batches(runtime, model, batches, config, execution)
    report = {
        "schema": "elixir_echo.legacy_heldout_evaluation@1",
        "scope": "full_test_split" if args.max_batches is None else "bounded_smoke",
        "max_batches": args.max_batches,
        "checkpoint_id": restored.checkpoint_id,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "cache_manifest_sha256": file_sha256(manifest_path),
        "training_config_fingerprint": config.fingerprint(),
        "execution_config_fingerprint": execution.fingerprint(),
        "device": str(device),
        "metrics": asdict(metrics),
        "interpretation": "Teacher-forced imitation loss; not gameplay win rate.",
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        destination = args.report.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
