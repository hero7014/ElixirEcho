"""Read-only decision diagnostics for a sealed recurrent V4 checkpoint.

The original project supplies the cache, checkpoint, model and legality rules.
Inputs remain teacher-forced; this does not simulate a live game.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import random
import sys
from typing import Any


@dataclass
class DecisionCounts:
    frames: int = 0
    wait: int = 0
    play: int = 0
    true_wait: int = 0
    false_act: int = 0
    false_wait: int = 0
    true_act: int = 0
    candidate_trials: int = 0
    candidate_hits: int = 0
    target_trials: int = 0
    target_hits: int = 0
    card_and_target_hits: int = 0
    full_decision_hits: int = 0
    random_candidate_expected_hits: float = 0.0
    random_target_expected_hits: float = 0.0

    def add(self, other: "DecisionCounts") -> None:
        for item in fields(self):
            setattr(self, item.name, getattr(self, item.name) + getattr(other, item.name))


def _rate(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def decision_metrics(counts: DecisionCounts) -> dict[str, float | None]:
    supported = counts.wait + counts.play
    precision = _rate(counts.true_act, counts.true_act + counts.false_act)
    recall = _rate(counts.true_act, counts.play)
    specificity = _rate(counts.true_wait, counts.wait)
    return {
        "gate_accuracy": _rate(counts.true_wait + counts.true_act, supported),
        "always_wait_gate_accuracy": _rate(counts.wait, supported),
        "act_precision": precision,
        "act_recall": recall,
        "act_f1": (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0 if supported else None
        ),
        "wait_specificity": specificity,
        "balanced_gate_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None else None
        ),
        "candidate_accuracy_given_expert_act": _rate(counts.candidate_hits, counts.candidate_trials),
        "uniform_legal_candidate_expected_accuracy": _rate(
            counts.random_candidate_expected_hits, counts.candidate_trials
        ),
        "target_accuracy_given_expert_card": _rate(counts.target_hits, counts.target_trials),
        "uniform_legal_target_expected_accuracy": _rate(
            counts.random_target_expected_hits, counts.target_trials
        ),
        "card_and_target_accuracy_given_expert_act": _rate(
            counts.card_and_target_hits, counts.target_trials
        ),
        "full_decision_accuracy_given_expert_act": _rate(
            counts.full_decision_hits, counts.target_trials
        ),
    }


def episode_bootstrap(
    episodes: dict[str, DecisionCounts], *, draws: int = 2000, seed: int = 20261024
) -> dict[str, list[float]]:
    if len(episodes) < 10:
        return {}
    population = tuple(episodes.values())
    rng = random.Random(seed)
    names = (
        "act_recall", "act_f1", "candidate_accuracy_given_expert_act",
        "card_and_target_accuracy_given_expert_act",
    )
    sampled: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(draws):
        total = DecisionCounts()
        for _ in population:
            total.add(rng.choice(population))
        metrics = decision_metrics(total)
        for name in names:
            value = metrics[name]
            if value is not None:
                sampled[name].append(value)
    return {
        name: [sorted(values)[int(0.025 * (len(values) - 1))],
               sorted(values)[int(0.975 * (len(values) - 1))]]
        for name, values in sampled.items() if values
    }


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
    parser.add_argument("--max-batches", type=int, help="bounded smoke; omit for the full sealed test split")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")
    legacy_src = args.legacy_root.resolve() / "src"
    if not (legacy_src / "training" / "recurrent_il.py").is_file():
        parser.error("--legacy-root does not contain the recurrent trainer")

    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(legacy_src))
    import torch
    from training.firstlight_v4 import load_firstlight_v4
    from training.recurrent_execution import TrainingExecutionConfigV1
    from training.recurrent_il import _autocast, _slice_sequence, load_checkpoint_v2
    from training.recurrent_labels import TrainingConfigV1
    from training.recurrent_streaming import StreamingRecurrentDatasetV1

    manifest_path = args.cache_manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = TrainingConfigV1()
    execution = TrainingExecutionConfigV1()
    runtime = load_firstlight_v4(args.firstlight_root)
    from native_runner.training.v4.decoding import ShadowCandidateLegality
    from native_runner.training.v4.tensors import concatenate_padded_tensor_records
    device = torch.device(args.device)
    model = runtime.build_model(device=device)
    restored = load_checkpoint_v2(
        runtime, checkpoint_path, model,
        expected_config=config, expected_execution_config=execution,
        expected_cache_manifest=manifest, map_location=device,
    )
    model.eval()
    stream = StreamingRecurrentDatasetV1(
        runtime, manifest_path, config, execution, split="test", epoch=0
    )
    batches = stream.iter_batches()
    if args.max_batches is not None:
        batches = islice(batches, args.max_batches)

    by_episode: dict[str, DecisionCounts] = {}
    batch_count = 0
    with torch.inference_mode():
        for batch_count, item in enumerate(batches, start=1):
            print(f"decision audit: test batch {batch_count}", file=sys.stderr, flush=True)
            batch = item.batch
            sequence = batch.sequence
            supervision = batch.supervision
            width = sequence.batch_size
            state = model.initial_state(width, device=device)
            for start in range(0, sequence.time_steps, config.tbptt_length):
                stop = min(start + config.tbptt_length, sequence.time_steps)
                chunk = _slice_sequence(sequence, start, stop).to(device)
                with _autocast(device, config):
                    flat = concatenate_padded_tensor_records(tuple(chunk.observations))
                    encoded = model.encode_observation(flat, validate=batch_count == 1 and start == 0)
                    predicted_gate_rows: list[Any] = []
                    candidate_hit_rows: list[Any] = []
                    target_hit_rows: list[Any] = []
                    random_candidate_rows: list[Any] = []
                    random_target_rows: list[Any] = []
                    for step in range(chunk.time_steps):
                        observation = flat.narrow_batch(step * width, width)
                        context = model.advance_core(
                            encoded.narrow_batch(step * width, width), state,
                            episode_start=chunk.episode_start[step],
                        )
                        state = context.next_state
                        shadow = ShadowCandidateLegality(observation.candidates, runtime.config)
                        legal = shadow.candidate_mask()
                        gate_logits = model.gate_head(context.policy_context)
                        predicted_gate_rows.append(
                            torch.where(
                                legal.any(dim=1), gate_logits.argmax(dim=1),
                                torch.full((width,), runtime.types.GATE_WAIT, device=device),
                            )
                        )
                        candidate_hit = torch.zeros(width, dtype=torch.bool, device=device)
                        target_hit = torch.zeros_like(candidate_hit)
                        random_candidate = torch.zeros(width, dtype=torch.float32, device=device)
                        random_target = torch.zeros_like(random_candidate)
                        candidate_mask = supervision.candidate[start + step].to(device) & chunk.valid_mask[step]
                        selected = candidate_mask.nonzero(as_tuple=False).flatten()
                        if selected.numel():
                            selected_observation = observation.index_select(selected)
                            selected_context = context.index_select(selected)
                            selected_action = chunk.actions[step].index_select(selected)
                            selected_shadow = ShadowCandidateLegality(
                                selected_observation.candidates, runtime.config
                            )
                            selected_legal = selected_shadow.candidate_mask()
                            expert_index = selected_action.candidate_index[:, 0]
                            if not bool(selected_legal.gather(1, expert_index[:, None]).all()):
                                raise ValueError("expert candidate is not legal in decision audit")
                            decoder = model._decoder_context(
                                model.decoder_initial(selected_context.policy_context),
                                selected_context.encoded,
                            )
                            logits = model.candidate_logits(
                                decoder, selected_context.encoded.candidate_memory
                            )
                            predicted = logits.masked_fill(~selected_legal, float("-inf")).argmax(dim=1)
                            candidate_hit.index_copy_(0, selected, predicted == expert_index)
                            random_candidate.index_copy_(
                                0, selected, selected_legal.sum(dim=1).float().reciprocal()
                            )

                        target_mask = supervision.target[start + step].to(device) & chunk.valid_mask[step]
                        targeted = target_mask.nonzero(as_tuple=False).flatten()
                        if targeted.numel():
                            targeted_observation = observation.index_select(targeted)
                            targeted_context = context.index_select(targeted)
                            targeted_action = chunk.actions[step].index_select(targeted)
                            expert_index = targeted_action.candidate_index[:, 0]
                            targeted_shadow = ShadowCandidateLegality(
                                targeted_observation.candidates, runtime.config
                            )
                            placement = targeted_shadow.selected_placement(expert_index).flatten(1)
                            if not bool(placement.any(dim=1).all()):
                                raise ValueError("expert target has no legal placement")
                            decoder = model._decoder_context(
                                model.decoder_initial(targeted_context.policy_context),
                                targeted_context.encoded,
                            )
                            memory = model._selected_candidate_memory(
                                targeted_context.encoded.candidate_memory, expert_index
                            )
                            location_logits, _ = model._location_logits(
                                decoder, memory, targeted_context.encoded.spatial_memory
                            )
                            predicted_target = location_logits.flatten(1).masked_fill(
                                ~placement, float("-inf")
                            ).argmax(dim=1)
                            target_hit.index_copy_(
                                0, targeted, predicted_target == targeted_action.target_cell[:, 0]
                            )
                            random_target.index_copy_(
                                0, targeted, placement.sum(dim=1).float().reciprocal()
                            )
                        candidate_hit_rows.append(candidate_hit)
                        target_hit_rows.append(target_hit)
                        random_candidate_rows.append(random_candidate)
                        random_target_rows.append(random_target)

                predicted_gate = torch.stack(predicted_gate_rows).cpu().tolist()
                expert_gate = torch.stack([action.gate for action in chunk.actions]).cpu().tolist()
                candidate_hits = torch.stack(candidate_hit_rows).cpu().tolist()
                target_hits = torch.stack(target_hit_rows).cpu().tolist()
                random_candidates = torch.stack(random_candidate_rows).cpu().tolist()
                random_targets = torch.stack(random_target_rows).cpu().tolist()
                valid = chunk.valid_mask.cpu().tolist()
                for offset in range(chunk.time_steps):
                    for lane, descriptor in enumerate(batch.descriptors):
                        if not valid[offset][lane]:
                            continue
                        counts = by_episode.setdefault(descriptor.episode_id, DecisionCounts())
                        counts.frames += 1
                        if supervision.gate[start + offset, lane]:
                            expected = expert_gate[offset][lane]
                            predicted = predicted_gate[offset][lane]
                            if expected == runtime.types.GATE_WAIT:
                                counts.wait += 1
                                if predicted == expected:
                                    counts.true_wait += 1
                                else:
                                    counts.false_act += 1
                            elif expected == runtime.types.GATE_ACT:
                                counts.play += 1
                                if predicted == expected:
                                    counts.true_act += 1
                                else:
                                    counts.false_wait += 1
                            else:
                                raise ValueError("unsupported supervised gate label")
                        if supervision.candidate[start + offset, lane]:
                            counts.candidate_trials += 1
                            counts.candidate_hits += int(candidate_hits[offset][lane])
                            counts.random_candidate_expected_hits += random_candidates[offset][lane]
                        if supervision.target[start + offset, lane]:
                            card_hit = bool(candidate_hits[offset][lane])
                            placement_hit = bool(target_hits[offset][lane])
                            counts.target_trials += 1
                            counts.target_hits += int(placement_hit)
                            counts.card_and_target_hits += int(card_hit and placement_hit)
                            counts.full_decision_hits += int(
                                card_hit and placement_hit
                                and predicted_gate[offset][lane] == runtime.types.GATE_ACT
                            )
                            counts.random_target_expected_hits += random_targets[offset][lane]
                state = state.detach()

    total = DecisionCounts()
    for counts in by_episode.values():
        total.add(counts)
    if not batch_count:
        raise ValueError("sealed test split has no batches")
    if total.wait + total.play != total.true_wait + total.false_act + total.false_wait + total.true_act:
        raise ValueError("gate confusion matrix does not reconcile")
    if args.max_batches is None:
        expected_frames = sum(
            item["frame_count"] for item in manifest["artifacts"] if item["split"] == "test"
        )
        if total.frames != expected_frames:
            raise ValueError("decision audit did not cover the full sealed test split")
    report = {
        "schema": "elixir_echo.legacy_decision_diagnostics@1",
        "scope": "full_test_split" if args.max_batches is None else "bounded_smoke",
        "checkpoint_id": restored.checkpoint_id,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "cache_manifest_sha256": file_sha256(manifest_path),
        "training_config_fingerprint": config.fingerprint(),
        "execution_config_fingerprint": execution.fingerprint(),
        "device": str(device),
        "batch_count": batch_count,
        "episode_count": len(by_episode),
        "counts": asdict(total),
        "metrics": decision_metrics(total),
        "episode_bootstrap_95pct": episode_bootstrap(by_episode),
        "interpretation": (
            "Actions are predicted from logged observations and public temporal inputs. "
            "Candidate and placement metrics condition on the expert ACT; placement also "
            "conditions on the expert card. This is not closed-loop gameplay."
        ),
    }
    destination = args.report.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"report": str(destination), "metrics": report["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
