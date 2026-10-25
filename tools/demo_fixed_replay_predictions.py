"""Compare FirstLight V4 greedy predictions with labels on one saved replay.

This is a bounded, read-only display tool. It does not run Native, train a
model, or write to the checkpoint, source replay, or trajectory. Only the
small JSON report named by --output is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from elixir_echo.datasets import TrainingTrajectoryReader
from elixir_echo.integrations.firstlight.policy import FirstLightPolicyBackend
from elixir_echo.integrations.firstlight.runtime import FirstLightRuntime
from elixir_echo.integrations.firstlight.tensorizer import PolicyInputBuilder
from elixir_echo.profiles import ObservationProfile
from elixir_echo.temporal import PublicStateTracker


DEFAULT_REPLAY_ID = "86582cdb2f0f4e8c94e9d9540d75e4cb"
DEFAULT_SAMPLE_TICKS = (90, 245, 425, 705, 830, 1310, 2000, 2700, 3505, 3600)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expert_action(step: Any) -> list[dict[str, Any]]:
    return [dict(action) for action in step.expert_action.actions]


def _verify_candidate_mapping(tensorized: Any, mask: Mapping[str, Any], actor: int) -> int:
    """Check every tensorized hand candidate against its source ActionMask row."""

    candidates = tensorized.batch.candidates
    hand_slots = tuple(bool(value) for value in mask.get("hand_slots", ()))
    placements = mask.get("placement_masks", {})
    if len(hand_slots) != 4 or not isinstance(placements, Mapping):
        raise ValueError("ActionMask has no four-slot hand/placement contract")
    if set(tensorized.candidate_index_by_slot) != {
        slot for slot, legal in enumerate(hand_slots) if legal
    }:
        raise ValueError("tensorized candidate slots do not match ActionMask legal hand slots")
    if tensorized.candidate_index_by_slot.keys() != tensorized.candidate_uid_by_slot.keys():
        raise ValueError("candidate row and UID slot maps disagree")

    checked = 0
    for slot, row in tensorized.candidate_index_by_slot.items():
        entry = placements.get(str(slot))
        if not isinstance(entry, Mapping):
            raise ValueError(f"legal hand slot {slot} has no ActionMask placement row")
        if not bool(candidates.mask[0, row].item()):
            raise ValueError(f"legal hand slot {slot} maps to a masked candidate row")
        if int(candidates.uid[0, row].item()) != tensorized.candidate_uid_by_slot[slot]:
            raise ValueError(f"candidate UID map differs for hand slot {slot}")
        if int(candidates.native_hand_slot[0, row].item()) != slot:
            raise ValueError(f"candidate row maps to the wrong native hand slot {slot}")
        card_id = int(candidates.native_visible_card_id[0, row].item())
        if entry.get("visible_card_id", entry.get("card_id")) != card_id:
            raise ValueError(f"candidate row maps to the wrong visible card at slot {slot}")

        expected = entry.get("row_major")
        if not isinstance(expected, Sequence) or len(expected) != candidates.placement.shape[-2]:
            raise ValueError(f"ActionMask placement matrix is malformed for slot {slot}")
        actual = candidates.placement[0, row].tolist()
        if actor == 1:
            actual = [list(reversed(line)) for line in reversed(actual)]
        if actual != [list(map(bool, line)) for line in expected]:
            raise ValueError(f"candidate placement row differs from ActionMask at slot {slot}")
        checked += 1
    return checked


def _decode_prediction(
    output: Any,
    tensorized: Any,
    *,
    actor: int,
    mask: dict[str, Any],
    components: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Decode V4's selected candidate rows using the tensorizer's slot map."""

    actions = output.actions
    candidates = tensorized.batch.candidates
    gate = int(actions.gate[0].item())
    if gate == components.types.GATE_WAIT:
        return ([{"kind": "wait"}], [{"gate": "WAIT", "candidate_rows_checked": 0}])
    if gate != components.types.GATE_ACT:
        raise ValueError(f"V4 returned unknown gate value {gate}")

    decoded: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    count = int(actions.micro_action_count[0].item())
    hand_slots = tuple(bool(value) for value in mask.get("hand_slots", ()))
    placement_masks = mask.get("placement_masks", {})
    if not bool(mask.get("kinds", {}).get("play_card", False)):
        raise ValueError("V4 selected ACT while the current ActionMask forbids card play")

    for action_index in range(count):
        candidate_row = int(actions.candidate_index[0, action_index].item())
        if not bool(candidates.mask[0, candidate_row].item()):
            raise ValueError("V4 selected a padded/illegal candidate row")
        variant = int(candidates.variant[0, candidate_row].item())
        if variant != components.types.CANDIDATE_DEPLOY:
            raise ValueError(f"prediction demo cannot display candidate variant {variant}")
        slot = int(candidates.native_hand_slot[0, candidate_row].item())
        card_id = int(candidates.native_visible_card_id[0, candidate_row].item())
        if not 0 <= slot < len(hand_slots) or not hand_slots[slot]:
            raise ValueError("V4 selected a hand slot forbidden by the ActionMask")
        placement_entry = placement_masks.get(str(slot))
        if not isinstance(placement_entry, Mapping):
            raise ValueError("selected hand slot has no current placement contract")
        expected_card = placement_entry.get("visible_card_id", placement_entry.get("card_id"))
        if expected_card != card_id:
            raise ValueError("candidate-to-hand-slot card mapping disagrees with ActionMask")

        target_cell = int(actions.target_cell[0, action_index].item())
        grid: list[int] | None = None
        target_mode = int(candidates.target_mode[0, candidate_row].item())
        if target_mode == components.types.TARGET_GRID:
            width, height = components.config.board_width, components.config.board_height
            model_x, model_y = target_cell % width, target_cell // width
            native_x, native_y = (width - 1 - model_x, height - 1 - model_y) if actor == 1 else (model_x, model_y)
            rows = placement_entry.get("row_major")
            if (
                not isinstance(rows, Sequence)
                or len(rows) != height
                or not isinstance(rows[native_y], Sequence)
                or len(rows[native_y]) != width
                or not bool(rows[native_y][native_x])
            ):
                raise ValueError("V4 selected a target cell forbidden by the native ActionMask")
            grid = [native_x, native_y]
        elif target_mode != components.types.TARGET_NONE:
            raise ValueError(f"unsupported V4 target mode {target_mode}")

        delay_bin = int(actions.delay_offset_bin[0, action_index].item())
        delay_ms = int(components.config.delay_offset_ms[delay_bin])
        decoded.append(
            {
                "kind": "play_card",
                "card_id": card_id,
                "hand_slot": slot,
                "target_grid": grid,
                "delay_ms": delay_ms,
                "delay_supervision": "not_trained_in_historical_checkpoint",
            }
        )
        checks.append(
            {
                "candidate_row": candidate_row,
                "candidate_uid": int(candidates.uid[0, candidate_row].item()),
                "mapped_hand_slot": slot,
                "mapped_card_id": card_id,
                "action_mask_slot_legal": True,
                "target_mask_legal": True,
            }
        )
    return decoded, checks


def _comparison(expert: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> str:
    if not expert:
        return "match" if len(predicted) == 1 and predicted[0].get("kind") == "wait" else "different"
    if len(expert) != 1 or expert[0].get("kind") != "play_card":
        return "not_comparable_to_wait_single_card_head"
    if len(predicted) != 1 or predicted[0].get("kind") != "play_card":
        return "different"
    keys = ("card_id", "hand_slot", "target_grid")
    return "match" if all(expert[0].get(key) == predicted[0].get(key) for key in keys) else "different"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--replay-id", default=DEFAULT_REPLAY_ID)
    parser.add_argument("--firstlight-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-ticks", default=",".join(map(str, DEFAULT_SAMPLE_TICKS)))
    args = parser.parse_args()

    sample_ticks = {int(item.strip()) for item in args.sample_ticks.split(",") if item.strip()}
    if not sample_ticks:
        parser.error("--sample-ticks must include at least one tick")
    for path, label in (
        (args.trajectory_root, "trajectory root"),
        (args.firstlight_root, "FirstLight checkout"),
        (args.checkpoint, "formal checkpoint"),
    ):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")

    episode = TrainingTrajectoryReader(args.trajectory_root).read_episode(args.replay_id)
    if episode.outcome != "terminated":
        raise ValueError(f"expected the fixed terminated replay; got {episode.outcome!r}")
    actors = {step.identity.actor for step in episode.steps}
    if actors != {0, 1}:
        raise ValueError(f"fixed episode must contain both actors; got {sorted(actors)}")

    runtime = FirstLightRuntime(args.firstlight_root)
    backend = FirstLightPolicyBackend(
        runtime,
        device=args.device,
        checkpoint=args.checkpoint,
    )
    backend.model.eval()
    if backend.checkpoint_path is None:
        raise ValueError("formal checkpoint was not loaded")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise ValueError("CUDA was requested but is unavailable")

    builder = PolicyInputBuilder(
        runtime,
        components=backend.components,
        legacy_checkpoint_input=True,
    )
    trackers: dict[int, PublicStateTracker] = {actor: PublicStateTracker() for actor in actors}
    states: dict[int, Any] = {
        actor: backend.model.initial_state(1, device=backend.device) for actor in actors
    }
    first_step = {actor: True for actor in actors}
    report_records: list[dict[str, Any]] = []
    inference_count = 0
    candidate_slots_checked = 0
    play_card_mask_frames = 0

    with torch.inference_mode():
        for step in episode.steps:
            actor = step.identity.actor
            temporal = trackers[actor].before_step(step)
            policy_input = builder.build(
                step.observation,
                step.action_mask,
                temporal,
                ObservationProfile.RICH,
            )
            tensorized = policy_input.tensors
            if tensorized.input_mode != "legacy_checkpoint":
                raise ValueError("fixed replay was not tensorized in legacy_checkpoint mode")
            constraints = step.action_mask.constraints
            candidate_slots_checked += _verify_candidate_mapping(tensorized, constraints, actor)
            play_card_mask_frames += int(bool(constraints.get("kinds", {}).get("play_card", False)))
            batch = tensorized.batch.to_model_input(backend.device)
            output = backend.model.act(
                batch,
                states[actor],
                episode_start=torch.tensor([first_step[actor]], dtype=torch.bool, device=backend.device),
                validate=True,
            )
            states[actor] = output.next_state
            first_step[actor] = False
            inference_count += 1

            if step.identity.tick in sample_ticks:
                expert = _expert_action(step)
                predicted, mask_checks = _decode_prediction(
                    output,
                    tensorized,
                    actor=actor,
                    mask=constraints,
                    components=backend.components,
                )
                report_records.append(
                    {
                        "episode_id": step.identity.episode_id,
                        "tick": step.identity.tick,
                        "actor": actor,
                        "input_profile": "RICH",
                        "checkpoint_input_mode": "legacy_checkpoint",
                        "action_mask": {
                            "wait": bool(step.action_mask.constraints.get("kinds", {}).get("wait", False)),
                            "play_card": bool(step.action_mask.constraints.get("kinds", {}).get("play_card", False)),
                            "ability": bool(step.action_mask.constraints.get("kinds", {}).get("activate_ability", False)),
                            "legal_hand_slots": [
                                i
                                for i, legal in enumerate(step.action_mask.constraints.get("hand_slots", ()))
                                if legal
                            ],
                            "tensorized_candidate_count": int(tensorized.batch.candidates.mask.sum().item()),
                            "candidate_slot_mapping_valid": True,
                        },
                        "expert_action": expert,
                        "model_prediction": predicted,
                        "candidate_and_action_mask_checks": mask_checks,
                        "comparison": _comparison(expert, predicted),
                    }
                )
            # The current action is committed only after producing its input and prediction.
            trackers[actor].after_action(step.expert_action)

    expected = len(sample_ticks) * len(actors)
    if len(report_records) != expected:
        found = [(item["actor"], item["tick"]) for item in report_records]
        raise ValueError(f"sample ticks were not present for both actors: {found}")

    report = {
        "schema": "elixirecho.fixed-replay-prediction-demo@1",
        "replay_id": episode.episode_id,
        "trajectory_outcome": episode.outcome,
        "trajectory_split": episode.split,
        "decision_ticks_per_actor": len(episode.steps) // len(actors),
        "actor_records": len(episode.steps),
        "inference_records_processed": inference_count,
        "action_mask_validation": {
            "candidate_slot_mappings_checked": candidate_slots_checked,
            "candidate_slot_mapping_errors": 0,
            "frames_where_card_play_was_legal": play_card_mask_frames,
            "sampled_candidate_and_target_checks_passed": True,
        },
        "checkpoint_sha256": _sha256(backend.checkpoint_path),
        "checkpoint_input_mode": "legacy_checkpoint",
        "observation_profile": "RICH",
        "inference": "FirstLight V4 model.act, greedy argmax with native ActionMask-backed candidates",
        "historical_split_membership": "not_asserted; see report notes",
        "notes": [
            "This is teacher-forced replay input, not live play or a game simulation.",
            "Model state is recurrent within each actor sequence and reset at that actor's first frame.",
            "The current expert action is never used to create its own input; it is committed to tracker only after prediction.",
            "Candidate rows map through TensorizedDecision to native hand slots and card IDs; chosen target cells are checked against the same ActionMask.",
            "Historical checkpoint delay supervision count was zero; predicted delay values are displayed only as unsupervised decoder output.",
            "Matching counts on this fixed replay are descriptive and are not a held-out quality estimate.",
        ],
        "records": report_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comparable = [item for item in report_records if item["comparison"] != "not_comparable_to_wait_single_card_head"]
    matches = sum(item["comparison"] == "match" for item in comparable)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(report_records),
                "inference_records_processed": inference_count,
                "comparable_records": len(comparable),
                "descriptive_matches": matches,
                "checkpoint_sha256": report["checkpoint_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
