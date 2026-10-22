"""Reusable synthetic V4 tensorizer inputs for unit and compatibility tests."""

from __future__ import annotations


def observation_payload(*, rich: bool) -> dict[str, object]:
    entity: dict[str, object] = {
        "entity_id": 44,
        "owner": 0,
        "card_id": 26_000_000,
        "current_form": {
            "status": "known",
            "value": {"variant": "base", "form_id": "base"},
        },
        "entity_kind": "troop",
        "position": [9_000.0, 12_000.0],
        "hitpoints": 400.0,
        "max_hitpoints": 500.0,
        "shield": {"hitpoints": 50.0},
        "visibility_basic": "visible",
    }
    if rich:
        entity.update(
            velocity=[100.0, -50.0],
            public_target_id=1,
            attack={"phase": "windup", "cooldown_remaining_ms": 100},
            effects=(
                {
                    "effect_id": "buff:1",
                    "active": True,
                    "remaining_ms": 2_000,
                    "stacks": 2,
                    "magnitude": 0.5,
                    "source_owner": 0,
                },
            ),
            causal_relation={"kind": "deploy", "source_card_id": 26_000_000},
        )
    return {
        "profile": "full" if rich else "core",
        "match": {"phase": "normal", "remaining_ms": 180_000, "elixir_multiplier": 1.0},
        "players": [
            {
                "owner": 0,
                "relation": "self",
                "crowns": 0,
                "self_private": {
                    "elixir": 5.0,
                    "hand": [26_000_000, 26_000_001, 26_000_002, 26_000_003],
                    "next_card": 26_000_004,
                },
                "current_evolution": [],
            },
            {"owner": 1, "relation": "opponent", "crowns": 0},
        ],
        "towers": [
            {
                "entity_id": 1,
                "owner": 0,
                "tower_kind": "king",
                "position": [9_000.0, 30_000.0],
                "hitpoints": 4_000.0,
                "max_hitpoints": 4_000.0,
                "tower_troop_id": 159_000_000,
                "active": True,
                "shield": {"hitpoints": 0.0},
            }
        ],
        "entities": [entity],
    }


def action_mask_payload() -> dict[str, object]:
    placement = [[False] * 18 for _ in range(32)]
    placement[10][7] = True
    placement[11][8] = True
    return {
        "hand_slots": [False, True, False, True],
        "placement_masks": {
            "1": {
                "visible_card_id": 26_000_001,
                "effective_card_id": 26_000_001,
                "effective_cost": 3.0,
                "form_code": 0,
                "footprint_width_tiles": 2.0,
                "footprint_height_tiles": 3.0,
                "model_subcell_offset": [0.5, -0.5],
                "row_major": placement,
            },
            "3": {
                "visible_card_id": 26_000_003,
                "effective_card_id": 26_000_003,
                "effective_cost": 4.0,
                "form_code": 0,
                "row_major": placement,
            },
        },
        "reasons": {"effective_elixir": 5.0},
    }
