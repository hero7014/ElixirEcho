from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from elixir_echo.integrations.firstlight.trainer import FirstLightILTrainer
from elixir_echo.training.config import TrainingConfig


class FakeCheckpointModule:
    CHECKPOINT_SCHEMA_V4 = "universal-card-actor-critic-checkpoint.v4"

    def __init__(self):
        self.save_calls = []
        self.load_calls = []

    @staticmethod
    def checkpoint_contract(_model):
        return {"policy": "v4", "catalog": "fixture-catalog"}

    def save_actor_critic_checkpoint(self, path, model, **kwargs):
        destination = Path(path)
        destination.write_bytes(b"fake-v4-checkpoint")
        self.save_calls.append((destination, model, kwargs))
        self.checkpoint_id = "checkpoint-id-7"
        return "checkpoint-id-7"

    def load_actor_critic_checkpoint(self, path, model, **kwargs):
        self.load_calls.append((Path(path), model, kwargs))
        return {
            "extra": self.save_calls[0][2]["extra"],
            "update_step": self.save_calls[0][2]["update_step"],
            "checkpoint_id": self.checkpoint_id,
        }


def _trainer(config: TrainingConfig | None = None):
    module = FakeCheckpointModule()
    model = object()
    runtime = SimpleNamespace(import_module=lambda _name: module)
    backend = SimpleNamespace(runtime=runtime, model=model, device="cpu")
    return FirstLightILTrainer(backend, config or TrainingConfig()), module, model


def _fake_payload(trainer, module):
    kwargs = module.save_calls[0][2]
    return {
        "schema": module.CHECKPOINT_SCHEMA_V4,
        "contract": module.checkpoint_contract(None),
        "checkpoint_id": module.checkpoint_id,
        "update_step": kwargs["update_step"],
        "extra": kwargs["extra"],
    }


def test_checkpoint_round_trip_binds_config_contract_optimizer_and_step(tmp_path, monkeypatch):
    trainer, module, model = _trainer()
    optimizer = object()
    destination = tmp_path / "checkpoint-0007"

    saved = trainer.save_checkpoint(destination, optimizer=optimizer, update_step=7)
    monkeypatch.setattr(
        trainer, "_read_checkpoint_payload", lambda _path: _fake_payload(trainer, module), raising=False
    )
    restored = trainer.restore_checkpoint(destination, optimizer=optimizer)

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_file = destination / "firstlight-v4.pt"
    assert saved.checkpoint_id == "checkpoint-id-7"
    assert saved.update_step == 7
    assert manifest["training_config_fingerprint"] == trainer.config.fingerprint()
    assert manifest["provider_contract"] == {"policy": "v4", "catalog": "fixture-catalog"}
    assert manifest["checkpoint_sha256"] == hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    assert module.save_calls[0][2]["optimizer"] is optimizer
    assert module.save_calls[0][2]["update_step"] == 7
    assert module.load_calls[0][1] is model
    assert module.load_calls[0][2]["optimizer"] is optimizer
    assert module.load_calls[0][2]["restore_rng"] is True
    assert restored.checkpoint_id == "checkpoint-id-7"
    assert restored.update_step == 7


def test_checkpoint_rejects_config_mismatch_before_loading_model(tmp_path):
    source_trainer, _, _ = _trainer(TrainingConfig(tbptt_length=32))
    destination = tmp_path / "checkpoint-0007"
    source_trainer.save_checkpoint(destination, update_step=7)
    incompatible, module, _ = _trainer(TrainingConfig(tbptt_length=16))

    with pytest.raises(ValueError, match="training configuration fingerprint"):
        incompatible.restore_checkpoint(destination)

    assert module.load_calls == []


def test_checkpoint_rejects_tampering_before_loading_model(tmp_path):
    trainer, module, _ = _trainer()
    destination = tmp_path / "checkpoint-0007"
    trainer.save_checkpoint(destination, update_step=7)
    (destination / "firstlight-v4.pt").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checkpoint file hash"):
        trainer.restore_checkpoint(destination)

    assert module.load_calls == []


def test_checkpoint_rejects_manifest_identity_mismatch_before_mutating_model_or_optimizer(
    tmp_path, monkeypatch
):
    trainer, module, _ = _trainer()
    destination = tmp_path / "checkpoint-0007"
    trainer.save_checkpoint(destination, update_step=7)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["update_step"] = 8
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload = _fake_payload(trainer, module)
    monkeypatch.setattr(trainer, "_read_checkpoint_payload", lambda _path: payload, raising=False)

    with pytest.raises(ValueError, match="update_step does not match"):
        trainer.restore_checkpoint(destination)

    assert module.load_calls == []


def test_checkpoint_directory_is_not_overwritten(tmp_path):
    trainer, module, _ = _trainer()
    destination = tmp_path / "checkpoint-0007"
    trainer.save_checkpoint(destination, update_step=7)

    with pytest.raises(FileExistsError):
        trainer.save_checkpoint(destination, update_step=8)

    assert len(module.save_calls) == 1
