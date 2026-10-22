from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from elixir_echo.domain import DecisionIdentity
from elixir_echo.integrations.firstlight.policy import FirstLightPolicyBackend
from elixir_echo.integrations.firstlight.runtime import FirstLightRuntime
from elixir_echo.integrations.firstlight.tensorizer import ObservationTensorizer
from elixir_echo.policies.protocol import PolicyInput
from tests.fixtures.firstlight_inputs import action_mask_payload as _mask
from tests.fixtures.firstlight_inputs import observation_payload as _profile


def test_formal_checkpoint_policy_wrapper_matches_raw_v4_forward() -> None:
    root = os.environ.get("ELIXIR_ECHO_FIRSTLIGHT_ROOT")
    checkpoint = os.environ.get("ELIXIR_ECHO_FIRSTLIGHT_CHECKPOINT")
    if not root or not checkpoint:
        pytest.skip("set FirstLight root and checkpoint paths for checkpoint-backed wrapper compatibility")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = FirstLightPolicyBackend(FirstLightRuntime(root), device=device, checkpoint=checkpoint)
    backend.model.eval()
    tensorizer = ObservationTensorizer(backend.components)
    decision = tensorizer.tensorize(_profile(rich=True), actor=0, action_mask=_mask())
    policy_input = PolicyInput(
        identity=DecisionIdentity("checkpoint-parity", 90, 0),
        profile="full",
        tensors=decision,
    )
    state = backend.initial_state()

    with torch.inference_mode():
        wrapped = backend.step(policy_input, state)
        batch = decision.batch.to_model_input(backend.device)
        raw_context = backend.model.forward(
            batch,
            state.payload,
            episode_start=torch.zeros(1, dtype=torch.bool, device=backend.device),
            validate=True,
        )
        raw_gate = backend.model.gate_head(raw_context.policy_context)
        decoder = backend.model._decoder_context(
            backend.model.decoder_initial(raw_context.policy_context),
            raw_context.encoded,
        )
        raw_candidate = backend.model.candidate_logits(decoder, raw_context.encoded.candidate_memory)

    torch.testing.assert_close(wrapped.action_logits["gate"], raw_gate, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(wrapped.action_logits["candidate"], raw_candidate, rtol=1e-6, atol=1e-7)
    assert torch.isfinite(raw_gate).all()
    assert torch.isfinite(raw_candidate).all()
