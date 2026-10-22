"""Read-only adapter from the project policy protocol to FirstLight V4."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from elixir_echo.integrations.firstlight.runtime import FirstLightRuntime
from elixir_echo.integrations.firstlight.tensorizer import load_v4_components
from elixir_echo.policies.protocol import PolicyInput, PolicyOutput, RecurrentState


CHECKPOINT_ENV = "ELIXIR_ECHO_FIRSTLIGHT_CHECKPOINT"


class FirstLightPolicyBackend:
    """Expose the frozen V4 actor as a provider-neutral recurrent policy backend."""

    def __init__(
        self,
        runtime: FirstLightRuntime,
        *,
        device: str | torch.device = "cpu",
        checkpoint: str | Path | None = None,
    ) -> None:
        self.runtime = runtime
        self.components = load_v4_components(runtime)
        self.device = torch.device(device)
        model_module = runtime.import_module("native_runner.training.v4.model")
        self.model = model_module.UniversalCardPolicyV4(
            self.components.bundle.card_catalog,
            ability_catalog=self.components.bundle.ability_catalog,
            entity_archetype_catalog=self.components.bundle.entity_archetype_catalog,
            effect_catalog=self.components.bundle.effect_catalog,
            mechanic_profile_catalog=self.components.bundle.mechanic_profile_catalog,
            config=self.components.config,
        ).to(self.device)
        checkpoint_path = Path(checkpoint) if checkpoint is not None else (
            Path(os.environ[CHECKPOINT_ENV]) if os.environ.get(CHECKPOINT_ENV) else None
        )
        self.checkpoint_path = checkpoint_path
        if checkpoint_path is not None:
            checkpoint_module = runtime.import_module("native_runner.training.v4.checkpoint")
            checkpoint_module.load_actor_critic_checkpoint(
                checkpoint_path,
                self.model,
                map_location=self.device,
                restore_rng=False,
            )

    def initial_state(self, batch_size: int = 1) -> RecurrentState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return RecurrentState(
            payload=self.model.initial_state(batch_size, device=self.device),
            batch_size=batch_size,
        )

    def step(self, policy_input: PolicyInput, state: RecurrentState) -> PolicyOutput:
        if state.batch_size != 1:
            raise ValueError("single-decision PolicyInput requires recurrent batch_size=1")
        if state.episode_id is not None and state.episode_id != policy_input.identity.episode_id:
            raise ValueError("recurrent state crossed an episode boundary without reset")
        if state.last_identity is not None and state.last_identity.actor != policy_input.identity.actor:
            raise ValueError("recurrent state crossed an actor boundary")
        if state.last_identity is not None and policy_input.identity.tick <= state.last_identity.tick:
            raise ValueError("recurrent-state tick must advance within an actor episode")
        if state.payload is None:
            state = self.initial_state()
        decision = policy_input.tensors
        if decision.actor != policy_input.identity.actor:
            raise ValueError("tensorized actor differs from PolicyInput identity")
        batch = decision.batch.to_model_input(self.device)
        context = self.model.forward(
            batch,
            state.payload,
            episode_start=torch.zeros(1, dtype=torch.bool, device=self.device),
            validate=True,
        )
        gate_logits = self.model.gate_head(context.policy_context)
        decoder_base = self.model.decoder_initial(context.policy_context)
        decoder_context = self.model._decoder_context(decoder_base, context.encoded)
        candidate_logits = self.model.candidate_logits(
            decoder_context,
            context.encoded.candidate_memory,
        )
        return PolicyOutput(
            action_logits={"gate": gate_logits, "candidate": candidate_logits},
            next_state=RecurrentState(
                payload=context.next_state,
                batch_size=1,
                episode_id=policy_input.identity.episode_id,
                last_identity=policy_input.identity,
            ),
            value=context.value,
        )
