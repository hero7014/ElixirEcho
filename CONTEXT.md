# Replay-to-Policy Learning

This context describes how an agent learns decisions from expert game trajectories
without confusing the world, the information available to an actor, and the values
actually supplied to a policy.

## Game semantics

**Environment State**:
The complete underlying game state at one moment, including information that may be
hidden from one or both actors.
_Avoid_: Observation, frame

**Observation**:
The current game facts available to one actor at one decision moment. It excludes
history, legal-action constraints, expert labels, provenance, and hidden information.
_Avoid_: Environment state, model input, frame

**ActionMask**:
The set of actions legal for an actor at the same decision moment as an Observation.
It describes permission, not intent or expert behavior.
_Avoid_: Action space, expert action

**DecisionStep**:
One actor's atomic decision moment, aligning the current Observation and ActionMask
with the expert behavior and public change recorded for that moment.
_Avoid_: Frame, sample

**Trajectory**:
An ordered sequence of DecisionSteps from one continuous episode, with explicit start,
termination, and reset boundaries.
_Avoid_: Batch, dataset

## Temporal knowledge

**History**:
Past Observations, public events, and past actions ordered before the current decision.
It never contains future information or the current expert label.
_Avoid_: Memory, tracker

**PublicStateTracker**:
Facts that can be determined exactly from public History, such as revealed cards and
strictly known cycle information. Uncertain facts remain unknown.
_Avoid_: Belief, predictor

**TemporalContext**:
The history-derived facts deliberately made available for the current decision,
including public tracked state and the previous action when representable.
_Avoid_: History, recurrent state

## Policy learning

**PolicyInput**:
The selected current Observation information and TemporalContext actually presented
to a Policy, together with legal-action support.
_Avoid_: Observation, representation

**Representation**:
The learned latent description of the current decision situation produced inside a
Policy.
_Avoid_: Observation, policy input

**RecurrentState**:
The Policy's learned summary carried between consecutive DecisionSteps in one
Trajectory and cleared at a reset boundary.
_Avoid_: History, tracker, belief

**Policy**:
A decision rule that combines PolicyInput and RecurrentState to produce a distribution
over legal actions.
_Avoid_: Model, action

**ExpertAction**:
The behavior demonstrated by the expert at a DecisionStep and used as supervision.
No demonstrated action represents WAIT.
_Avoid_: ActionMask, policy output

**ImitationLoss**:
The discrepancy between the Policy's action distribution and the supported parts of
the ExpertAction.
_Avoid_: Reward, evaluation

**Checkpoint**:
A reproducible trained-policy artifact tied to its learning configuration, input
contract, and compatibility identity.
_Avoid_: Model source, training run

**Evaluation**:
Evidence measuring a Checkpoint's decision quality, coverage, and gameplay behavior
against declared baselines.
_Avoid_: Training loss, validation alone
