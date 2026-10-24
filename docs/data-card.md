# Data card

## Source and purpose

The optional data source is `VanguardX101/IL_Replay`, an anonymized replay
dataset intended for imitation-learning and offline game research. This
repository does not include its Parquet files. Its dataset card does not state
a redistributable license in the local copy reviewed for this release; obtain
and use the data under its own terms.

## Selection and transformation

The capture entry point selects an exact `replay_tag` from the `replays`
configuration. Source-only preflight rejects missing gameplay actions, actions
before the first decision window, more than two actions in one actor/window,
and ambiguous player orientation. FirstLight then prepares and calibrates the
replay. Upstream reconstruction failures are distinct from ElixirEcho semantic
or storage failures. Rejected replay identities are retained in reports; a
different replay is never used silently.

At each five-tick decision boundary from tick 90, the actor's current
Observation, ActionMask, public event delta, and expert label are atomically
paired. The current label and its effects are excluded from the current model
input. EpisodeContext decks exist only for reproduction/validation. Training
readers do not expose them.

## Limitations

Replay reconstruction depends on the FirstLight version, game assets, and
Native calibration. A source replay can be rejected before reaching the
ElixirEcho Adapter. Optional low-frequency mechanics may remain unobserved in
small samples; absence in one sample is not evidence that a field is unused.
Historical training and checkpoint counts belong to the earlier local project
until a new ElixirEcho run independently reproduces them.
