# Model card: recurrent imitation-learning baseline

## Model and ownership

The compatible policy uses FirstLight V4's card, tower, entity and spatial
encoders, relation processing, autoregressive action head, and recurrent state.
ElixirEcho supplies actor-fair input selection, a public temporal context,
sequence boundaries, label masks, imitation-loss wiring, and checkpoint
verification. The formal historical checkpoint was trained in the earlier
local research project using its fingerprinted input contract. ElixirEcho
verifies compatibility with that contract; its current richer input mode is a
separate experiment. The weights are not redistributed here.

## Intended use

Research into replay-based imitation learning and controlled comparison of
semantic input choices. This is not an official Clash Royale system and is
not presented as a production player.

## Evidence and limitations

The tiny-overfit and one-episode checks establish that the computational
chain reaches loss and backpropagation. The old-project compatibility bridge
uses a selected sealed episode, Layer 2 shard, and formal checkpoint.

### Historical checkpoint: sealed test split

The read-only [legacy evaluation procedure](reproducibility.md#read-only-evaluation-of-a-historical-checkpoint)
ran the original project's recurrent V4 loss on its full sealed test split.
The checkpoint SHA-256 is
`49a76994020c49e60b3b7d90b7eaf7cc53ffaef3c679f3caa4130cb5c0c71548`;
the cache manifest SHA-256 is
`a3682bbc911b82ad33b7c2f73dd829895f6589c228e2b5dd06b299a2fe623a10`.
The loader verified the checkpoint's training and execution fingerprints
against that manifest. Test episode IDs are disjoint from train and validation.

| Measure | Full test result |
| --- | ---: |
| Complete episodes / actor sequences | 38 / 76 |
| Decision frames | 67,894 |
| Teacher-forced total IL loss | 6.0717 |
| WAIT labels | 64,792 |
| Single play-card labels | 2,920 |
| Single ability / multi-action / unrepresentable labels | 152 / 1 / 29 |
| Supervised gate / candidate / target decisions | 67,712 / 2,920 / 2,920 |

The gate, candidate, and target head losses were 0.7390, 1.1586, and 4.3410
respectively. Delay and value had zero supervised decisions in this run.
WAIT accounts for most frames; this loss alone cannot show whether the policy
makes useful play-card decisions. Action accuracy, gameplay win rate, and a
declared baseline comparison are still required for a model-quality claim.
No competitive win rate is claimed here.

Ability and multi-action decisions, delay supervision, and value prediction
have separate coverage and loss masks. This historical baseline supervised
WAIT and single play-card labels. New results must name the data split,
checkpoint hash, input mode, supported-label denominator, and baseline.
