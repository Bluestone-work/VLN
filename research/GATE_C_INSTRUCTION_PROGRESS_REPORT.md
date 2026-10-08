# Gate C: Instruction-Progress Representation Diagnostic

## Question

Gate B found no high-precision selective intervention signal in geometry/logit
features. Gate C tested the narrower hypothesis that the missing signal is
instruction execution progress and candidate semantic compatibility.

## Frozen data and audit note

The authoritative Gate B CSV was used unchanged: 1,377 native-relative rows
with `INTERVENE=76`, `KEEP=533`, and `AMBIGUOUS=768`, split by train versus
held-out val_unseen scenes. The CSV contains 107 unique `(episode_id,
high_level_step)` join keys although the historical Gate B summary records 117
states. This is a bookkeeping discrepancy in the inherited report, not a row
drop: all 1,377 rows joined to their graph capture and all class counts match.
The Gate C result therefore preserves the exact corrected samples and labels,
and reports the observed join-key count explicitly.

The graph captures contain the frozen 768-D `gmap_embeds` for every native and
alternative option. Token representations were reconstructed offline with
ETPNav `forward_txt`, using the `release_r2r/ckpt.iter12000.pth` `vln_bert`
state over the same base ETP pretrain. No simulator rollout, action, waypoint
predictor, or controller was changed.

## Nested features

| Level | Added information | Dimensions |
| --- | --- | ---: |
| F0 | Exact Gate B geometry/logit/policy features | 19 |
| F1 | Frozen native/alternative graph embedding norms, cosine, difference and distribution summaries | 31 |
| F2 | Frozen token alignment progress: expected/peak position, entropy, prefix/suffix mass, progress margin | 37 |
| F3 | Candidate current/suffix instruction compatibility and alternative-minus-native deltas | 52 |

AMBIGUOUS rows were excluded from binary fitting and retained in all held-out
decision evaluation. Models were logistic regression, ridge linear classifier,
and a 16-unit MLP. The only thresholds were 0.50, 0.70, 0.80, 0.90, and 0.95.

## Held-out results

The table below uses the primary logistic model; every model and threshold is
available in `results/gate_c_instruction_progress/run_003/summary.json` and
`experiment_table.csv`.

| Level | Threshold | Coverage | Precision | AMBIGUOUS intervention rate | Positive scenes |
| --- | ---: | ---: | ---: | ---: | ---: |
| F0 | 0.95 | 0.641 | 0.366 | 0.634 | 6 |
| F1 | 0.95 | 0.422 | 0.407 | 0.593 | 5 |
| F2 | 0.95 | 0.359 | 0.304 | 0.696 | 5 |
| F3 | 0.95 | 0.359 | 0.435 | 0.565 | 5 |

The best non-baseline region across all fixed levels/models/thresholds is F3
logistic at threshold 0.90: precision `0.462`, coverage `0.406`, and
AMBIGUOUS intervention rate `0.538`. F2 peaks at precision `0.432`; no F2/F3
region reaches the preregistered precision `0.70`
with nonzero coverage. The apparent zero KEEP harm is not a safety result,
because the inherited cohort contains only native-failure routes and cannot
estimate native-success destruction.

## Mechanism check

The frozen progress statistic has nearly identical mean expected token position
for INTERVENE (`0.5002`), KEEP (`0.5017`), and AMBIGUOUS (`0.5017`). The
alternative-minus-native suffix compatibility means are also close:
INTERVENE `0.0061`, KEEP `0.0017`, AMBIGUOUS `-0.0011`. Thus the data do not
show a mechanism in which beneficial alternatives consistently advance a
different instruction segment. The semantic additions change confidence and
coverage, but do not separate beneficial from incomparable long-horizon
continuations.

## Decision

**NO-GO for instruction-progress selective intervention.** The pre-registered
Gate C conditions fail: F2/F3 do not produce a precision-at-least-0.70,
nonzero-coverage region; ambiguity remains dominant; and the strongest signal
comes from F1 rather than explicit progress features. This is a feasibility
diagnostic, not a navigation SR claim.

Do not run the single-intervention online test, semantic feature expansion,
waypoint retraining, PPO/RL, VLM/LLM, adaptive action density, or collision
trigger from this cohort. Gate A remains an oracle-only proposal-coverage
finding; Gate B and Gate C close the deployable selective-intervention branch.

## Artifacts

- `research/INSTRUCTION_PROGRESS_REPRESENTATION_AUDIT.md`
- `research/GATE_C_INSTRUCTION_PROGRESS_PROTOCOL.md`
- `research/results/gate_c_instruction_progress/run_003/summary.json`
- `research/results/gate_c_instruction_progress/run_003/per_sample.csv`
- `research/results/gate_c_instruction_progress/run_003/heldout_predictions.csv`
- `research/results/gate_c_instruction_progress/per_route.csv`
- `research/results/gate_c_instruction_progress/per_scene.csv`
- `research/results/gate_c_instruction_progress/figures/`
- `research/results/gate_c_instruction_progress/GO_NO_GO_SUMMARY.json`
