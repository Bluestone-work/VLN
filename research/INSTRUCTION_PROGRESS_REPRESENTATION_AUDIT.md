# Instruction-Progress Representation Audit

## Scope

This audit covers the frozen ETPNav decision path used by Gate B. It is a
representation diagnostic only. It does not change waypoint generation,
navigation logits, action execution, labels, or the Gate B split.

## Existing decision path

```text
instruction token ids
        |
        v
ETP `forward_txt()` -> txt_embeds [B, L, 768]
        |
        v
global graph encoder (language-conditioned cross-modal layers)
        |
        v
gmap_embeds [B, N, 768] -> global_sap_head -> graph logits [B, N]
        |
        v
native graph action and low-level controller
```

The `gmap_embeds` saved by the existing graph-option capture are the output
of `forward_navigation`, before `global_sap_head` scores the candidates. They
therefore are available before action execution and already include the
frozen instruction-conditioned graph context. Their shape is `[N, 768]` per
captured state. STOP is index zero and remains part of the candidate set.

The original capture did not serialize `txt_embeds` or attention maps. The
Gate C extraction therefore reconstructs `txt_embeds` from the same frozen
checkpoint and the BERT-index instruction tokens, offline and without
re-running navigation. This is equivalent to the deployed `forward_txt`
representation and is provenance-checked against the capture checkpoint.

## Representation inventory

| Representation | Source | Shape | Available before action | Language conditioned | Privileged/future data |
|---|---|---:|---|---|---|
| instruction tokens | dataset observation | `[L]` (padded to 80) | yes | n/a | no |
| `txt_embeds` | `GlocalTextPathNavCMT.forward_txt` | `[L, 768]` | yes | instruction only | no |
| `gmap_embeds` | global cross-modal encoder | `[N, 768]` | yes | yes | no |
| graph logit | `global_sap_head(gmap_embeds)` | `[N]` | yes | yes | no |
| history/visited flags | graph state | scalar fields | yes | no | no |
| attention maps | model internals | not captured | theoretically yes, not serialized | yes | no |

The graph logit is a learned function of the graph representation. F0 keeps
the already frozen geometry/logit/policy features. F1 adds only low-dimensional
statistics of the existing candidate `gmap_embeds`; F2 adds token-alignment
statistics derived from the frozen `txt_embeds` and the current graph state;
F3 adds candidate-to-current/suffix instruction compatibility.

## Exclusions

The following fields are excluded from all Gate C features: goal distance,
reference route, future collision or success, full-return metrics, endpoint
error after execution, primitive count after execution, and any outcome-derived
instruction segmentation. AMBIGUOUS labels remain in held-out evaluation but
are excluded from binary model fitting, exactly as in Gate B.

## Frozen feature data flow

```text
instruction ids -> frozen `forward_txt` -> token vectors / progress statistics
                                      \
candidate graph state -> frozen `gmap_embeds` -> F1/F3 compatibility
                                      |
                                      v
             nested F0 -> F1 -> F2 -> F3 selective diagnostic
```

## Provenance

The source graph captures use `release_r2r/ckpt.iter12000.pth`, R2R BERT
indices, seed 100, native ETPNav control, and the existing Gate B
train/val_unseen scene-disjoint cohorts. The exact paths, hashes, and Python
environment are recorded in the Gate C result directory.
