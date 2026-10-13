# Directional semantic oracle v1

This is an additional audit annotation. It does not modify R2R-CE or any
official dataset annotation.

The frozen sample is defined by `scene_split.json`, `sampling_manifest.json`,
and `ANNOTATION_PROTOCOL.md`. Open `index.html` in a browser, or serve this
directory with:

```bash
python -m http.server 8765 --directory research/annotations/directional_semantic_oracle_v1
```

The page shows only the current pre-action RGB sector, instruction, candidate
phrases, and Qwen preannotation. Qwen output is an editing aid and is not a
ground-truth label. It never shows endpoint images, goal/reference route,
BENEFIT/HARM labels, evaluator metrics, or final success.

Use `Save JSONL`, then place the downloaded file at
`research/annotations/directional_semantic_oracle_v1/annotations.jsonl`.
The evaluator rejects missing rows, duplicate keys, image hash mismatches, and
scene/node/bearing mismatches before it reads any causal outcome.

The development scene is for checking the UI and ontology. The three test
scenes are held out for the oracle feasibility result; do not change the label
definitions after test annotation begins.
