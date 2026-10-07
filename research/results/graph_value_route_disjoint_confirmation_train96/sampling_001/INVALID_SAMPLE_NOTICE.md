# Sampling 001 invalidation notice

This 96-route cohort is retained as a pipeline pilot only. It is not valid route-disjoint confirmation evidence.

The complete exclusion audit found 12 selected routes overlapping the historical `ranker_embedding_train960_oracle.jsonl` diagnostic source, all in scene `gTV8FGcVJC9`. The original sampler excluded five main cohort manifests but omitted this historical ranker source and did not canonicalize `data/scene_datasets/` scene prefixes. No intervention outcome from this cohort may be used as confirmation evidence.

The exact overlap list and source hashes are recorded in `INVALID_OVERLAP_AUDIT.json`. A new sample will use the corrected sampler and the full exclusion inventory, with a fresh output directory and a new manifest hash.
