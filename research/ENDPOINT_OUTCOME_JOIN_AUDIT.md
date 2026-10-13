# Endpoint/Outcome Join Audit

The native endpoint enumeration now contains action-conditioned endpoint RGB-D
records, but the available `dense_single_override_001/single_override_pairs.jsonl`
contains zero causal pairs. Therefore no endpoint record can currently be labeled
BENEFIT, HARM, or UNCERTAIN by exact state/action identity.

The join is explicitly blocked. No classifier, threshold, or semantic claim is
trained from unmatched records. The next required data collection is a native
single-action full-return continuation with exact state and action signatures,
after which this audit can be rerun without changing the semantic protocol.
