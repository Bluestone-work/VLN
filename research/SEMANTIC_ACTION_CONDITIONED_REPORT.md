# Semantic Action-Conditioned Replay Report

The first action-conditioned semantic capture was run on two episodes from the
frozen val-unseen semantic alignment smoke. It replayed 19 recorded native ETPNav
decisions in isolated branches and saved the post-controller RGB-D endpoint for
each decision.

Replay fidelity passed completely:

```text
endpoint / primitive / collision / RNG / terminal / sensor checks: 19 / 19
max endpoint error: 0.0 m
max rotation error: 4.21e-8 rad
max progress error: 0.0 m
```

This resolves the previous limitation that only the current panorama was
available. The saved endpoint image is now action-conditioned and can be paired
with the exact native action identity and instruction. It still does not answer
whether an alternative native ghost would be better: the smoke replays the
recorded selected action only, and no semantic model or intervention was trained.

The result is therefore **capture GO / semantic-method NO-GO pending enumeration**.
The next experiment is to enumerate all admissible native ghosts at a small,
pre-registered set of states, save each endpoint observation, and compare
instruction-grounded evidence across actions. Any branch failing the selected
action sentinel invalidates the corresponding state and must be excluded rather
than repaired post hoc.
