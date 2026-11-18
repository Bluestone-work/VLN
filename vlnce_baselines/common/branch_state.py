"""Deterministic environment snapshots for counterfactual navigation.

CBR-GRPO must compare branches from exactly the same simulator, graph-map and
random-number state.  This module intentionally has no trainer dependency so
it can be tested with a tiny fake environment before touching Habitat.
"""

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import pickle
import random

import numpy as np
import torch


@dataclass
class BranchState:
    env_state: object
    gmap: object
    prev_vp: object
    python_rng: object
    numpy_rng: object
    torch_rng: object
    cuda_rng: object
    fingerprint: str


def _digest(value):
    """Return a stable digest for common Python/numpy state objects."""
    try:
        payload = pickle.dumps(value, protocol=4)
    except Exception:
        payload = repr(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_fingerprint(env_state, gmap, prev_vp):
    return _digest((env_state, gmap, prev_vp))


def capture_branch_state(env, gmap, prev_vp):
    """Capture all mutable state needed to replay a candidate action.

    ``env`` may be a Habitat worker exposing ``capture_agent_state`` or a test
    double with the same method.  Failing loudly here prevents accidental
    pseudo-counterfactual training from an unrecoverable state.
    """
    capture = getattr(env, "capture_agent_state", None)
    if capture is None:
        raise TypeError("CBR environment must implement capture_agent_state()")
    env_state = deepcopy(capture())
    snapshot = BranchState(
        env_state=env_state,
        gmap=deepcopy(gmap),
        prev_vp=deepcopy(prev_vp),
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state().clone(),
        cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None,
        fingerprint="",
    )
    snapshot.fingerprint = state_fingerprint(
        snapshot.env_state, snapshot.gmap, snapshot.prev_vp
    )
    return snapshot


def restore_branch_state(env, snapshot):
    """Restore a snapshot and return a verification report.

    Restoration of the simulator is intentionally delegated to the R1Env API;
    silently falling back to teleporting without that API would invalidate the
    branch comparison.
    """
    restore = getattr(env, "restore_agent_state", None)
    if restore is None:
        raise TypeError("CBR environment must implement restore_agent_state()")
    restore(deepcopy(snapshot.env_state))
    random.setstate(snapshot.python_rng)
    np.random.set_state(snapshot.numpy_rng)
    torch.set_rng_state(snapshot.torch_rng)
    if snapshot.cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot.cuda_rng)
    restored = {
        "env_state": deepcopy(env.capture_agent_state()),
        "gmap": deepcopy(snapshot.gmap),
        "prev_vp": deepcopy(snapshot.prev_vp),
    }
    observed = state_fingerprint(
        restored["env_state"], restored["gmap"], restored["prev_vp"]
    )
    return {
        "ok": observed == snapshot.fingerprint,
        "expected": snapshot.fingerprint,
        "observed": observed,
        "state": restored,
    }


def restore_into(env, snapshot, gmap=None, prev_vp=None):
    """Restore simulator and return deep-copied graph/action state."""
    report = restore_branch_state(env, snapshot)
    if not report["ok"]:
        raise RuntimeError(
            "CBR branch restore verification failed: expected %s, got %s"
            % (report["expected"], report["observed"])
        )
    return deepcopy(snapshot.gmap), deepcopy(snapshot.prev_vp)

