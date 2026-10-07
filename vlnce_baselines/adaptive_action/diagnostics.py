"""Append-only JSONL diagnostics for action-abstraction experiments."""

import json
import math
import os


def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


class DecisionLogger(object):
    def __init__(self, path, rank=0):
        self.path = path
        self.rank = int(rank)
        self.count = 0
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

    def write(self, record):
        if not self.path:
            return
        payload = dict(record)
        payload["writer_rank"] = self.rank
        payload["diagnostic_index"] = self.count
        with open(self.path, "a") as stream:
            stream.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
        self.count += 1
