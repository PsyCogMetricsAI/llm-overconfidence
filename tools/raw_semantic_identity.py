"""Exact per-array identities for acquired NPZs, independent of ZIP metadata."""
import gzip
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core_reproduction'))
from array_identity import array_record

DEFAULT = Path(__file__).resolve().parents[1] / 'upstream_metadata/RAW_SEMANTIC_IDENTITY.json.gz'


def load(path=DEFAULT):
    with gzip.open(path, 'rt') as f:
        value = json.load(f)
    if value['schema'] != 'llm-overconfidence/raw-semantic-identity/1':
        raise ValueError('Unknown raw semantic identity schema')
    return value


def verify(arrays, expected):
    if set(arrays) != set(expected['arrays']):
        raise ValueError('RAW_SEMANTIC_KEYS_MISMATCH')
    result = {}
    for k, want in expected['arrays'].items():
        got = array_record(np.asarray(arrays[k]))
        for field in ('shape', 'dtype', 'digest'):
            if got[field] != want[field]:
                raise ValueError(f'RAW_SEMANTIC_{field.upper()}_MISMATCH: {k}')
        result[k] = got
    return result
