#!/usr/bin/env python3
"""Canonical, container-independent array identity for `source_arrays.npz`.

The archived raw phase compared the freshly rebuilt `source_arrays.npz` with the withheld
117,410,167-byte original-input baseline array by array (`atol=0`, NaN-equal). The public
release replaces that private file with a small identity record carrying the same array-level
assertions. NPZ container framing and compression are never compared: only the array payload
and its metadata are, through the fixed canonical form below.

Canonicalisation (frozen; also recorded inside the emitted JSON):

    digest = sha256( dtype.str | 0x00 | shape as little-endian int64 | C-order contiguous bytes )

Big-endian numeric arrays are converted to little-endian first, so the digest does not depend on
the byte order of the machine that wrote the file. Object dtype is never allowed.

Used by `tools/make_array_identity.py` (generation from the verified original) and by
`core_reproduction/run_core.py --baseline identity` (verification of a fresh rebuild).
"""
from __future__ import annotations

import hashlib

import numpy as np

SCHEMA = "llm-overconfidence/array-identity/1"
CANONICALISATION = (
    "sha256(dtype.str | 0x00 | shape as little-endian int64 | C-order contiguous bytes); "
    "big-endian numeric arrays converted to little-endian; npz container framing/compression never compared"
)


def canonical_array(a) -> np.ndarray:
    """Return the canonical C-contiguous little-endian view/copy of `a`."""
    a = np.asarray(a)
    if a.dtype.hasobject:
        raise ValueError("object dtype is not supported by the canonical array identity")
    if a.dtype.kind in "iufc" and a.dtype.itemsize > 1 and a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(a)


def array_digest(a) -> str:
    a = canonical_array(a)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode("ascii"))
    h.update(b"\x00")
    h.update(np.asarray(a.shape, dtype="<i8").tobytes())
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def array_record(a) -> dict:
    a = np.asarray(a)
    rec = {"shape": [int(x) for x in a.shape], "dtype": a.dtype.str, "digest": array_digest(a)}
    if a.dtype.kind in "fc":
        with np.errstate(invalid="ignore"):
            rec["nan"] = int(np.isnan(a).sum())
        rec["posinf"] = int(np.isposinf(a).sum())
        rec["neginf"] = int(np.isneginf(a).sum())
    if a.dtype.kind in "iu":
        rec["min"] = int(a.min()) if a.size else None
        rec["max"] = int(a.max()) if a.size else None
    return rec


def identity_record(npz_path, *, npz_sha256: str, npz_bytes: int, recorded_utc: str, note: str) -> dict:
    """Build the identity record for a verified original `source_arrays.npz`."""
    with np.load(npz_path, allow_pickle=False) as z:
        keys = list(z.files)
        arrays = {k: array_record(z[k]) for k in keys}
        n_models = int(len(z["model_ids"])) if "model_ids" in keys else None
        n_items = int(len(z["item_ids"])) if "item_ids" in keys else None
    return {
        "schema": SCHEMA,
        "recorded_utc": recorded_utc,
        "source": note,
        "source_sha256": npz_sha256,
        "source_bytes": int(npz_bytes),
        "canonicalisation": CANONICALISATION,
        "exact_zip_bytes_compared": False,
        "keys": keys,
        "n_models": n_models,
        "n_items": n_items,
        "arrays": arrays,
    }


def check_arrays_against_identity(npz_path, identity: dict) -> dict:
    """Fail-closed per-array check of a freshly rebuilt npz against the identity record.

    Digest equality over the canonical byte image is exact array equality (`atol=0`) for every
    key, including the NaN/+-inf masks; shape and dtype are asserted separately for clear errors.
    """
    assert identity.get("schema") == SCHEMA, ("UNSUPPORTED_IDENTITY_SCHEMA", identity.get("schema"))
    want_all = identity["arrays"]
    checks = {}
    with np.load(npz_path, allow_pickle=False) as z:
        keys = list(z.files)
        missing = sorted(set(want_all) - set(keys))
        extra = sorted(set(keys) - set(want_all))
        assert not missing and not extra, ("ARRAY_KEY_SET_MISMATCH", missing, extra)
        for name in keys:
            arr = z[name]
            want = want_all[name]
            assert [int(x) for x in arr.shape] == [int(x) for x in want["shape"]], (
                "ARRAY_SHAPE_MISMATCH", name, [int(x) for x in arr.shape], want["shape"])
            assert arr.dtype.str == want["dtype"], ("ARRAY_DTYPE_MISMATCH", name, arr.dtype.str, want["dtype"])
            got = array_digest(arr)
            assert got == want["digest"], ("ARRAY_DIGEST_MISMATCH", name, got, want["digest"])
            rec = {"shape": [int(x) for x in arr.shape], "dtype": arr.dtype.str, "digest": got}
            if arr.dtype.kind in "fc":
                with np.errstate(invalid="ignore"):
                    rec["nan"] = int(np.isnan(arr).sum())
                rec["posinf"] = int(np.isposinf(arr).sum())
                rec["neginf"] = int(np.isneginf(arr).sum())
            checks[name] = rec
        n_models = int(len(z["model_ids"])) if "model_ids" in z.files else None
        n_items = int(len(z["item_ids"])) if "item_ids" in z.files else None
    if identity.get("n_models") is not None:
        assert n_models == identity["n_models"], ("N_MODELS_MISMATCH", n_models, identity["n_models"])
    if identity.get("n_items") is not None:
        assert n_items == identity["n_items"], ("N_ITEMS_MISMATCH", n_items, identity["n_items"])
    return checks
