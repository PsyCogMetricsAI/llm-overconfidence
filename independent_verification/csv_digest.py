#!/usr/bin/env python3
"""Canonical value-level digest for comparison CSV tables.

The archived historical gates compared cold CSVs with the frozen originals row by row and cell by
cell (`compare_csv`, absolute tolerance 1e-10). Public summary mode replaces that comparison for the
50+50 feature tables with a canonical digest, because those tables are ~172 MB and are not bundled.

Canonical form (frozen): for every cell, in frozen header order and row order,

  * a cell that parses as a float is written as its Python `repr` (so `1.0`, `1.00`, `1e0` are the
    same value; `nan`/`NaN`/`-nan` normalise to `NaN`); non-finite values normalise to `inf`/`-inf`;
  * any other cell is written verbatim with a `S` prefix.

Cells are separated by 0x1f and rows by 0x1e; the header, row count and column count are included.
Digest equality therefore implies the archived numeric comparison passed: it is exact per-value
equality, which is strictly stronger than the 1e-10 tolerance check. A digest mismatch is a real
failure and is never skipped.
"""
from __future__ import annotations

import csv
import hashlib
import math

CELL_SEP = b"\x1f"
ROW_SEP = b"\x1e"


def _token(value: str) -> bytes:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return b"S" + str(value).encode("utf-8", "surrogateescape")
    if math.isnan(number):
        text = "NaN"
    elif math.isinf(number):
        text = "inf" if number > 0 else "-inf"
    else:
        text = repr(number)
    return b"N" + text.encode("ascii")


def canonical_csv_digest(path) -> dict:
    """Return {sha256, rows, columns, header} for the canonical form of `path`."""
    digest = hashlib.sha256()
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise AssertionError(f"empty CSV without header: {path}")
        digest.update(str(len(header)).encode() + ROW_SEP)
        digest.update(CELL_SEP.join(_token(name) for name in header))
        rows = 0
        for row in reader:
            if len(row) != len(header):
                raise AssertionError(f"ragged CSV row {rows} in {path}: {len(row)} != {len(header)}")
            digest.update(ROW_SEP)
            digest.update(CELL_SEP.join(_token(cell) for cell in row))
            rows += 1
    return {"sha256": digest.hexdigest(), "rows": rows, "columns": len(header), "header": header}
