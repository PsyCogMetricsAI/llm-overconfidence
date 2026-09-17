"""Frozen A1--B7 source metrics; pure functions, no filesystem or target access."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata

from legacy_primitives import observations

METRIC_IDS = tuple([f'A{i}' for i in range(1, 6)] + [f'B{i}' for i in range(1, 8)])
SPLIT_IDS = ('full', 'half0', 'half1')


def prepare_ab_items(loglik, char_lens, gold, item_ids, n_options=None):
    """Return per-item arrays in the original raw order, retaining invalid rows.

    Legacy observations fix the valid mask and ell/c/y exactly. Extra probability
    functions use centered log scores, so an underflowed probability is never
    passed to log, or used to rank the gold option. Invalid derived values are NaN.
    """
    ll = np.asarray(loglik, dtype=np.float64)
    hh = np.asarray(char_lens, dtype=np.float64)
    ids = np.asarray(item_ids).astype(str)
    if ll.ndim != 2 or ids.shape != (ll.shape[0],):
        raise ValueError('item_ids_shape')
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('duplicate item ID')
    ell, c, y, valid, reasons, nr, probability_error = observations(ll, hh, gold, n_options)
    n = len(ids)
    out = dict(item_ids=ids, valid=valid, ell=ell, c=c, y=y,
               validity_reason_counts=reasons, n_options=nr,
               probability_sum_max_error=probability_error)
    for name in ('e', 'z', 'nll', 'gold_rank_depth', 'nonwinner_entropy'):
        out[name] = np.full(n, np.nan, dtype=np.float64)
    use = np.flatnonzero(valid)
    if not len(use):
        return out

    real = np.isfinite(hh[use]) & (hh[use] > 0)
    u = np.full(ll[use].shape, -np.inf, dtype=np.float64)
    np.divide(ll[use], hh[use], out=u, where=real)
    if not np.isfinite(u[real]).all():
        raise ValueError('finite_inputs_nonfinite_normalized_scores')
    g = np.asarray(gold)[use].astype(np.int64)
    ii = np.arange(len(use))
    winner = np.argmax(u, axis=1)
    gold_u = u[ii, g]
    centered = u - np.max(u, axis=1, keepdims=True)
    out['e'][use] = 1.0 - y[use]
    # Subtract the raw scores directly: do not subtract underflowed log(p).
    out['z'][use] = u[ii, winner] - gold_u
    out['nll'][use] = logsumexp(centered, axis=1) - centered[ii, g]

    indices = np.arange(ll.shape[1])[None, :]
    preceding = (u > gold_u[:, None]) | ((u == gold_u[:, None]) & (indices < g[:, None]))
    out['gold_rank_depth'][use] = preceding.sum(axis=1) / (nr[use] - 1)

    remaining = u.copy()
    remaining[ii, winner] = -np.inf
    remaining -= np.max(remaining, axis=1, keepdims=True)
    q = np.exp(remaining - logsumexp(remaining, axis=1, keepdims=True))
    logq = np.zeros_like(q)
    np.log(q, out=logq, where=q > 0)
    h = -np.sum(q * logq, axis=1)
    conditional = np.zeros(len(use), dtype=np.float64)
    multiclass = nr[use] > 2
    conditional[multiclass] = h[multiclass] / np.log(nr[use][multiclass] - 1)
    out['nonwinner_entropy'][use] = conditional
    return out


def compute_ab(items, subset_mask=None):
    """Compute the 12 metrics on valid items intersected with a Boolean subset.

    Return metric-ID keyed values/reasons and the actual valid denominator.
    An empty reason denotes a finite result. Missing values are NaN in memory;
    the serialization layer keeps a corresponding explicit reason.
    """
    valid = np.asarray(items['valid'], dtype=bool)
    if subset_mask is not None:
        subset = np.asarray(subset_mask)
        if subset.shape != valid.shape or subset.dtype.kind != 'b':
            raise ValueError('subset_mask_must_be_aligned_boolean')
        valid = valid & subset
    use = np.flatnonzero(valid)
    n = len(use)
    values = {k: float('nan') for k in METRIC_IDS}
    reasons = {k: 'NO_VALID_ITEMS' for k in METRIC_IDS}
    out = dict(values=values, reasons=reasons, n_items=n)
    if not n:
        return out

    def put(name, value):
        value = float(value)
        values[name] = value if np.isfinite(value) else float('nan')
        reasons[name] = '' if np.isfinite(value) else 'NONFINITE_RESULT'

    c = np.asarray(items['c'], dtype=np.float64)[use]
    y = np.asarray(items['y'], dtype=bool)[use]
    if not np.isfinite(c).all() or np.any((c < 0) | (c > 1)):
        raise ValueError('invalid_prepared_confidence')
    ids = np.asarray(items['item_ids']).astype(str)[use]
    e = 1.0 - y
    n1 = int(y.sum())
    n0 = n - n1
    if n1 and n0:
        ranks = rankdata(c, method='average')
        put('A1', (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    else:
        reasons['A1'] = 'SINGLE_CLASS'
    order = np.lexsort((ids, -c))
    ranked_errors = e[order]
    put('A2', np.mean(np.cumsum(ranked_errors) / np.arange(1, n + 1)))
    bins = np.minimum(np.floor(10 * c).astype(np.int64), 9)
    count = np.bincount(bins, minlength=10)
    correct = np.bincount(bins, weights=y.astype(float), minlength=10)
    occupied = count > 0
    put('A3', np.sum(count[occupied] / n * (correct[occupied] / count[occupied] - y.mean()) ** 2))
    put('A4', np.mean(ranked_errors[:int(np.ceil(0.10 * n))]))
    put('A5', np.mean(ranked_errors[:int(np.ceil(0.25 * n))]))
    put('B1', np.mean(np.asarray(items['nll'])[use]))
    put('B2', np.mean(e * c))
    put('B3', np.mean(e * c ** 2))
    z = np.asarray(items['z'], dtype=np.float64)[use]
    put('B4', np.mean(z))
    zorder = np.lexsort((ids, -z))
    put('B5', np.mean(z[zorder[:int(np.ceil(0.10 * n))]]))
    put('B6', np.mean(np.asarray(items['gold_rank_depth'])[use]))
    put('B7', np.mean(np.asarray(items['nonwinner_entropy'])[use]))
    return out


def compute_ab_splits(items, source_half):
    """Compute full/H0/H1 after caller aligns source_half to items.item_ids."""
    half = np.asarray(source_half)
    if half.shape != np.asarray(items['item_ids']).shape or not np.isin(half, (0, 1)).all():
        raise ValueError('source_half_must_be_aligned_0_or_1')
    return dict(full=compute_ab(items), half0=compute_ab(items, half == 0),
                half1=compute_ab(items, half == 1))
