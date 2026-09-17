"""Frozen C1-C4 / D1-D4 source-only mathematics; no file or target access.

All item axes use the frozen canonical source-item order. D direction h means
fit on half h, evaluate on half 1-h. Evaluation b is never restandardized.
"""
import numpy as np
from scipy.stats import rankdata

from legacy_primitives import als_reference, project_source

C_IDS = ('C1', 'C2', 'C3', 'C4')
D_IDS = ('D1', 'D2', 'D3', 'D4')
METRIC_IDS = C_IDS + D_IDS


def _empty(reason, n=0):
    return dict(values=np.full(4, np.nan), reasons=np.full(4, reason, dtype='U160'), n=int(n))


def reference_ease(y, valid, org_ids, reference_orgs, scored_orgs=()):
    """Return organization-equal Jeffreys-smoothed accuracy and cell counts."""
    y, valid = np.asarray(y), np.asarray(valid, bool)
    org_ids = np.asarray(org_ids).astype(str)
    ref, score = set(map(str, reference_orgs)), set(map(str, scored_orgs))
    if ref & score:
        raise ValueError('REFERENCE_SCORED_ORG_OVERLAP')
    if y.shape != valid.shape or y.ndim != 2 or len(org_ids) != len(y):
        raise ValueError('REFERENCE_EASE_SHAPE')
    if not ref or not ref <= set(org_ids):
        raise ValueError('REFERENCE_ORGS_MISSING')
    total = np.zeros(y.shape[1], np.int64)
    groups = np.zeros(y.shape[1], np.int64)
    sums = np.zeros(y.shape[1], np.float64)
    for org in sorted(ref):
        take = org_ids == org
        n = valid[take].sum(axis=0)
        successes = np.where(valid[take], y[take], 0).sum(axis=0)
        active = n > 0
        sums[active] += (successes[active] + .5) / (n[active] + 1.)
        total += n
        groups += active
    w = np.full(y.shape[1], np.nan)
    good = total >= 20
    w[good] = sums[good] / groups[good]
    return dict(w=w, n_reference=total, n_reference_orgs=groups)


def compute_c(c, y, z, valid, item_ids, w, item_mask=None):
    c, y, z, w = [np.asarray(v, np.float64) for v in (c, y, z, w)]
    ids = np.asarray(item_ids).astype(str)
    valid = np.asarray(valid, bool)
    if not all(v.shape == valid.shape for v in (c, y, z, w, ids)) or valid.ndim != 1:
        raise ValueError('C_SHAPE')
    if len(set(ids)) != len(ids):
        raise ValueError('DUPLICATE_ITEM_ID')
    use = valid & np.isfinite(w)
    if item_mask is not None:
        use &= np.asarray(item_mask, bool)
    # The frozen effective source-valid mask is authoritative, not repaired here.
    if not np.isfinite(c[use]).all() or not np.isfinite(z[use]).all() or not np.isfinite(y[use]).all():
        raise ValueError('NONFINITE_VALID_C_OBSERVATION')
    n = int(use.sum())
    out = _empty('NO_REFERENCE_VALID_ITEMS', n)
    if n == 0:
        return out
    cc, yy, zz, ww = c[use], y[use], z[use], w[use]
    out['values'][:3] = [np.mean(ww * (1 - yy)), np.mean(ww * (1 - yy) * cc), np.mean(ww * zz)]
    out['reasons'][:3] = ''
    if n < 4:
        out['reasons'][3] = 'REFERENCE_VALID_ITEMS_LT4'
    else:
        order = np.lexsort((ids[use], ww))
        v = (cc - yy)[order]
        bins = 4 * np.arange(n, dtype=np.int64) // n
        mean = v.mean()
        out['values'][3] = sum(np.mean(bins == b) * (v[bins == b].mean() - mean) ** 2 for b in range(4))
        out['reasons'][3] = ''
    bad = ~np.isfinite(out['values']) & (out['reasons'] == '')
    out['reasons'][bad] = 'NONFINITE_METRIC'
    out['values'][bad] = np.nan
    return out


def reference_parameters(ell, valid, b):
    """Recompute final reference a/s in the exact cached or new fit-half gauge."""
    ell, b = np.asarray(ell, float), np.asarray(b, float)
    mask = np.asarray(valid, bool) & np.isfinite(ell) & np.isfinite(b)[None, :]
    n = mask.sum(axis=1)
    if np.any(n < 2):
        raise ValueError('REFERENCE_MODEL_WITH_LT2_QUALIFIED_ITEMS')
    x = np.where(np.isfinite(b), b, 0.)
    e = np.where(mask, ell, 0.)
    mb = (mask * x).sum(axis=1) / n
    me = e.sum(axis=1) / n
    var = (mask * x * x).sum(axis=1) / n - mb * mb
    if np.any(var <= 1e-12):
        raise ValueError('DEGENERATE_REFERENCE_B_VARIANCE')
    s = -((e * x).sum(axis=1) / n - me * mb) / var
    a = me + s * mb
    if not np.isfinite(a).all() or not np.isfinite(s).all():
        raise ValueError('NONFINITE_REFERENCE_PARAMETERS')
    return a, s


def project_reference_items(ell, valid, a, s):
    ell = np.asarray(ell, np.float64)
    mask = np.asarray(valid, bool) & np.isfinite(ell)
    a, s = np.asarray(a, float), np.asarray(s, float)
    if ell.ndim != 2 or mask.shape != ell.shape or a.shape != (len(ell),) or s.shape != a.shape:
        raise ValueError('REFERENCE_PROJECTION_SHAPE')
    n = mask.sum(axis=0)
    den = (mask * s[:, None] ** 2).sum(axis=0)
    num = (np.where(mask, a[:, None] - ell, 0.) * s[:, None]).sum(axis=0)
    b = np.full(ell.shape[1], np.nan)
    good = (n >= 20) & (den > 1e-20) & np.isfinite(den) & np.isfinite(num)
    b[good] = num[good] / den[good]
    bad = ~np.isfinite(b)
    reason = np.full(ell.shape[1], '', dtype='U64')
    reason[n < 20] = 'REFERENCE_COUNT_LT20'
    reason[(n >= 20) & (den <= 1e-20)] = 'REFERENCE_DENOMINATOR_LE1E20'
    reason[bad & (reason == '')] = 'NONFINITE_REFERENCE_PROJECTION'
    b[bad] = np.nan
    return dict(b=b, n_reference=n, denominator=den, reasons=reason)


def fit_reference_direction(ell, valid, source_half, h, cached_b=None):
    """Fit one reference direction, or use an identity-verified half-0 b."""
    ell, valid, halves = np.asarray(ell, float), np.asarray(valid, bool), np.asarray(source_half)
    if h not in (0, 1) or ell.shape != valid.shape or ell.shape[1] != len(halves):
        raise ValueError('REFERENCE_DIRECTION_SHAPE')
    fit, evaluate = halves == h, halves == 1 - h
    if cached_b is None:
        b, diag = als_reference(ell[:, fit], valid[:, fit])
    else:
        if h != 0:
            raise ValueError('LEGACY_CACHE_ONLY_HALF0')
        b = np.asarray(cached_b, float).copy()
        eligible = (valid[:, fit] & np.isfinite(ell[:, fit])).sum(axis=0) >= 20
        if b.shape != (int(fit.sum()),) or not np.array_equal(np.isfinite(b), eligible):
            raise ValueError('LEGACY_B_ELIGIBLE_MISMATCH')
        if eligible.sum() < 100:
            raise ValueError('INSUFFICIENT_REFERENCE_ITEMS')
        diag = dict(reused_b=True, n_qualified_items=int(eligible.sum()))
    a, s = reference_parameters(ell[:, fit], valid[:, fit], b)
    projected = project_reference_items(ell[:, evaluate], valid[:, evaluate], a, s)
    b_fit, b_eval = np.full(ell.shape[1], np.nan), np.full(ell.shape[1], np.nan)
    b_fit[fit], b_eval[evaluate] = b, projected['b']
    return dict(b_fit=b_fit, b_eval=b_eval, a_reference=a, s_reference=s,
                eval_n_reference=projected['n_reference'], eval_denominator=projected['denominator'],
                eval_reasons=projected['reasons'], diagnostics=diag)


def residual_statistics(r, b):
    r, b = np.asarray(r, float), np.asarray(b, float)
    out = _empty('', len(r))
    if len(r) == 0 or not np.isfinite(r).all() or not np.isfinite(b).all():
        return _empty('NONFINITE_OR_EMPTY_RESIDUAL', len(r))
    den = float(np.sum(r ** 2))
    if den <= 1e-12:
        out['reasons'][0] = 'RESIDUAL_ENERGY_LE1E12'
    else:
        out['values'][0] = np.sum(np.maximum(-r, 0.) ** 2) / den
    centered = r - r.mean()
    m2 = float(np.mean(centered ** 2))
    if m2 <= 1e-12:
        out['reasons'][1] = 'RESIDUAL_M2_LE1E12'
    else:
        out['values'][1] = np.mean(centered ** 4) / m2 ** 2 - 3.
    for j, x, y in [(2, rankdata(abs(r), method='average'), rankdata(b, method='average')),
                    (3, r, b ** 2)]:
        x, y = x - x.mean(), y - y.mean()
        vx, vy = np.mean(x ** 2), np.mean(y ** 2)
        if vx <= 1e-12 or vy <= 1e-12:
            out['reasons'][j] = 'CORRELATION_VARIANCE_LE1E12'
        else:
            out['values'][j] = np.mean(x * y) / np.sqrt(vx * vy)
    bad = ~np.isfinite(out['values']) & (out['reasons'] == '')
    out['reasons'][bad] = 'NONFINITE_METRIC'
    out['values'][~np.isfinite(out['values'])] = np.nan
    return out


def compute_d_direction(ell, valid, source_half, h, b_fit, b_eval, reference_failure=''):
    if reference_failure:
        out = _empty('REFERENCE_FIT_FAILED:' + reference_failure)
        out.update(n_fit=0, parameters=np.full(5, np.nan))
        return out
    ell, valid, halves = np.asarray(ell, float), np.asarray(valid, bool), np.asarray(source_half)
    b_fit, b_eval = np.asarray(b_fit, float), np.asarray(b_eval, float)
    fitmask = valid & (halves == h)
    fit = project_source(ell, fitmask, b_fit)
    use = valid & np.isfinite(ell) & (halves == 1 - h) & np.isfinite(b_eval)
    parameters = np.array([fit.get(k, np.nan) for k in ('a', 's', 'sigma', 'theta', 'var_ell')])
    if not fit['valid_source_fit']:
        out = _empty('MODEL_FIT_FAILED:' + fit['exclusion_reason'], int(use.sum()))
    elif use.sum() < 100:
        out = _empty('EVALUATION_ITEMS_LT100', int(use.sum()))
    else:
        r = (ell[use] - fit['a'] + fit['s'] * b_eval[use]) / fit['sigma']
        out = residual_statistics(r, b_eval[use])
    out.update(n_fit=int(fit['n_valid_F']), parameters=parameters)
    return out


def combine_d(d0, d1):
    values = np.full(4, np.nan)
    reasons = np.full(4, '', dtype='U340')
    ok = np.isfinite(d0['values']) & np.isfinite(d1['values'])
    values[ok] = (d0['values'][ok] + d1['values'][ok]) / 2.
    for j in np.flatnonzero(~ok):
        reasons[j] = ';'.join('H%d:%s' % (h, d['reasons'][j] or 'NONFINITE_METRIC')
                              for h, d in enumerate((d0, d1)) if not np.isfinite(d['values'][j]))
    return dict(values=values, reasons=reasons)


def compute_structure_one(source_record, context_reference):
    """Pure one-model API for production and raw-record warm extraction.

    source_record has aligned ell,c,y,z,valid,item_ids,source_half. Reference has
    w, b_fit[2,I], b_eval[2,I], direction_failures[2]. No reference ALS fitting
    occurs here; scored-model OLS is recomputed from its source fitting half.
    """
    s, ref = source_record, context_reference
    c = compute_c(s['c'], s['y'], s['z'], s['valid'], s['item_ids'], ref['w'])
    halves = [compute_c(s['c'], s['y'], s['z'], s['valid'], s['item_ids'], ref['w'],
                       np.asarray(s['source_half']) == h) for h in (0, 1)]
    d = [compute_d_direction(s['ell'], s['valid'], s['source_half'], h,
                             ref['b_fit'][h], ref['b_eval'][h], str(ref['direction_failures'][h])) for h in (0, 1)]
    combined = combine_d(*d)
    return dict(values=np.r_[c['values'], combined['values']],
                reasons=np.r_[c['reasons'], combined['reasons']],
                half_values=np.array([np.r_[halves[h]['values'], d[h]['values']] for h in (0, 1)]),
                half_reasons=np.array([np.r_[halves[h]['reasons'], d[h]['reasons']] for h in (0, 1)]),
                n_C=c['n'], half_n_C=np.array([x['n'] for x in halves]),
                n_D_fit=np.array([x['n_fit'] for x in d]), n_D_eval=np.array([x['n'] for x in d]),
                D_parameters=np.array([x['parameters'] for x in d]))
