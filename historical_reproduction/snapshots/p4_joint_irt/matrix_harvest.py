#!/usr/bin/env python3
"""M1 — build the item×model matrices for matrix-validation (re-harvest, keeping PER-ITEM values).

Per (model, item) keeps: gold_lp_mean, gold_lp_sum, correct_cn (binary), chosen_prob_cn (confidence).
Reuses atlas_harvest.harvest_byfile (filename read + hf_transfer + string-decode + None-handling).
Canonical item axis = the gold dataset's ids (ARC-Challenge test ids / HellaSwag val inds). Resumable JSONL.

Usage:
  matrix_harvest.py <arc|hellaswag> harvest [workers]   # re-harvest ok-models -> matrix_ckpt_<bench>.jsonl
  matrix_harvest.py <arc|hellaswag> assemble            # -> aligned/matrix_<bench>.npz (LP_mean/LP_sum/BIN/CONF)
"""
import os
os.environ.setdefault('HF_HOME', '/tmp/hfm')
os.environ.setdefault('HF_HUB_CACHE', '/tmp/hfm/hub')
os.environ.setdefault('HF_DATASETS_CACHE', '/tmp/hfm/datasets')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
import sys
import json
import time
import numpy as np
import datasets
from atlas_harvest import get_res, harvest_byfile, clear_repo_cache

HERE = os.path.dirname(os.path.abspath(__file__))
ALN = os.path.join(HERE, 'aligned')
CKPT_ATLAS = os.path.join(ALN, 'atlas_checkpoint.jsonl')
# per-item tuple indices in harvest_oll_full cells: 0 gold_lp_mean, 7 gold_lp_sum, 10 correct_cn, 14 chosen_prob_cn
I_GLM, I_GLS, I_CORR, I_CP = 0, 7, 10, 14
_BENCH = None


def ckpt_path(bench):
    return os.path.join(ALN, f'matrix_ckpt_{bench}.jsonl')


def canonical_items(bench):
    if bench == 'arc':
        d = datasets.load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
        return [str(ex['id']) for ex in d]
    d = datasets.load_dataset('Rowan/hellaswag', split='validation')
    return [str(ex['ind']) for ex in d]


def ok_models():
    by = {}
    for line in open(CKPT_ATLAS):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        by[r['repo']] = r
    return [rp for rp, r in by.items() if r.get('ok')]


def process_repo(repo):
    """Worker: harvest one model's per-item values for _BENCH. Never raises.
    Retries transient fails (HF list/download rate-limit '(Request ID...)' etc.); structural fails stop early."""
    h, st = None, ''
    for t in range(4):
        try:
            h, st = harvest_byfile(repo, _BENCH, get_res(_BENCH))
        except Exception as e:
            h, st = None, 'exc:' + str(e)[:40]
        if h:
            break
        if any(k in (st or '').lower() for k in ('no_parquet', 'too_few', 'no_cells', 'not found. available')):
            break                                          # structural -> no retry
        time.sleep(2.0 * (t + 1))                          # transient (rate-limit) -> backoff
    if not h:
        rec = {'repo': repo, 'fail': st}
    else:
        cells = h['cells']
        # compact: {cid: [gold_lp_mean, gold_lp_sum, correct, chosen_prob]}
        rec = {'repo': repo, 'cells': {cid: [round(float(c[I_GLM]), 6), round(float(c[I_GLS]), 4),
                                              int(c[I_CORR]), round(float(c[I_CP]), 6)]
                                       for cid, c in cells.items()}}
    if not os.environ.get('KEEP_RAW'):       # KEEP_RAW=1 -> keep raw parquet on disk (never re-download)
        try:
            clear_repo_cache(repo)
        except Exception:
            pass
    return rec


def cmd_harvest(bench, workers):
    global _BENCH
    _BENCH = bench
    from concurrent.futures import ProcessPoolExecutor, as_completed
    models = ok_models()
    cp = ckpt_path(bench)
    done = set()
    if os.path.exists(cp):
        for line in open(cp):
            try:
                r = json.loads(line)
                if 'cells' in r:                  # only OK records are 'done'; failed ones get re-attempted
                    done.add(r['repo'])
            except Exception:
                pass
    todo = [m for m in models if m not in done]
    print(f'matrix harvest [{bench}]: {len(models)} ok-models, {len(done)} done, {len(todo)} todo, {workers}w', flush=True)
    if not todo:
        print('nothing to do', flush=True)
        return
    t0 = time.time(); n = 0; nok = 0
    fout = open(cp, 'a', buffering=1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_repo, r): r for r in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                rec = {'repo': futs[fut], 'fail': 'fut_exc:' + str(e)[:40]}
            fout.write(json.dumps(rec) + '\n')
            n += 1; nok += int('cells' in rec)
            if n % 200 == 0:
                rate = n / (time.time() - t0)
                print(f'  [{bench} {n}/{len(todo)}] ok={nok} {rate:.1f}/s ETA={(len(todo)-n)/rate/60:.0f}min', flush=True)
    fout.close()
    print(f'matrix harvest [{bench}] done: {n} processed, {nok} ok, {time.time()-t0:.0f}s', flush=True)


def cmd_assemble(bench):
    items = canonical_items(bench)
    iidx = {it: i for i, it in enumerate(items)}
    cp = ckpt_path(bench)
    recs = []
    seen = set()
    for line in open(cp):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r['repo'] in seen or 'cells' not in r:
            continue
        seen.add(r['repo'])
        recs.append(r)
    models = [r['repo'] for r in recs]
    nI, nM = len(items), len(models)
    LPm = np.full((nI, nM), np.nan, np.float32)
    LPs = np.full((nI, nM), np.nan, np.float32)
    BIN = np.full((nI, nM), np.nan, np.float32)
    CONF = np.full((nI, nM), np.nan, np.float32)
    for j, r in enumerate(recs):
        for cid, v in r['cells'].items():
            i = iidx.get(str(cid))
            if i is None:
                continue
            LPm[i, j], LPs[i, j], BIN[i, j], CONF[i, j] = v[0], v[1], v[2], v[3]
    # item coverage: keep items present in >=95% of models (dense for SVD/IRT); log dropped
    present = (~np.isnan(LPm)).mean(1)
    keep = present >= 0.95
    items_k = [it for it, k in zip(items, keep) if k]
    LPm, LPs, BIN, CONF = LPm[keep], LPs[keep], BIN[keep], CONF[keep]
    np.savez(os.path.join(ALN, f'matrix_{bench}.npz'),
             items=np.array(items_k), models=np.array(models),
             LP_mean=LPm, LP_sum=LPs, BIN=BIN, CONF=CONF)
    # cross-check vs atlas scalars: per-model nanmean(LP_mean) ≈ gauge, nanmean(BIN) ≈ ability
    print(f'assemble [{bench}]: {nM} models x {len(items_k)} items (from {nI}, kept >=95% present)', flush=True)
    print(f'  per-model fill: median items/model = {np.median((~np.isnan(LPm)).sum(0)):.0f}', flush=True)
    print(f'  -> aligned/matrix_{bench}.npz (LP_mean/LP_sum/BIN/CONF)', flush=True)


if __name__ == '__main__':
    bench = sys.argv[1]
    cmd = sys.argv[2] if len(sys.argv) > 2 else 'harvest'
    if cmd == 'harvest':
        cmd_harvest(bench, int(sys.argv[3]) if len(sys.argv) > 3 else 16)
    elif cmd == 'assemble':
        cmd_assemble(bench)
    else:
        raise SystemExit(f'unknown cmd {cmd}')
