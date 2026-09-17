# Environment

## Recorded platform

The core cold runs and asset checks use Linux CPU with CPython 3.12.11
(`environment/PYTHON_VERSION.txt`). The drivers export single-threaded numerical settings
(`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`) so that results do not
depend on host thread counts.

Recorded wall-clock times are machine- and load-dependent. The cached-workflow timing
snapshot and the measured warm ledger are reported separately, and the complete cold end-to-end cost
is `UNMEASURED`.

## Pinned environments

| Purpose | File | Contents |
|---|---|---|
| Core pipeline (enforced at run time) | `environment/requirements-core.lock` | `cloudpickle 3.1.2`, `joblib 1.6.0`, `numpy 2.5.3`, `scikit-learn 1.6.1`, `scipy 1.18.1`, `threadpoolctl 3.6.0` |
| Analysis figures/tables | `environment/requirements-figures.lock` | the core lock plus `matplotlib 3.11.0` and its pinned dependencies |

`run_core.py raw` compares the installed distributions against the core lock and writes
`ACTUAL_ENVIRONMENT.json` into the fresh work tree; a version mismatch stops the run.

```bash
python3 -m venv ../llm-overconfidence-env
../llm-overconfidence-env/bin/python -m pip install -r environment/requirements-core.lock
../llm-overconfidence-env/bin/python -m pip check
```

## Per-branch dependencies

| Branch | Packages | Notes |
|---|---|---|
| Core experiment | core lock | version-checked by the driver |
| Supporting slope analysis | NumPy/SciPy from the core lock | the adapter sets the thread variables itself |
| Five/nine-summary predictions | core lock (needs `scikit-learn 1.6.1` for `HistGradientBoostingRegressor`) | no extra packages |
| Supporting algebra (`historical_reproduction/code/check_algebra.py`) | NumPy + `sympy 1.14.0` | SymPy is deliberately outside the core lock; without it the script fails at import |
| Full-panel joint item-response analysis | `environment/requirements-joint.lock` (separate environment; NumPy 2.5.0, SciPy 1.18.1, PyTorch 2.12.1+cu130) | full-fit and public pilot/validation records use CPython 3.12.11 and NVIDIA RTX 4000 Ada Generation; float64, seed 0; see `joint_reproduction/` |
| Pinned upstream acquisition | `environment/requirements-acquisition.lock` | observed NumPy/PyArrow/Hub-client versions; no full clean-install claim |
| Figure/table/asset-check scripts | figures lock | standalone rendering and checks require no LaTeX or PDF inspection tools |

## Acquisition and computation

The computational stages operate on local files. The separate upstream acquisition step
(`tools/reharvest_peritem.py`) fetches records at the bundled revision pins and verifies their
semantic identities. Building a new pin index with `tools/make_upstream_pin_index.py` also uses
network metadata; it is not required to use the bundled pins.

## Where the checks ran

The figure/table/algebra checks were executed on a container with Python 3.12.11, NumPy, SciPy,
Matplotlib and SymPy; the fitting phases need the locked core environment and the external archival
inputs before they can run. The environment and verification scope are recorded in `docs/03_FULL_COLD_REPRODUCTION.md`.
