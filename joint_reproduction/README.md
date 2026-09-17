# Full-panel joint item-response analysis

This directory contains the full-panel supporting analysis: 6,717 ARC models,
6,720 HellaSwag models and 6,716 shared models.

To recompute the fitted-vector statistics and coupled likelihood-scale diagnostic:

```bash
python joint_reproduction/analyze.py --out /tmp/joint-analysis-new
```

NumPy is required. Use a new output directory. The command checks all bundled artifact hashes,
loads the analyzer with paths resolved within this directory, and recomputes the
statistical tables. Item-length vectors were derived exactly from the response matrices using the
specified formula; their hashes and source-matrix hashes are in `ARTIFACTS.json`. They permit the
likelihood-scale diagnostic without distributing the large matrices. Fitted-result recomputation
does not refit the latent model or validate raw data acquisition.

To reconstruct the complete matrices from acquired raw NPZ files, then run the fitter:

```bash
python joint_reproduction/cold.py build --bench arc --raw-root /data/raw --out /data/joint-arc
python joint_reproduction/cold.py build --bench hellaswag --raw-root /data/raw --out /data/joint-hellaswag
python joint_reproduction/cold.py fit --bench arc --matrix /data/joint-arc/matrix_arc.npz --out /data/fit-arc --validate-only
python joint_reproduction/cold.py fit --bench arc --matrix /data/joint-arc/matrix_arc.npz --out /data/fit-arc-full
python joint_reproduction/cold.py fit --bench hellaswag --matrix /data/joint-hellaswag/matrix_hellaswag.npz --out /data/fit-hellaswag-full
```

Analyze the actual newly fitted outputs by passing both fit directories:

```bash
python joint_reproduction/analyze.py --arc-fit /data/fit-arc-full --hellaswag-fit /data/fit-hellaswag-full --out /data/analysis-fresh
```

This mode checks the fixed full-fit configuration, complete matrix identities, model order and
run-to-vector hashes before using the supplied files. Omitting both fit arguments selects the
bundled reference-fit recomputation mode shown above.

Acquire the two supplemental ARC members using `tools/reharvest_peritem.py --include-supplemental`.
If stored separately, pass `--supplemental-root` to `build`. Every raw array and scalar is checked
against the public semantic identity index before applying the item mask and the
at-least-50-valid-item rule. The resulting complete matrix, item order and model order must match
`input_contract/CONTRACT.json` exactly. A `--npz` sample build is explicitly not fit-eligible.

The fit command requires PyTorch and a CUDA GPU for the fixed 2,500-epoch analysis. `--validate-only`
checks inputs/configuration without fitting. `--pilot` runs at most five epochs and marks its
outputs as pilot-only; it cannot supply the reported full-fit results. Use new output directories.

`original_code/` retains computational sources with portable paths. The public commands resolve runtime paths and verify matrix and configuration identities. `reference_results/` contains
the fitted vectors, fixed-epoch run records and numerical results. The commands resolve their own paths. The two continuous-response
and chosen-option-confidence definitions remain distinct. The 2,500-epoch runs are not a convergence
proof.

The public checks distinguish full fitting, matrix reconstruction, bounded pilot runs and
fitted-result recomputation.
