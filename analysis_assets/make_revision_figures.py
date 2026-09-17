"""Regenerate analysis figures from frozen sources; no fitting or recalibration.

Run: python make_revision_figures.py --experiment VERIFIED_RESULT_DIR --metadata-source ARCHIVED_MODEL_METADATA.csv
The constructed example is not an empirical result or a demonstration of C2/C4.
"""
from pathlib import Path
import csv
import argparse
import hashlib
import json
import collections
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment', default=HERE.parent / 'confidence_comparison/risk_metrics_v1', type=Path, help='Verified experiment result directory.')
parser.add_argument('--metadata-source', default=HERE.parent / 'external_inputs/_model_type_labels.csv', type=Path, help='Archived leaderboard model metadata CSV.')
parser.add_argument('--output-dir', type=Path, default=HERE, help='Destination for figures, metadata, and provenance (default: this directory).')
args = parser.parse_args()
DEST = args.output_dir.resolve()
DEST.mkdir(parents=True, exist_ok=True)
EXPERIMENT = args.experiment.resolve()
MATRIX = EXPERIMENT / 'CANDIDATE_MATRIX.csv'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.labelsize': 8, 'xtick.labelsize': 7.5,
                     'ytick.labelsize': 7.2, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'savefig.dpi': 300})
OUT = DEST / 'figs'
OUT.mkdir(exist_ok=True)

def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', facecolor='white')
    plt.close(fig)

# Valid 100-item, four-option construction; confidence multiset unchanged.
c = np.repeat([.9, .7], 50)
y1 = np.ones(100, dtype=int); y1[80:] = 0
y2 = np.ones(100, dtype=int); y2[:20] = 0
checks = []
for y in (y1, y2):
    # Fixed gold=0; errors choose option 1. Probabilities are strictly positive.
    p = np.repeat(((1-c)/3)[:,None], 4, axis=1)
    chosen = np.where(y, 0, 1)
    p[np.arange(100), chosen] = c
    assert np.all(p > 0) and np.allclose(p.sum(axis=1), 1)
    assert np.array_equal((p.argmax(axis=1)==0).astype(int), y)
    assert np.isclose(y.mean(), .8) and np.isclose(c.mean(), .8)
    checks.append({'A':float(y.mean()), 'C':float(c.mean()), 'O':float(c.mean()-y.mean()),
                   'R50':float(1-y[:50].mean())})
assert np.allclose([r['R50'] for r in checks], [0,.4])
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.65))
fig.subplots_adjust(left=.14, right=.985, bottom=.24, top=.76, wspace=.20)
correct, wrong = '#dce5ed', '#ad452a'
for k, (ax, y) in enumerate(zip(axes, (y1,y2))):
    for j in range(100):
        col, row = j%10, j//10
        ax.add_patch(Rectangle((col+.08,9-row+.08), .84,.84,
                              facecolor=correct if y[j] else wrong,
                              edgecolor='none'))
    ax.add_patch(Rectangle((-.05,4.95),10.1,5.1,fill=False,edgecolor='#203c52',linewidth=.9))
    ax.set(xlim=(-.2,10.2),ylim=(-.2,10.2),aspect='equal')
    ax.set_axis_off()
    ax.set_title(f'Configuration {k+1}', fontsize=8, pad=6)
    ax.text(.5,-.11, r'$R_{50}=0$' if k==0 else r'$R_{50}=0.40$',
            transform=ax.transAxes,ha='center',va='top',fontsize=9)
axes[0].text(-.09,.75,'$c=.9$',ha='right',va='center',transform=axes[0].transAxes,fontsize=7)
axes[0].text(-.09,.25,'$c=.7$',ha='right',va='center',transform=axes[0].transAxes,fontsize=7)
fig.text(.5,.965,'Same means and confidence distribution',ha='center',va='top',fontsize=8.5)
fig.text(.5,.885,r'$N=100,\quad A=C=0.80,\quad O=0$',ha='center',va='top',fontsize=9)
fig.legend(handles=[Patch(facecolor=correct,label='Correct'),Patch(facecolor=wrong,label='Incorrect')],
           loc='lower center',bbox_to_anchor=(.5,.06),ncol=2,frameon=False,fontsize=7.5,
           handlelength=1,columnspacing=1.2)
fig.text(.5,.025,'Constructed example; outlined items are retained (50%).',ha='center',fontsize=7)
save(fig,'fig_concept')

rows = list(csv.DictReader(MATRIX.open()))
assert len(rows)==31 and len({r['id'] for r in rows})==31
byid = {r['id']:r for r in rows}
labels = {
'C1':'C1  Ease-weighted errors',
'C2':'C2  Ease-weighted\n       confident errors',
'C3':'C3  Ease-weighted\n       error log-score gap',
'C4':'C4  Between-ease-bin\n       bias variance',
'D1':'D1  Negative residual\n       energy share',
'D2':'D2  Residual excess kurtosis',
'D3':'D3  Residual magnitude–\n       difficulty rank correlation',
'D4':'D4  Residual vs. squared\n       difficulty correlation',
}
ids=[f'Q_{k}' for k in labels]
selected=[byid[k] for k in ids]
assert all(r['baseline']=='Bd' for r in selected)
fig, ax=plt.subplots(figsize=(3.5,3.6))
fig.subplots_adjust(left=.54,right=.96,bottom=.17,top=.86)
y=np.arange(8)[::-1]
for yy,r in zip(y,selected):
    val,lo,hi=[float(r[k])*1e4 for k in ('delta','simultaneous_CI_low','simultaneous_CI_high')]
    assert lo<=val<=hi
    supported=r['status']=='EXPLORATORY_CANDIDATE'
    col='#087f8c' if supported else '#5b6269'
    ax.errorbar(val,yy,xerr=[[val-lo],[hi-val]],fmt='o',markersize=4.2,
                color=col,ecolor=col,elinewidth=1.15,capsize=2.2)
ax.axvline(0,color='#92979b',linestyle='--',linewidth=.8,zorder=0)
ax.set_yticks(y,[labels[r['metric']] for r in selected])
ax.tick_params(axis='y',length=0,pad=6)
ax.set_ylim(-.7,7.7)
lo=min(0., min(float(r['simultaneous_CI_low'])*1e4 for r in selected))
hi=max(0., max(float(r['simultaneous_CI_high'])*1e4 for r in selected))
pad=max(.05,(hi-lo)*.08)
ax.set_xlim(lo-pad,hi+pad)
fig.text(.5,.073,r'Prediction MSE reduction (units of $10^{-4}$)',ha='center',fontsize=8)
ax.spines[['top','right','left']].set_visible(False)
fig.text(.5,.965,'All eight structural additions to the rich baseline',ha='center',va='top',fontsize=8)
fig.text(.5,.915,'Simultaneous intervals adjusted over all 31 contrasts',ha='center',va='top',fontsize=7.3)
fig.text(.985,.025,'Positive values favor the added indicator.',ha='right',fontsize=7)
save(fig,'fig_contrasts')

# Metadata is recomputed for the actual evaluated model IDs.
eval_path=EXPERIMENT/'evaluation_cohort.csv'
eval_rows=list(csv.DictReader(eval_path.open()))
assert len({r['model_id'] for r in eval_rows}) == len(eval_rows)
source=args.metadata_source.resolve()
raw={r['repo']:r for r in csv.DictReader(source.open())}
type_map={'🔶 fine-tuned on domain-specific datasets':'Fine-tuned','💬 chat models (RLHF, DPO, IFT, ...)':'Chat/post-trained','🤝 base merges and moerges':'Merged','🟢 pretrained':'Pretrained','🟩 continuously pretrained':'Continually pretrained','':'Unknown'}
meta=[]
for r in eval_rows:
    t=raw.get('open-llm-leaderboard-old/details_'+r['model_id'].replace('/','__'),{})
    typ=type_map[t.get('model_type','')]
    try:n=float(t.get('params_B',''))
    except ValueError:n=math.nan
    if n<=0:n=math.nan
    b='Unknown' if not math.isfinite(n) else '<3B' if n<3 else '3–<7B' if n<7 else '7–<15B' if n<15 else '15–<40B' if n<40 else '≥40B'
    meta.append(dict(model_id=r['model_id'],org_id=r['org_id'],model_type=typ,
                     params_B=n if math.isfinite(n) else '',parameter_group=b))
counts=collections.Counter(r['model_type'] for r in meta)
size=collections.Counter(r['parameter_group'] for r in meta)
with (DEST/'corrected_cohort_metadata.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(meta[0]));writer.writeheader();writer.writerows(meta)
def provenance_path(path):
    try:
        return path.relative_to(HERE.parent).as_posix()
    except ValueError:
        return str(path)

report={
    'status':'Generated; numerical asset checks and independent scientific review are separate',
    'matrix_sha256':hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
    'matrix':provenance_path(MATRIX),'constructed_checks':checks,
    'contrasts':[{k:r[k] for k in ('id','metric','baseline','delta','simultaneous_CI_low','simultaneous_CI_high','status')} for r in selected],
    'metadata':{'evaluation_cohort':provenance_path(eval_path),'evaluation_cohort_sha256':hashlib.sha256(eval_path.read_bytes()).hexdigest(),
                'original_source':provenance_path(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'n_models':len(meta),'n_orgs':len({r['org_id'] for r in meta}),
                'types':dict(counts),'parameter_bins':dict(size)},
}
(DEST/'FIGURE_GENERATION.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(f"Generated figures and metadata for {len(meta)} evaluated models.")
