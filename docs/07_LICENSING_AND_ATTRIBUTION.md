# Licensing and attribution

## MIT scope for this release

`LICENSE` (MIT) and `LICENSE_SCOPE.md` cover the original project software and the reproduction
adapters, manifests, reader documentation and tooling bundled in this repository. The MIT grant is
deliberately narrow: it does not extend to third-party material.

## Third-party material

| Material | Handling in this release |
|---|---|
| Benchmark records (ARC, HellaSwag) and evaluated-result repositories used as inputs | **not bundled.** Several upstream result repositories declare no licence; absence of a declaration is not a grant. Identity manifests (`RAW_INPUT_MANIFEST.json`, `upstream_metadata/RAW_SEMANTIC_IDENTITY.json.gz`), the upstream metadata records and the exact upstream source index (`upstream_metadata/UPSTREAM_SOURCE_INDEX.json.gz`, `upstream_metadata/UPSTREAM_GOLD_SET_PINS.json`) are kept for traceability; a reader can re-obtain the raw records from upstream with `tools/reharvest_peritem.py` under the upstream terms |
| Derived outputs and reference identities | Summary results, source-table fingerprints and joint fitted vectors are bundled for comparison and analysis. Large response matrices and raw benchmark records are not bundled. Provenance is recorded in the core and supporting-analysis reference manifests and `joint_reproduction/ARTIFACTS.json`; inclusion does not relicense third-party benchmark material |
| Cited papers | **no PDFs bundled.** Citation metadata lives in `references/refs.bib`; the availability/rights list is `references/LITERATURE_AVAILABILITY.md` and the per-source records are `references/SOURCE_AND_LICENSE_RECORDS.json` |
| Transplanted historical source snapshots (`historical_reproduction/snapshots/`) | bundled as reference evidence with their original provenance; licences for upstream/transplanted code remain separate from the MIT grant above |
| Third-party Python packages | not bundled; install from the pinned lock files under their own licences |

## Attribution notes

- The paper title is: *LLM Overconfidence Is Not a Measurable, Comparable Trait*.
- Author and citation metadata are recorded in `CITATION.cff`.
