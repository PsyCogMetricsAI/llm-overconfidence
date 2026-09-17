# Inputs and provenance

The release separates compact scientific materials from large upstream response files.

| Material | Location or source | Purpose |
|---|---|---|
| Figures, table snippets, and reference results | Bundled | Rebuild standalone analysis tables and figures |
| Core and supporting-analysis code | Bundled | Recompute descriptors, fits, predictions, and statistics |
| Fixed definitions, candidate identities, splits, and numerical fingerprints | Bundled | Keep the rerun aligned with the reported scientific design |
| Model labels and canonical item ordering | Bundled metadata | Cohort description and deterministic matrix construction |
| Item-level response records | Retrieved from pinned upstream repositories | Starting data for the complete computation |
| Generated matrices, features, estimators, and run logs | Created locally | Intermediate outputs; not required as private inputs |

## Response identity

The core archive contains 6,746 ARC and 6,748 HellaSwag NPZ records. Two additional ARC records are included for the broader joint analysis. The public semantic index covers all 13,496 files and binds array keys, shapes, dtypes, values, missingness, and scalar metadata. This allows reconstruction without possession of the original archive containers.

The acquisition tool writes both reconstructed files and retrieval logs. A reconstructed NPZ may have a different container hash while carrying exactly the required arrays. The filtering tool verifies those arrays, applies the specified item exclusions and array ordering, and writes deterministic analysis files. It records the actual downloaded container identity separately from the archived scientific-source identity.

The item-exclusion mask lists 4 ARC IDs and 22 HellaSwag IDs associated with ambiguous question/option mappings. The same mask applies to the supervised and joint analyses. The two additional ARC files expand the joint panel only; the core candidate model universe stays fixed.

## Source revisions and availability

The source index identifies the repository and parquet path for each record. The pin index fixes revisions for all 6,748 collected repositories. Some pins have direct download evidence; others come from metadata queries. Listing provenance is recorded separately: a harvest-time listing does not certify availability at a later pinned revision. The downloader checks the actual pinned source and requires matching response fingerprints.

Acquisition metadata do not include a revision for every source record. Pinned retrieval is checked against the recorded scientific values; it does not establish the identity of the upstream containers used during collection. If an upstream source is removed, gated, or incompatible with those identities, acquisition stops and records the failure. The pipeline never fills missing responses with zeros or silently chooses a replacement model.

## Portable checks

The public workflow verifies inputs using compact array and feature fingerprints. It supplies a public protocol for runtime provenance checks and reference summaries for numerical comparison. The bundled code and adapters implement the computations, resolve runtime paths and validate analysis inputs. Fresh feature generation, prediction, and scoring are checked separately.

`tools/check_required_inputs.py --mode public` checks the bundled dependency set. It is a local integrity check, not a test that every upstream repository is currently online. The complete commands are in the [reproduction guide](03_FULL_COLD_REPRODUCTION.md).

## Data terms

Original code is MIT-licensed. Dataset licenses and upstream repository terms apply independently. The package includes retrieval information and metadata without asserting a new redistribution license for upstream responses. See [licensing and attribution](07_LICENSING_AND_ATTRIBUTION.md).
