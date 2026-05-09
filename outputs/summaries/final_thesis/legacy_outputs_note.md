# Legacy Outputs Note

Legacy batch-name roots were migrated out of the final delivery layout.

## Migrated Names

- `full_multiseed` became `mock`.
- `real_public_benchmark/{dataset}` became `{dataset}`.
- The old final report root became `outputs/summaries/final_thesis/`.
- The old final figure root became `outputs/figures/summaries/final_thesis/`.

## Current Canonical Roots

- Source artifacts: `outputs/experiments/`
- Final summaries: `outputs/summaries/final_thesis/`
- Final figures: `outputs/figures/summaries/final_thesis/`
- Migration map: `outputs/summaries/layout/migration_map.csv`

Old paths are retained only as provenance strings inside per-combination `source_manifest.json` files and migration reports. They are not final citation paths.
