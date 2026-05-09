# Repository Bloat Audit

- Generated at: `2026-05-09T15:05:41`
- Tracked files: `1183`
- Total tracked bytes: `14798118`
- Delete candidates: `0`
- Path hygiene issues: `17`

## Category Counts

- `code`: 73
- `docs`: 5
- `final_figure_required`: 22
- `final_thesis_required`: 54
- `generated_config`: 123
- `source_artifact_referenced`: 887
- `unknown`: 5
- `web_asset`: 14

## Recommendation Counts

- `keep`: 1183

## Path Hygiene Notes

- Path hygiene issues are not deletion candidates by themselves.
- The remaining issues are mainly Cooja local WSL radio/app log paths retained to document the completed local evaluation source.
- Portable Cooja reproduction should use `configs/cooja_defense_dummy_logs.template.json` with `COOJA_LOG_ROOT` instead of the local WSL paths.

## Delete Candidates
