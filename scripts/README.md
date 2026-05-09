# Scripts

The root-level scripts are compatibility wrappers for stable commands:

```bash
python scripts/build_final_thesis_results.py
python scripts/audit_experiment_symmetry.py
python scripts/audit_repository_bloat.py
python scripts/audit_code_structure.py
```

Organized implementations live in:

- `scripts/final_thesis/`: final thesis summary builder.
- `scripts/audit/`: audit scripts for experiment symmetry, repository bloat, and code structure.

These scripts are lightweight maintenance and audit tools. They do not run the full experiment matrices.
