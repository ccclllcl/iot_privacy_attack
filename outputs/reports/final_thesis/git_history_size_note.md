# Git History Size Note

The current working tree has been pruned and `repository_bloat_audit.md` reports no deletion candidates. The tracked working-tree size is now small relative to the earlier repository state.

GitHub repository size may still remain large because Git history keeps old large objects from previous commits. Deleting files in the latest commit does not remove those objects from historical commits.

To truly reduce clone size, the repository history would need to be rewritten with a tool such as `git filter-repo` or BFG. That requires a force push and should be treated as a separate maintenance operation.

Example command to review before any history rewrite:

```bash
git filter-repo --path data/processed --path data/defended --path outputs/models --path outputs/defense/dataset_matrix --path outputs/reports/dataset_matrix --invert-paths
```

Before running any history rewrite:

- Back up the repository.
- Confirm no collaborators depend on the old history.
- Force push only after confirming the rewritten history is correct.
- Ask users to re-clone the repository after the rewrite.

This cleanup does not rewrite history and does not force push.
