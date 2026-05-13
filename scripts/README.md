# scripts 说明

`scripts/` 只保留最终结果构建与审计脚本，不再提供根目录兼容入口。

常用命令：

```bash
python scripts/final_thesis/build_final_thesis_results.py
python scripts/audit/audit_experiment_symmetry.py
python scripts/audit/audit_repository_bloat.py
python scripts/audit/audit_code_structure.py
python scripts/audit/generate_project_file_report.py
```

目录分工：

- `scripts/final_thesis/`：构建最终论文结果汇总、覆盖审计、参数扫描汇总和论文图。
- `scripts/audit/`：检查实验对称性、仓库体积、代码结构和项目文件功能。

这些脚本是轻量维护与审计工具，不会运行完整实验矩阵。
