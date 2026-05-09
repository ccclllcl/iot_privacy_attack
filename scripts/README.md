# scripts 说明

根目录脚本是稳定命令的兼容入口：

```bash
python scripts/build_final_thesis_results.py
python scripts/audit_experiment_symmetry.py
python scripts/audit_repository_bloat.py
python scripts/audit_code_structure.py
python scripts/generate_project_file_report.py
```

实际实现位于：

- `scripts/final_thesis/`：最终论文结果汇总构建。
- `scripts/audit/`：实验对称性、仓库体积、代码结构和项目文件功能报告审计。

这些脚本是轻量维护与审计工具，不会运行完整实验矩阵。
