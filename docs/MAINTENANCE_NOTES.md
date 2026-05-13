# 维护说明

本文记录项目维护时需要遵守的技术约定。这些约定服务于脚本可运行性和结果可复核性，不属于论文结论本身。

## 标识符约定

代码和产物中的 dataset、method、mode、model、JSON/CSV 字段名保持英文标识。例如：

- dataset：`mock`、`uci_har`、`kasteren`、`casas_hh101`、`cooja`
- method：`adaptive_ldp`、`ldp`、`noise`
- mode：`fixed_attacker`、`retrain_attacker`
- Cooja dummy method：`dummy_noise`、`dummy_ldp`、`dummy_adaptive_ldp`

这些标识直接被脚本、Dashboard 和结果汇总读取，因此文档说明可以使用中文叙述，但不改变标识符本身。

## 路径约定

最终结果复核围绕以下路径展开：

```text
outputs/experiments/
outputs/summaries/final_thesis/
outputs/figures/summaries/final_thesis/
outputs/figures/experiments/
```

早期批次路径的迁移关系记录在 `outputs/summaries/layout/migration_map.csv` 和 `outputs/summaries/layout/migration_report.md`。

## Cooja 约定

Cooja 结果用于 fixed/retrain 攻击准确率和节点侧 dummy 流量功能性验证。当前节点级开销实验已经量化 dummy/real 包比例、packet/byte overhead、Cooja 仿真时间端到端时延和 Contiki-NG Energest 仿真能耗估计。复现日志评估时，可基于 `configs/cooja_defense_dummy_logs.template.json` 和 `COOJA_LOG_ROOT` 配置本地日志路径；复现开销实验时还需要 `configs/cooja_energy_model.json` 中的电压/电流参数。能耗估计不等同于硬件功耗仪测量。

## 轻量审计命令

以下命令用于复核最终结果包和代码结构：

```bash
python scripts/audit/audit_experiment_symmetry.py
python scripts/audit/audit_repository_bloat.py
python scripts/audit/audit_code_structure.py
```
