# iot_privacy_attack

智能家居（Smart* 风格）数据上的 **行为推断攻击基线** 与 **数据侧防御评测**：从设备事件/功耗时序中识别用户行为（睡眠、离家、做饭、使用电脑等），并在对时序特征施加扰动或本地差分隐私（LDP）机制后，量化攻击者分类器准确率下降与数据失真，用于毕业设计「隐私–效用」分析。

## 项目定位

- **攻击者基线（第一部分）**：在原始（或规则标注）数据上训练 LSTM / MLP，报告无保护情形下的识别性能上界。
- **防御评测（第二部分）**：对预处理后的滑窗序列 **X** 施加 **加性噪声**、**固定 ε 的 LDP**，或 **自适应本地差分隐私（adaptive_ldp）**；标签 **y** 不变，复用同一攻击模型结构，对比防御前后准确率、F1、混淆矩阵及 MSE/MAE 等失真指标。

后续可在同一框架下替换为更复杂的防御（梯度扰动、联邦 DP 等），或接入真实 Smart* 数据。

## 与工作定位的对应关系（毕设）

与本课题「流量分析下的行为隐私保护 / 自适应本地 DP / 边缘辅助 / 轻量化评估」的映射如下（实现为**可运行仿真**，便于论文描述与实验复现）：

| 论文表述 | 本项目中的体现 |
|----------|----------------|
| 流量侧信道推断行为 | 攻击模型从**设备事件与功耗时序**（上报强度、活动模式）推断行为；对应「不解密内容亦可从元数据/统计推断」的威胁模型。 |
| 数据上报阶段扰动 | `run_defense.py`：在**预处理后的滑窗张量**上施加噪声 / LDP / **adaptive_ldp**（模拟网关节点或可信上报代理）。 |
| **自适应**本地差分隐私 | `defense.method: adaptive_ldp` + `adaptive_ldp.*`：按窗口计算**敏感度代理**（时序 std）与**流量代理**（L1 能量和），动态分配 **epsilon**，高风险窗口更强噪声。 |
| 边缘辅助 | `src/edge/budget_allocator.py`：可选对一批窗口的 **epsilon 序列**做**总逆预算**裁剪；`defense_pipeline` 在**全量样本上 fit** 标定分位数，模拟边缘掌握历史统计后下发策略。 |
| 轻量化与系统性能 | `defense_summary.json` 中的 **`system_performance`**：记录各划分 `transform` 耗时（秒），便于与基线方法对比开销。 |
| 全流程：数据—分析 | `raw` → `preprocess` → `train/evaluate`（攻击基线）→ `defense` → `defense_eval`（防御效果）。 |

**说明**：真实网络流量包级特征可替换或接入本 pipeline 的 CSV 映射；当前以智能家居**通用事件流**为主，论文中可论证其与流量统计特征在「侧信道推断」意义上的一致性。

## 环境要求

- Python **3.10+**
- 依赖见 `requirements.txt`（pandas、numpy、scikit-learn、matplotlib、PyTorch 等）。

## 安装

在项目根目录 `iot_privacy_attack/` 下执行：

```bash
pip install -r requirements.txt
```

若需 GPU 训练，请按 [PyTorch 官网](https://pytorch.org/) 选择对应 CUDA 版本安装命令。

## Web UI（推荐）

本项目提供一个**简洁的 Web 界面**用于替代命令行操作，包含：

- **侧边栏导航**：Instructions / Run / History
- **操作说明**：在界面内直接查看典型流程与注意事项
- **一键运行**：预处理、训练、评估、防御、防御评估、参数扫描
- **运行历史**：自动记录每次运行的命令、耗时、返回码、stdout/stderr（尾部）以及常见输出文件路径

启动方式（Windows / PowerShell）：

```bash
py -3 -m streamlit run ui_app.py
```

运行历史默认写入：

- `outputs/ui/run_history.jsonl`

## 目录结构

```
iot_privacy_attack/
├── data/
│   ├── raw/
│   ├── processed/
│   └── defended/              # 扰动后的序列与 MLP 特征
├── outputs/
│   ├── models/
│   ├── figures/
│   ├── reports/
│   └── defense/               # 防御评估报告与图表
│       └── comparisons/       # 批量参数扫描结果 CSV / 曲线
├── src/
│   ├── defenses/              # 防御抽象与实现（含 adaptive_ldp）
│   ├── edge/                  # 边缘预算调度（仿真）
│   ├── config.py
│   ├── preprocess.py
│   ├── features.py
│   ├── dataset.py
│   ├── models/
│   ├── train.py
│   ├── evaluate.py
│   ├── defense_eval.py
│   └── experiment_compare.py
├── configs/default.yaml
├── run_preprocess.py
├── run_train.py
├── run_evaluate.py
├── run_defense.py
├── run_defense_eval.py
├── run_compare.py
├── run_import_uci_har.py
├── generate_mock_data.py
├── requirements.txt
└── README.md
```

## 数据放置与字段映射

1. 将长表格式 CSV 放到 `data/raw/`，路径在 `configs/default.yaml` 的 `paths.raw_csv` 中配置。
2. 默认列名：`timestamp`, `device_id`, `value`, `behavior_label`。若列名不同，修改 `columns` 映射即可。
3. **无标签**时：将 `label_mapping.source` 设为 `rules`，并按 `device_groups` / `rules` 调整。

## 真实数据集（推荐：UCI HAR，一键下载并导入）

如果你暂时拿不到 Smart* / CASAS 等智能家居原始事件流，本项目也支持一个**公开可直接下载**的真实时序行为数据集：

- **UCI HAR (Human Activity Recognition Using Smartphones)**：50Hz 加速度计/陀螺仪，窗口长度 128，6 类活动标签（walking/sitting/laying...）。

该数据集**原生就是滑窗样本**，因此不走 `run_preprocess.py`（它面向“智能家居长表 CSV → 重采样 → 滑窗”），而是通过导入脚本直接生成本项目统一的：

- `data/processed/sequences.npz`
- `data/processed/meta.json`
- （可选）`data/processed/mlp_features.npz`

一键导入（自动下载 + 解压 + 转换）：

```bash
python run_import_uci_har.py --config configs/default.yaml --auto-download
```

之后训练/评估/防御流程与原来完全一致：

```bash
python run_train.py --config configs/default.yaml --model lstm
python run_evaluate.py --config configs/default.yaml --model_path outputs/models/best_lstm.pt
python run_defense.py --config configs/default.yaml
python run_defense_eval.py --config configs/default.yaml --mode fixed_attacker --model_path outputs/models/best_lstm.pt
```

说明：
- UCI HAR 导入后的特征名固定为 9 个惯性通道（见 `data/processed/meta.json` 的 `feature_names`），防御配置里的 `selected_features` 也应与之对应。
- 导入脚本会清理解压产生的 `__MACOSX/`、`._*`、`.DS_Store` 这类附件文件（不影响数据内容，只减少干扰）。

## 快速跑通（合成数据 + 攻击基线）

```bash
python generate_mock_data.py
python run_preprocess.py --config configs/default.yaml
python run_train.py --config configs/default.yaml --model lstm
python run_evaluate.py --config configs/default.yaml --model_path outputs/models/best_lstm.pt
```

## 防御模块简介

- **噪声防御（`defense.method: noise`）**：对选定特征维加入高斯或拉普拉斯噪声，强度由 `noise_scale` 控制；不提供形式化 DP 保证，适合作为简单基线。
- **LDP 防御（`defense.method: ldp`）**：
  - **数值特征**：Laplace 机制，尺度与 `epsilon`、`ldp_sensitivity`（L1 敏感度上界代理）相关；**epsilon 越小，噪声越大，隐私越强**（形式化保证需在真实查询与裁剪范围下严格推导）。
  - **二值特征**：随机响应（Randomized Response），由 `binary_features` 与 `binary_threshold` 指定；满足经典 **ε-LDP**（epsilon-LDP）参数化。
- **自适应 LDP（`defense.method: adaptive_ldp`，默认）**：见配置节 `adaptive_ldp`。**敏感度**用窗口内时序标准差，**流量代理**用窗口 L1 能量；二者加权得到风险，在 `epsilon_min`～`epsilon_max` 间为**每个窗口**分配不同 epsilon，再施加 Laplace / RR。可选 `use_edge_budget_cap` 启用边缘总预算裁剪。
- **作用范围**：`apply_to: all` 或 `selected` + `selected_features`（与 `meta.json` 中设备/列名一致）。
- **粒度**：对每个滑动窗口样本 **X**（形状约为 `T × F`：时间步 × 特征维）独立扰动，输出 shape 不变，标签 **y** 不变。

## 两种评估模式（论文威胁模型）

| 模式 | 含义 | 典型结论 |
|------|------|----------|
| **A. fixed_attacker（固定攻击者）** | 攻击者在**干净数据**上训练好的分类器**不变**，仅将输入替换为防御后的测试数据。 | 刻画「识别器已部署、用户发布扰动数据」的场景；若准确率显著下降，说明**既有模型被削弱**。 |
| **B. retrain_attacker（防御后重训）** | 攻击者获得**与防御同分布**的训练集并**重新训练**，再在防御后测试集上评估。 | 刻画**自适应对手**；若仍能保持高准确率，说明防御对自适应攻击**稳健性不足**。 |

**为何同时报告攻击性能下降与数据失真？**  
仅看准确率下降无法区分「破坏了推断」还是「数据已不可用」；结合 MSE/MAE/Pearson 等相关性指标，才能讨论 **隐私–可用性权衡**。

## 防御实验命令

**1）生成防御后数据（写入 `data/defended/`）**

```bash
python run_defense.py --config configs/default.yaml
```

生成文件示例：`defended_train.npz`、`defended_val.npz`、`defended_test.npz`、`defended_sequences.npz`、`defended_mlp_features.npz`、`defense_artifact.json`、`defense_summary.json`。

**2）模式 A：固定攻击者（需已训练 `best_lstm.pt`）**

```bash
python run_defense_eval.py --config configs/default.yaml --mode fixed_attacker --model_path outputs/models/best_lstm.pt
```

若防御数据已生成且配置未改，可加 `--skip-pipeline` 跳过重复扰动。

**3）模式 B：防御后重训攻击者**

```bash
python run_defense_eval.py --config configs/default.yaml --mode retrain_attacker
```

重训模型默认保存为 `outputs/models/best_lstm_defended_retrain.pt`（可在 `defense_eval.retrained_model_name` 修改）。**请保持 `train.model_type` 与你要训练的架构一致（lstm / mlp）。**

**4）批量参数扫描（固定攻击者）**

```bash
python run_compare.py --config configs/default.yaml --method ldp --model_path outputs/models/best_lstm.pt
python run_compare.py --config configs/default.yaml --method noise --model_path outputs/models/best_lstm.pt
```

列表在 `compare.ldp_epsilon_list`、`compare.noise_scale_list` 配置。

## 输出说明（防御相关）

| 路径 | 内容 |
|------|------|
| `data/defended/defended_*.npz` | 扰动后训练/验证/测试集 |
| `outputs/defense/defense_report.json` / `.txt` | 失真指标、攻击性能对比、模式说明 |
| `outputs/defense/accuracy_comparison.png` | 模式 A：干净 vs 防御后测试准确率柱状图 |
| `outputs/defense/confusion_matrix_baseline.png` | 固定攻击者在干净测试集上混淆矩阵 |
| `outputs/defense/confusion_matrix_defended.png` | 固定攻击者在防御后测试集上混淆矩阵 |
| `outputs/defense/comparisons/comparison_results.csv` | 批量扫描汇总 |
| `outputs/defense/comparisons/epsilon_vs_accuracy.png` | LDP：epsilon–准确率曲线 |
| `outputs/defense/comparisons/epsilon_vs_distortion.png` | LDP：epsilon–MSE 曲线 |
| `outputs/defense/comparisons/distortion_vs_noise.png` | 噪声：强度–MSE 与准确率（双轴） |
| `outputs/defense/comparisons/noise_scale_vs_accuracy.png` | 噪声强度–准确率曲线 |

## 基线流程命令速查

| 命令 | 作用 |
|------|------|
| `python generate_mock_data.py` | 生成示例 CSV |
| `python run_preprocess.py --config configs/default.yaml` | 预处理 |
| `python run_import_uci_har.py --config configs/default.yaml --auto-download` | 下载并导入真实 UCI HAR 数据集（跳过预处理） |
| `python run_train.py --config configs/default.yaml --model lstm` | 训练 LSTM |
| `python run_evaluate.py --config configs/default.yaml --model_path outputs/models/best_lstm.pt` | 基线评估 |

## 配置项提示

- **全局随机种子**：`experiment.random_seed` 控制 Python / NumPy / PyTorch、预处理划分、训练、防御 RNG 等；`generate_mock_data.py` 默认从同一配置文件读取（可用 `--seed` 临时覆盖）。
- **防御**：`defense.*`（`method`、`epsilon`、`noise_scale`、`apply_to`、`selected_features`、`binary_features`、`numeric_features` 等）。
- **路径**：`paths.defended_dir`、`paths.defense_dir`。
- **训练/评估设备**：`train.device`、`evaluate.device`。

## 如何解读实验结果

- **防御有效**：在**固定攻击者**设定下，防御后 **Accuracy / Macro-F1 明显下降**，且 **Accuracy Drop**、**Relative Accuracy Drop (%)** 较大。
- **可用性受损**：**MSE/MAE 升高**、**Pearson 相关降低**，说明扰动过大，下游任务可能不可用；需在隐私与效用间折中。
- **自适应风险**：若 **重训后** 在防御测试集上准确率仍高，说明攻击者可适应扰动，需更强机制或特征级防护。

## 许可证与引用

毕业设计自用项目。若使用 Smart* 等公开数据集，请在论文中按原数据集要求标注引用来源。
