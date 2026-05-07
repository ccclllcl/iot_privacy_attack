# 给 GPT 修改论文的 Prompt

请把下面整段内容复制给 GPT，并同时提供论文初稿、老师修改建议文档以及本仓库的 `outputs/reports/final_thesis/` 结果包。

```text
你是一名熟悉本科毕业论文写作、物联网安全、流量分析、行为识别攻击、本地差分隐私和实验结果表述规范的中文论文修改助手。请基于我提供的论文初稿、老师的修改建议，以及代码仓库中的最终实验产物，对论文进行系统性修改。

一、论文基本信息

论文题目应统一为：
物联网环境下的隐私保护机制设计与实现

英文题目可改为：
Design and Implementation of Privacy Protection Mechanisms in IoT Environments

指导教师姓名统一为：
欧晓聪老师

学院名称统一为：
网络空间安全学院

二、必须优先完成的硬性修改

1. 修正致谢中的导师姓名错误：不得再出现“于忠德老师”“方老师”等与封面不一致的称呼，统一改为“欧晓聪老师”。
2. 修正致谢中的学院名称错误：将“软件学院”改为“网络空间安全学院”。
3. 重写致谢，避免模板化表达，内容要真实、自然、得体，感谢导师、学院、同学/朋友、家人以及毕业设计过程中的帮助。
4. 统一章节标题。第一章 1.4 中对第 2 章的描述必须与正文标题“相关技术与实验基础”一致，不要写成“系统设计与实现”。
5. 修改题目、摘要、英文摘要、关键词，使它们与“物联网环境下的隐私保护机制设计与实现”一致。

三、摘要与关键词修改要求

1. 中文摘要控制在 300-350 字，结构清晰呈现“研究问题、方法、实验结果、结论”。
2. 摘要中不要写“GitHub 仓库当前公开 master 分支”等过于工程化、过于具体的表达，可改为“本文基于公开的代码仓库和可复现实验流程……”。
3. 英文摘要要语法自然、专业，不要直译中文，不要出现明显中式英语。
4. 主题词中“自适应隐私保护”讨论较少，建议替换为“本地差分隐私”。推荐关键词：
物联网；流量分析；行为推断；本地差分隐私；隐私-可用性权衡

四、章节级修改要求

第一章 绪论：
1. 1.2 中“仓库采用 mock 数据与真实公开数据集并行的双轨数据策略”属于项目实现描述，不属于相关工作，应移到第 3 章或实验设计部分。
2. 1.3 “本文主要工作”可以保留 5 点结构，但语言要更精炼，减少“仓库保留了”“项目补充了”等口语化/工程化表达。
3. 1.4 的章节组织描述要与正文标题完全一致。

第二章 相关技术与实验基础：
1. 2.2 中关于“真实系统中的严格敏感度仍需结合具体查询函数……”的表述过长，请精简为类似：“需要说明的是，当前实现主要服务于可复现实验，严格敏感度仍需结合具体查询函数、裁剪范围和部署边界进行推导。”
2. 2.5 本章小结过于简单，请扩充，概括本章介绍的威胁模型、差分隐私、本地差分隐私、模型基础与实验流程。

第三章 隐私防御机制设计与实现：
1. 本章篇幅过长，约 6 页，存在大量对代码仓库的逐项罗列。请精简 10%-15%，把相似脚本、目录、实现细节合并表达。
2. 写作重点从“逐个介绍代码文件”改为“提炼系统机制”：攻击者可见什么、系统扰动什么、防御如何进入数据流、如何区分 fixed_attacker 和 retrain_attacker。
3. 将第一章中不适合放在相关工作的“双轨数据策略”移入本章或实验设计部分，说明 mock 数据用于流程复现，UCI HAR、Kasteren、CASAS 用于真实公开数据验证。
4. 3.8 本章小结要扩充，概括加性噪声、固定参数 LDP、自适应 LDP、Cooja 节点级日志链路和真实数据导入流程。

第四章 实验设计与结果分析：
1. 本章篇幅过长，约 12 页，内容密集。请精简重复性描述，尤其是反复出现的“LSTM 强于 MLP”“LDP 抑制最强但失真最高”等结论。
2. 保留关键实验表格、图和核心解释，但减少过程性叙述。
3. 增加对失败案例或异常结果的讨论，例如：
   - Cooja 中 dummy_noise 在 retrain_attacker 下可能出现准确率恢复，说明自适应攻击者可重新学习防御后分布。
   - Kasteren 类别多、样本稀疏，基线准确率较低，不能简单与 UCI HAR 做绝对值排序。
   - 强 LDP 虽降低攻击准确率，但 MSE/MAE 上升、Pearson 下降，不能只用准确率下降证明方案优越。
4. 4.8 本章小结要扩充，归纳 mock、真实数据、参数扫描、混淆矩阵、Cooja 节点级实验的共同结论和限制。
5. 真实数据集之间不要直接比较绝对准确率高低，因为类别空间、样本分布、传感器维度和标签定义不同。只做各数据集内部的 baseline 与 defended 对比。

第五章 总结与展望：
1. 当前篇幅过短，约 1 页，请扩充。
2. 总结部分要归纳本文主要贡献：行为推断攻击基线、三类数据侧防御、固定/重训攻击者评估、多数据集验证、Cooja 节点级日志评估、最终结果包可复现。
3. 展望部分可从以下方向展开：
   - 接入真实智能家居网关或传感器节点，验证端到端部署效果。
   - 补充真实系统开销指标，包括包数、字节数、时延、功耗、CPU 开销。
   - 分数据集调参，分别处理 UCI HAR、Kasteren、CASAS 的不同数据特征。
   - 做消融实验，分析窗口风险评分、epsilon 范围、边缘预算裁剪的单独贡献。
   - 提升 Cooja dummy 流量面对重训攻击者时的稳健性。
   - 探索与联邦学习、安全多方计算、SecretFlow 等隐私计算框架结合。

六、参考文献修改要求

1. 当前参考文献约 15 条，数量偏少，请扩充到 20-25 条。
2. 增加近两年与 IoT 隐私、流量分析、差分隐私、本地差分隐私、隐私计算框架相关的文献。
3. 修正已有参考文献格式：
   - “Computers & Security, 2025”期刊名应写为“Computers & Security”。
   - “计算机研究与发展，2017”需要补充卷期号。
4. 补充 SecretFlow、TensorFlow、PyTorch 等框架或工程生态相关引用，但不要把它们写成本文核心理论贡献，只作为实现和扩展参考。
5. 全文引用编号要与参考文献列表对应，避免正文有编号但列表缺失，或列表有文献但正文未引用。

七、实验结果引用口径

请优先读取并引用以下最终结果包，不要随意使用旧的单次运行文件：

1. 总体审计与总结：
   - outputs/reports/final_thesis/final_coverage_audit.json
   - outputs/reports/final_thesis/final_missing_outputs.json
   - outputs/reports/final_thesis/final_thesis_summary.md
   - outputs/reports/final_thesis/final_summary.csv
   - outputs/reports/final_thesis/final_summary.json

2. mock 实验：
   - outputs/reports/final_thesis/mock/mock_summary.csv
   - outputs/reports/final_thesis/mock/mock_parameter_scan_ldp.csv
   - outputs/reports/final_thesis/mock/mock_parameter_scan_noise.csv
   - outputs/defense/final_thesis/mock/

3. 真实公开数据实验：
   - outputs/reports/final_thesis/real/real_summary.csv
   - outputs/reports/final_thesis/real/real_parameter_scan_ldp.csv
   - outputs/reports/final_thesis/real/real_parameter_scan_noise.csv
   - outputs/defense/final_thesis/real/uci_har/
   - outputs/defense/final_thesis/real/kasteren/
   - outputs/defense/final_thesis/real/casas_hh101/

4. Cooja 节点级实验：
   - outputs/reports/final_thesis/cooja/cooja_summary.csv
   - outputs/reports/final_thesis/cooja/cooja_overhead_summary.csv

5. 图表：
   - outputs/reports/final_thesis/figure_table_list.md
   - outputs/figures/final_thesis/

已完成实验覆盖情况：
mock 合成数据 36/36，真实数据集 108/108，Cooja 汇总 6 条，final_missing_outputs.json 为空数组。论文中若引用“全部实验已完成”，请使用这个审计口径。

八、写作风格要求

1. 保持本科毕业论文的正式中文学术表达，避免口语化。
2. 不要频繁写“仓库”“脚本”“项目里补充了”等工程口吻，除非是在说明可复现性。
3. 不要夸大结论。尤其不要声称已经完成真实能耗、真实时延或真实带宽开销量化；Cooja 当前只能作为窗口数量代理和节点级日志分析。
4. 不要直接说某个真实数据集“效果最好/最差”，除非限定为同一数据集内部比较。
5. 所有实验结论都要说明适用口径：数据集、模型、攻击者模式、防御方法、参数范围。
6. 修改时尽量保留原论文结构，不要重写成完全不同主题。

九、请输出的内容

请按以下顺序输出：

1. 修改后的论文全文，保持原有章节结构。
2. 单独列出“已完成的关键修改清单”，逐条对应老师修改建议。
3. 单独列出“仍需人工确认的地方”，例如目录页码、学校格式模板、参考文献格式细节、图表编号是否与 Word 自动编号一致。
4. 若无法直接编辑 Word 文件，请输出可直接粘贴回论文的章节文本，并标明每段应替换的位置。
```
