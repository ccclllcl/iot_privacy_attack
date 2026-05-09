# Thesis Text Sync Suggestions

These notes align the thesis text with the final delivery package. They do not add new experiment results.

## Section 5.2: adaptive_ldp ablation

Old phrasing to update:

> 后续可以开展消融实验……

Suggested phrasing:

> 当前结果包已补充 profile 级消融汇总；更细粒度真实部署消融、真实边缘预算约束和硬件部署条件下的消融仍可作为后续工作。

## Section 4.3.3: parameter scans

If the text still says only "UCI HAR 参数扫描已补齐", update it to:

> 最终结果包已经补齐 mock 与 UCI HAR、Kasteren、CASAS 的参数扫描矩阵；正文表格仍以 mock LSTM fixed_attacker 为主要展示口径，其余完整扫描见 final_thesis 结果包。

## Cooja overhead wording

Keep Cooja wording cautious:

- fixed/retrain attack accuracy can be reported.
- Do not claim that real energy consumption has been measured.
- Do not claim that real end-to-end delay has been measured.
- If packet/byte/IAT fields are NaN, interpret them as unavailable log fields, not as an unrun experiment.
