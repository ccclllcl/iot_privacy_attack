# Cooja 节点级开销缺口报告

本报告记录本轮补全前的 Cooja 开销缺口和补全方案。此前 `cooja_traffic_metrics.csv` 中 packet/byte/IAT、`dummy_packet_ratio`、`energy_metric_available` 和 `delay_metric_available` 主要为 `nan` 或 `False`，原因是旧 Cooja app/radio 日志没有显式区分 REAL 与 DUMMY 包，也没有导出 Energest 计数器。

## 缺失指标

- dummy/real 包比例：旧日志没有 `packet_type=REAL/DUMMY` 标签。
- packet overhead 与 byte overhead：旧日志不能可靠区分 dummy 包与真实业务包。
- 端到端时延：旧日志没有可匹配的发送序号、发送时间和接收时间。
- Energest 能耗估计：旧节点程序没有输出 `ENERGEST` 计数器。

## 节点程序补充内容

- 真实业务包和 dummy 包均携带 `magic/version/node_id/seq/packet_type/send_time_ms/payload_len`。
- 客户端发送时输出 `METRIC_TX type=REAL|DUMMY node=<id> seq=<seq> bytes=<bytes> time_ms=<time>`。
- 服务端接收时输出 `METRIC_RX type=REAL|DUMMY src=<id> seq=<seq> send_ms=<send_ms> recv_ms=<recv_ms> bytes=<bytes>`。
- 节点周期性输出 `ENERGEST node=<id> cpu_ticks=<...> lpm_ticks=<...> tx_ticks=<...> rx_ticks=<...> total_ticks=<...> time_ms=<...>`。

## 可补全指标

- dummy_packet_ratio
- packet_overhead_ratio
- byte_overhead_ratio
- mean_delay_ms / median_delay_ms / p95_delay_ms
- energy_mj
- energy_overhead_ratio_vs_baseline

## 口径限制

- `energy_mj` 是 Cooja/Contiki-NG Energest 仿真估计，不是硬件功耗仪测量。
- `mean_delay_ms` 和 `p95_delay_ms` 是 Cooja simulation-time 端到端时延，不是真实部署时延。
- 如果某次日志缺少结构化字段，解析器会写入 `null` 和 `unavailable_reason`，不会伪造数值。
