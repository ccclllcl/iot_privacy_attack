# Cooja 节点级开销补全报告

- 生成时间: `2026-05-11T10:54:25`
- 是否修改论文 Word: 否
- 日志字段: `METRIC_TX`、`METRIC_RX`、`ENERGEST`
- 能耗口径: Contiki-NG Energest 仿真估计，不是硬件功耗仪测量。
- 时延口径: Cooja 仿真时间下的 REAL 包端到端时延。
- 覆盖方法: `baseline, dummy_noise, dummy_ldp, dummy_adaptive_ldp`
- 覆盖 seed: `42, 123, 2026`
- 缺失项数量: `0`

## 生成产物

- `outputs/summaries/final_thesis/cooja/cooja_overhead_metrics.csv`
- `outputs/summaries/final_thesis/cooja/cooja_overhead_metrics.json`
- `outputs/summaries/final_thesis/cooja/cooja_traffic_metrics.csv`
- `outputs/summaries/final_thesis/cooja/cooja_overhead_summary.csv`
- `outputs/figures/summaries/final_thesis/thesis_fig4_13_cooja_overhead_metrics.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_14_cooja_dummy_ratio.png`
- `outputs/figures/summaries/final_thesis/thesis_fig4_15_cooja_energy_delay.png`

## 指标概览

```csv
method,seed_count,real_packet_count_mean,dummy_packet_count_mean,total_packet_count_mean,dummy_packet_ratio_mean,packet_overhead_ratio_mean,real_byte_count_mean,dummy_byte_count_mean,total_byte_count_mean,dummy_byte_ratio_mean,byte_overhead_ratio_mean,mean_iat_ms_mean,p95_iat_ms_mean,mean_delay_ms_mean,p95_delay_ms_mean,energy_mj_mean,energy_overhead_ratio_mean,delay_metric_available,energy_metric_available,metric_type,is_hardware_measurement
baseline,3,1064.0,0.0,1064.0,0.0,0.0,51072.0,0.0,51072.0,0.0,0.0,1675.7289792335353,4353.716666666666,39.27065007726771,76.1680000000216,752272.3434576001,,True,True,cooja_simulation_and_energest_estimate,False
dummy_noise,3,1063.0,209.33333333333334,1272.3333333333333,0.1644224774448029,0.19691761088264012,51024.0,10048.0,61072.0,0.1644224774448029,0.19691761088264012,1401.773728595721,4508.416666666664,41.51874107570399,82.16800000000015,752858.2628512001,0.0007847524770290452,True,True,cooja_simulation_and_energest_estimate,False
dummy_ldp,3,1069.3333333333333,367.3333333333333,1436.6666666666667,0.2555743250051861,0.3435114084534374,51328.0,17632.0,68960.0,0.2555743250051861,0.3435114084534374,1241.7890317593192,4203.7,41.46311729787541,82.96800000000802,753396.2765824,0.0015016495581371715,True,True,cooja_simulation_and_energest_estimate,False
dummy_adaptive_ldp,3,1066.6666666666667,279.3333333333333,1346.0,0.20749977215199342,0.26190932085022517,51200.0,13408.0,64608.0,0.20749977215199342,0.26190932085022517,1325.1740412944307,4240.133333333333,40.765289347908244,80.96800000004684,750774.8022048,-0.001993377069051864,True,True,cooja_simulation_and_energest_estimate,False
```
