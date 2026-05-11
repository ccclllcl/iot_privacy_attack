# Cooja Limitations

- Cooja outputs can be used for fixed/retrain attacker accuracy reporting.
- Cooja overhead metrics now include dummy/real packet ratio, packet/byte overhead, Cooja simulation-time delay, and Contiki-NG Energest-based energy estimate.
- Energy values are simulation-level estimates based on Energest counters and current-draw configuration, not hardware power-meter measurements.
- Delay values are Cooja simulation-time end-to-end delays, not real deployment latency.
- Dummy/real packet ratios are computed from explicitly labeled METRIC_TX/METRIC_RX logs.
