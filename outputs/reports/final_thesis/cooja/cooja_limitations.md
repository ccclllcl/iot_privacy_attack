# Cooja Limitations

- Cooja outputs in this package do not include real energy or delay measurements.
- `cooja_overhead_summary.csv` is a window-count proxy, not measured energy or latency.
- Radio log does not distinguish dummy packets from real packets, so dummy packet and byte ratios are reported as null.
