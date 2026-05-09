# Cooja Limitations

- Cooja outputs can be used for fixed/retrain attacker accuracy reporting.
- Cooja currently does not provide real energy measurements.
- Cooja currently does not provide real end-to-end delay measurements.
- Radio/app log paths may point to local WSL-exported files; those paths document the local evaluation source and are not portable reproduction paths.
- Current radio logs do not distinguish dummy packets from real packets, so dummy packet and byte ratios are reported as null.
- Packet, byte, and IAT fields reported as NaN indicate unavailable log fields, not an unrun experiment.
- `cooja_overhead_summary.csv` remains a window-count proxy, not measured energy or latency.
