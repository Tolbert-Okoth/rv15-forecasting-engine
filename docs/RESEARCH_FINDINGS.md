# Retrospective research findings

| Comparison | QLIKE improvement |
| --- | ---: |
| Primary frozen HAR versus EWMA | 23.37% |
| Additional 2026 HAR versus EWMA | 18.24% |
| Additional 2026 C2 versus HAR | 2.94% |
| LATE01 tick-matched C2 versus HAR | 1.90% |

The original engine reproduced 68,291 origins and 18 upstream hashes. The maximum absolute QLIKE difference from the original was 1.42 × 10⁻¹³. In the full 2026 sample, HAR and C2 QLIKE were 0.299738 and 0.290930. Aggregate calibration was 0.807 and 0.949. The LATE01 result used 8,491 source-event-matched origins and represents a subset of the 2026 population. These results share data and research history.

C2's gain was concentrated in extremes. QLIKE was slightly worse outside the frozen tail union in full 2026 and in the matched subset. Future performance is unknown. The original private 1,909-outcome synthetic demo used the frozen models. The standalone software demo uses different, unfitted constants and provides no independent empirical confirmation.
