# Model card

| Research identity | SHA-256 |
| --- | --- |
| Frozen C0 HAR | `efdf79e7d406521770b67c1de744fb3b2af2a082b5c9de0c1b9ebe77369afd48` |
| Frozen C2 adaptive calibration | `7426520c260a2de0e35749b8531c983aa2e7cbe42290d28cd1fe9771508fe5f1` |

C0 was trained on 142,498 eligible observations from 2016 through January 2023. The predictors are trailing 15- and 60-minute realised variance, previous completed weekday-session variance and a fitted quarter-hour seasonal profile. Forecasts use the original log-linear transform, variance floor and smearing factor. C2 applies a causal level correction based on matured prior sessions with a ten-session half-life, 20-session minimum and bounds [0.5, 2]. Both are point forecasts for the next 15 squared completed-close log returns.

The historical frozen coefficient and seasonal bundle is excluded. A separately authorized manifest can be verified against the immutable identities. The included demonstration has distinct model identifiers, uses arbitrary unfitted software constants, and retains the same algorithmic execution paths.

Use the application to study forecast losses, calibration and changing variance distributions. A variance point forecast does not supply a probability of a tail event or a signal for price direction.
