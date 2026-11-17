# Architecture

- `data.py` validates M1 openings, maps New York sessions, computes completed-bar inputs, and holds outcomes separately.
- `primitives.py` carries the frozen HAR transform, timestamp handling and QLIKE primitives. `frozen.py` validates external model hashes and implements the frozen prior-session calibration fold.
- `ledger.py` binds each file to input provenance and model/protocol identity. Transactions commit complete sessions. Append-only triggers, foreign keys and exact session signatures support recovery and identical retries.
- `scoring.py` and `evaluation.py` validate matched outcomes and produce paired QLIKE, aggregate calibration, additive errors, historical breakdowns and session-block uncertainty.
- `cli.py` contains the four local entry points. It requires neither the original research tree nor a broker SDK.
