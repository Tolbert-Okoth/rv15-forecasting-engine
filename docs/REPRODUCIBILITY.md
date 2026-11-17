# Reproducibility

Install the project and use the commands in the README. The standalone demo generates deterministic synthetic M1 prices. Its model identities, population and numeric outputs are reproducible with the supported environment. Database recording times are contemporary and can differ between runs. A second run on the same output directory exercises idempotent replay.

The optional CSV interface verifies an external frozen manifest against the original C0/C2 identities and bucket fingerprint. It builds its own chronological sample. Reproducing the archived 68,291 rows and LATE01 source-event subset additionally requires the original private archives, frozen population rules and timing evidence. The public package does not claim this archive-level reproduction.

The release verification used an installed wheel, Python isolated mode where applicable, a working directory outside the source and original research trees, and preinstalled dependency versions. The Windows commands were exercised in that setting. POSIX commands are supplied for standard Python environments and were not executed on Windows.
