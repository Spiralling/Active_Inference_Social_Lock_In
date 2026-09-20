# How to read the archived engine

The 46 Python files in `paper/four_variable_experiments/engine` are included unchanged to preserve the identity of the accepted adaptive implementation. Their comments document several earlier experiments and hypotheses; they are not the scientific claims of this handoff.

In particular, historical text in `src/structural/rival.py` describes a nonzero limiting log-odds gap as a “genuine fixed point.” A constant gap does **not** establish opposite candidate preferences: if the common log-odds component continues to drift, both groups can ultimately favor the same candidate. See equation (9) in `METHODS.md`, the independent stationary checks, and the new results.

The package also retains historical module names and optional utility/gating capabilities required by the original package imports. The new communication suite executes no belief-utility term, social reliability gate, model-weight gate or model-score decay. Its settings and scope are given by the frozen `protocol.json` and per-condition `setup.json` records.

For manuscript integration, use the new methods, measured results and captions. Do not promote historical engine comments or the unverified external pilot to results of this suite.
