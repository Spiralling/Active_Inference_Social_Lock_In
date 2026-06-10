"""Big resumable sweep over structural pluralism & paradigm lock-in.

A pre-registered, multi-axis sweep on the cosmology substrate with a FIXED metric battery,
resumable checkpointing, one tidy row per run, and a data-driven analysis pass. The point is
that conclusions come from the landscape (phase maps + anomaly flags), not from a narrative over
a handful of hand-picked runs.

Modules:
  space    -- the config space (axis grids) + tiered job generator (pure data).
  run_one  -- build scenario/graph/AgentSpec for one config, call run_simulation, return metrics.
  metrics  -- the metric battery (reused readouts + a few glue functions).
  runner   -- resumable runner: hash, skip-if-done, append a JSONL row + a small traj npz.
  analyze  -- phase-map heatmaps + anomaly flags + REPORT.md.
  run_big_sweep -- CLI entry.

Built entirely on stable public APIs in src/structural/ (run_simulation, cosmology_scenario,
graphs, observables, shells); touches no src/.
"""
