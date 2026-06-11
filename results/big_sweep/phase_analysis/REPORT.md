# Big-sweep phase reanalysis

Re-read of the existing 2207 healthy runs in `results/big_sweep/rows.csv` (no new simulation).

## Fig 1 -- `fig1_transition.png` (+ `_trajectories.png`)
**The consensus--pluralism boundary behaves like a phase transition, not a smooth dial.** Along
the social-coupling axis (stable omegas pooled, 20-25 runs/point), three independent signatures
align in the same critical window (inter ~ 0.2): the order parameter (structural divergence)
falls, run-to-run susceptibility peaks (0.16, vs 0.005 when disconnected), and
settling time peaks (critical slowing down). The fourth panel shows the raw fates: at the
critical coupling, identical parameters fan out between the consensus and pluralism branches.
The realized graph connectivity barely predicts which branch a run takes (|corr| < 0.3 inside
the critical cells), so the spread is dynamical, not just graph percolation.

## Fig 2 -- `fig2_lockin.png`
**Lock-in is never free: it is either bought by infinite memory or earned by conviction.** The
full omega x conviction C map at inter=0 (changing-epochs world, omega < 1) shows lock-in =
0.00 everywhere: with any forgetting and tilt <= 4, every disconnected community re-tracks
every epoch. Across the whole random tier the structure is exactly the paper's claim, at scale:
with omega = 1 (no forgetting) lock-in is essentially automatic even at tilt 0 (the memory
wall); with omega < 1, lock-in stays near zero until conviction tilt ~ 2 and then rises -- the
earned-threshold regime. *Caveat*: tier-2 cells are single-seed and confounded across the other
random axes; the two-regime shape is the robust read-out, exact thresholds are not.
