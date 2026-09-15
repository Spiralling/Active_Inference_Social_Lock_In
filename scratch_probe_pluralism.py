"""Scratch probe: can MULTIPLE paradigms coexist when the evidence cannot decide
between them? Two equal camps start certain of OPPOSITE paradigms (phlogiston /
oxygen via L_init); the world stays in the phlogiston regime throughout. The first
probe (sigma_o sweep) showed noise only STRETCHES the collapse -- weak evidence
still accumulates. The decisive axis is whether the DISCRIMINATING experiment is
run at all: attention 0 on the rows where the candidates disagree makes them
prequentially identical (true underdetermination), so only social pooling moves
q(m). Cross that with contact (complete vs sparse two-community graph). Expected:
experiment run -> the world picks the winner everywhere; experiment NOT run + dense
contact -> pooling blurs both camps toward a torn 50/50; NOT run + sparse contact
-> both paradigms survive as camps. Throwaway -- promote into an experiment if the
quadrants are clean. Run from the repo root: python scratch_probe_pluralism.py"""


def main():
    import numpy as np
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.structural import rival as rv
    from src.structural import graphs
    from src.structural.phlogiston import (StructuralConfig, phlogiston_bn,
                                           gravimetric_H, gravimetric_rows,
                                           phi_true_at, DISAGREEMENT_NODES)
    from experiments.rival_kuhn import oxygen_bn

    N, T, SEEDS, CONF = 24, 400, (0, 1, 2), 4.6        # CONF: logit, q ~ 0.99
    half = N // 2
    out_rows = {}
    cfg = StructuralConfig(regime_schedule="step", t_shift=T + 1, n_steps=T,
                           sigma_o=1.0)                 # never flips: phlogiston world
    inc = phlogiston_bn(cfg, conviction=2.0).to_info()
    oxy = oxygen_bn(cfg).to_info()
    H = gravimetric_H(cfg)
    m = int(H.shape[0])
    rows_lab = gravimetric_rows(cfg)
    disc = [i for i, r in enumerate(rows_lab)
            if ("mass_balance" in r) or (r in DISAGREEMENT_NODES)]
    phis = jnp.asarray(np.stack([np.asarray(phi_true_at(cfg, t))
                                 for t in range(T)]))
    cfgr = rv.RivalConfig(Pi0=jnp.stack([inc.Pi, oxy.Pi]),
                          h0=jnp.stack([inc.h, oxy.h]), H=H,
                          sigma_o=1.0, omega=0.9, alpha_m=0.02)
    L0 = np.zeros((N, 2))
    L0[:half, 0] = CONF                                 # camp A: phlogiston
    L0[half:, 1] = CONF                                 # camp B: oxygen
    w_run = np.ones((N, m))
    w_norun = np.ones((N, m))
    w_norun[:, disc] = 0.0                              # nobody runs the crucial expt
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.8), sharex=True, sharey=True)
    for i, (w_att, att_lab) in enumerate(
            ((w_run, "the discriminating experiment IS run"),
             (w_norun, "the discriminating experiment is NOT run"))):
        for j, (gname, graph) in enumerate(
                (("complete", graphs.complete(N)),
                 ("sparse two-community ($\\beta$=0.01)",
                  graphs.community([half, half], intra=1.0, inter=0.01, seed=0)))):
            W = graph.trust_W()
            A_self = jnp.asarray(np.asarray(graph.A) + np.eye(N))
            qs = []
            for s in SEEDS:
                r = rv.run_rival(cfgr, W, A_self, jnp.asarray(w_att), phis,
                                 seed=s, L_init=jnp.asarray(L0),
                                 track_means=False)
                qs.append(r["q_t"][:, :, 1])            # (T, N)
            q = np.stack(qs)                            # (S, T, N)
            ax = axes[i, j]
            for blk, lab, col in ((slice(0, half), "camp A (starts phlogiston)",
                                   "#922b21"),
                                  (slice(half, N), "camp B (starts oxygen)",
                                   "#2c6fbb")):
                mtr = q[:, :, blk].mean(axis=(0, 2))
                std = q[:, :, blk].mean(axis=2).std(axis=0)
                ax.fill_between(np.arange(T), mtr - std, mtr + std,
                                color=col, alpha=0.2)
                ax.plot(mtr, color=col, lw=2.0, label=lab)
            ax.axhline(0.5, color="#7f8c8d", ls="--", lw=0.7, alpha=0.7)
            ax.set_title(f"{att_lab}  |  {gname}", fontsize=9)
            ax.set_ylim(-0.04, 1.04)
            end_a, end_b = q[:, -1, :half].mean(), q[:, -1, half:].mean()
            out_rows[(att_lab, gname)] = (float(end_a), float(end_b))
            ax.text(0.98, 0.5, f"end: A={end_a:.2f}  B={end_b:.2f}",
                    transform=ax.transAxes, ha="right", fontsize=8)
    axes[0, 0].legend(fontsize=8, loc="center right")
    for ax in axes[1]:
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("camp mean q(oxygen)")
    fig.suptitle("when can two paradigms coexist? camps start on OPPOSITE paradigms; "
                 "the world is phlogiston throughout;\nunderdetermination = the "
                 "experiment that would decide is never run", fontsize=11)
    plt.tight_layout()
    from pathlib import Path
    out = Path("results/_pluralism_probe")
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "probe_pluralism.png", dpi=130)
    print("\nend-state camp means (A, B):")
    for k, v in out_rows.items():
        print(f"  sigma={k[0]}, {k[1]}: A={v[0]:.3f} B={v[1]:.3f}")
    print(f"figure -> {out / 'probe_pluralism.png'}")


if __name__ == "__main__":
    main()
