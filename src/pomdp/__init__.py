"""Categorical fixed-state epistemic POMDP — the David-Hyland-reframed rebuild.

Phase 0 (de-risk): ``social_fold`` (0b go/no-go + the 0.5 environment-coupled
fold gate) and ``jacobian_probe`` (0c monostability of the old substrate).

Phase 1 (scaffold, this build):
  - ``gen_model``  : per-paradigm likelihoods A_theta(o|s,a), B=identity,
                     C, D, and the belief-utility vector U(theta).
  - ``agent_pop``  : the batched categorical AIF agent (state inference + EFE
                     with the belief-utility term) over the trust graph.
  - ``step``       : the inter-agent cycle (emit -> route -> infer -> act).
  - ``observables``: paradigm occupancy, EFE/q(pi), polarization, capture.

Hidden state theta = which paradigm is true (B = strict identity: fixed truth,
absorbing capture). Actions are *measurements* (which experiment to run), not
control. Belief-utility enters policy evaluation only, never the belief update.
"""
