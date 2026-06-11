"""Multi-candidate structure search: the candidate space searched, not given.

The limitations section calls model expansion the model's narrow waist: "Everything
revolutionary in the model passes through one move: waking a single, pre-allocated slot
(oxygen), wired to a fixed candidate set, in the direction of the leading eigenvector of a
windowed residual... a real discovery process entertains many structures at once -- several
hubs, new couplings among existing commitments, new observation channels -- and the rank-one
reading of the residual is exact only for a single hidden common cause."

This module implements the searched version for the two structure families the model can
price in closed form, all on the SAME ledger:

* **several hubs** -- the top-``k`` eigenpairs of the windowed residual are competing
  common-cause candidates (``action.propose_hubs_topk``), each scored by the bordered-model
  log Bayes factor (``action.expansion_score``) against its own pre-allocated slot;
* **new couplings among existing commitments** -- the largest off-diagonal residual entries
  are candidate direct edges (``action.coupling_candidates``), scored by the same
  log-Bayes-factor construction (``action.coupling_score``) with NO epistemic subsidy (no
  new latent).

Coupling magnitudes and hub patterns are ESTIMATED from the residual (eigenpairs / entries),
not selected from a fixed set. New observation channels remain out of scope: they enlarge
the likelihood itself (the same Eq. (5) line the action module brackets as the Knightian
frontier).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from src.structural.action import (MoveScore, coupling_candidates, coupling_score,
                                   expansion_score, propose_hubs_topk)
from src.structural.belief import GaussianBeliefNet


@dataclass(frozen=True)
class CandidateConfig:
    """The search width. ``k_hubs=1, top_couplings=0`` reproduces the legacy single-
    proposal pipeline (one eigenvector, no edge candidates).

    ``accept`` : ``"best"`` applies only the highest-scoring positive candidate per arrival;
                 ``"all_positive"`` applies every candidate the ledger accepts.
    """

    k_hubs: int = 1
    top_couplings: int = 0
    accept: str = "best"

    def __post_init__(self):
        if self.accept not in ("best", "all_positive"):
            raise ValueError(f"accept must be 'best'|'all_positive', got {self.accept!r}")


@dataclass(frozen=True)
class ScoredCandidate:
    """One candidate structure, priced. ``kind`` is ``"hub"`` (payload: the HubProposal,
    target: the slot name) or ``"coupling"`` (payload: ``(node_a, node_b, magnitude)``,
    target: ``"a~b"``)."""

    kind: str
    target: str
    score: MoveScore
    payload: Any


def search_candidates(net_post: GaussianBeliefNet, net_prior: GaussianBeliefNet,
                      neighbours: tuple[str, ...], residual: jax.Array,
                      cfg: CandidateConfig, slot_names: tuple[str, ...],
                      hub_self_prec: float = 2.0) -> list[ScoredCandidate]:
    """Price every candidate structure on one ledger; return them best-first.

    ``slot_names``: the pre-allocated dormant slots available for hub candidates (candidate
    ``m`` is scored against slot ``m``; at most ``len(slot_names)`` hubs are entertained).
    ``net_post`` / ``net_prior`` are the agent's current net and anchor on the CONCEIVED
    basis (the slots excluded) -- exactly what the single-proposal pipeline feeds
    ``expansion_score``."""
    out: list[ScoredCandidate] = []
    hubs = propose_hubs_topk(net_post, neighbours, residual=residual,
                             k=min(cfg.k_hubs, len(slot_names)))
    for m, prop in enumerate(hubs):
        ms = expansion_score(net_post, net_prior, prop, slot_names[m],
                             hub_self_prec=hub_self_prec)
        out.append(ScoredCandidate(kind="hub", target=slot_names[m], score=ms,
                                   payload=prop))
    for (a, b, mag) in coupling_candidates(residual, neighbours, cfg.top_couplings):
        ms = coupling_score(net_post, net_prior, (a, b), mag)
        out.append(ScoredCandidate(kind="coupling", target=f"{a}~{b}", score=ms,
                                   payload=(a, b, ms.detail["magnitude"])))
    out.sort(key=lambda c: c.score.score, reverse=True)
    return out
