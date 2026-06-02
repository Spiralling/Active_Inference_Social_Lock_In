# Archived background-draft predecessors

The current background draft is a single consolidated document:
`notes/background_drafts/background.tex` (2026-05-30 structural rewrite — Phenomenon →
Desiderata → Steps 1–7 → Central Result → Discussion → Appendix). It `\input`s nothing and
reads top-to-bottom.

These are its superseded predecessors, kept for provenance:

| File(s) | What it was |
|---|---|
| `S1_theory_laden.tex` … `S7_discussion.tex` | The seven section fragments that were merged into `background.tex`. Orphans — not in any compile graph. |
| `background_scalar_walkthrough.tex` | The earlier *scalar*-model walkthrough (2026-05-30), superseded by the structural rewrite. |
| `background_litreview.tex` | The lit-review-style survey draft that preceded the walkthrough. |
| `figures_old.tex` | The six TikZ figure blocks; the live ones are now inline in `background.tex`. |

The live draft compiles via `notes/background_drafts/_build/` (`python _prepare.py` then
`pdflatex main.tex`).
