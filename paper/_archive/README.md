# Archived paper drafts

Standalone writeups and figure scratch that are **not** part of the live LNCS submission
(`paper/main.tex`, which inputs only `notation.tex`, `authors-deanon.tex`, and
`sections/01`–`06`). Kept for provenance. PDFs are date-prefixed.

| File(s) | Date | What it was |
|---|---|---|
| `pomdp_paradigm_model.tex` / `2026-05-21_pomdp_paradigm_model.pdf` | 2026-05-21 | Pre-pivot standalone categorical-POMDP model writeup. |
| `_trustfig.tex`, `_trustfig_T.tex`, `_trustfig_adj.tex`, `_trustfig_graph.tex`, `_gen_trustfig.py` | 2026-05-24 | Trust-figure fragments + generator, `\input` only by `pomdp_paradigm_model.tex` (moved here with it so it still compiles). |
| `structured_belief_revision.tex` / `2026-05-24_structured_belief_revision.pdf` | 2026-05-24 | Standalone structured-belief-revision draft (never in main.tex's input chain). |
| `_tikz_test_loop.tex` / `2026-05-18_tikz_test_loop.pdf` | 2026-05-18 | TikZ rendering scratch. |

To compile an archived draft, run `pdflatex` from inside this directory (the trustfig
fragments resolve relative to `pomdp_paradigm_model.tex`).
