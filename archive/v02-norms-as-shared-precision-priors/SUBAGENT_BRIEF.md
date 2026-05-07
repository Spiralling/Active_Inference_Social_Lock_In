# Subagent Brief Template (per-section)

This template is filled in per-section and embedded directly in each Agent
spawn prompt. It is the canonical contract subagents read before writing.

## Fixed conventions (all sections)

- LaTeX class is `\documentclass[11pt,a4paper]{article}`. Mirror the style
  of `rest-focus-unification/paper/main_v2.tex`.
- Bibliography is `natbib`: `\citep{key}` for parenthetical, `\citet{key}`
  for in-text. Use ONLY citation keys from `values-salience.bib` —
  if a new key is needed, add it to `section_N_bib_additions.bib` (NOT to
  `values-salience.bib` directly).
- Notation macros are pre-defined in `notation.tex`. Use them. Do NOT
  redefine them. Do NOT add new ones in your section file.
- Cross-references to other sections that may not exist yet: use
  `\todo{xref to §X}` rather than `\ref{}` to avoid compile failures.
- Do NOT write the abstract, introduction, conclusion, or discussion —
  those are Jonas's. Write only your assigned section.
- Inline TikZ figures inside your section file. PGFPlots is loaded.
- `graphs` + `graphdrawing` libraries are available but require LuaLaTeX.
  If your figure uses them, add a comment noting the LuaLaTeX dependency.

## Output format

Write exactly two files:
1. `paper/section_N.tex` — your section, self-contained, beginning with
   `\section{...}\label{sec:...}` and ending with no trailing whitespace.
2. `paper/section_N_bib_additions.bib` — any new BibTeX entries needed,
   following the `firstauthorYearKeyword` key convention.

Then return a 200-word max summary covering: (a) length actually delivered,
(b) any anchors that didn't fit and were dropped, (c) any framing decisions
worth flagging to Jonas, (d) compile-time concerns (e.g. did you use the
`graphs` library? does the section need LuaLaTeX?).

## Read paths (relative to vault)

All paths under `Documents/Productivity/Obsidian/Research/Projects/Values-as-salience-priors/`:
- `lit-review/synthesis.md` — contribution shape, citation corrections
- `lit-review/lit-0X-*.md` — your assigned cluster file(s)
- `paper/notation.tex` — macros you can use
- `paper/main.tex` — skeleton showing how your section gets `\input`'d
- `paper/values-salience.bib` — pre-seeded keys

## Universal "do not do" list

- Do NOT redefine notation macros
- Do NOT edit `notation.tex`, `main.tex`, or `values-salience.bib`
- Do NOT write abstract/intro/conclusion/discussion
- Do NOT cite group selection in support of values transmission (cite
  cultural evolution mechanisms — Henrich, Heyes — instead)
- Do NOT use EFE-heavy formalism in §3; prefer VFE + expected precision
- Do NOT skirt Ransom et al. 2020 (§3 only) — engage it directly
- Do NOT use `\cite{}` (use `\citep{}` or `\citet{}`)
