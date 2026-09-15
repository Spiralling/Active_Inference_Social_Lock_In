# Port kit: today's figure swap → Overleaf (IWAI_26_Paradigm)

Five edits + one file upload. ~10 minutes. Do them in order, then recompile and
check the page count (must stay ≤ 12 main / ≤ 12 appendix; step 6 is the fix if
it overflows).

## 1. Upload the figure

Upload `paper/figures/fig_kuhn_trajectories_by_topology.png` (from this repo) to
the Overleaf project's figures folder (wherever `fig_abc_variance.png` lives).

## 2. Delete the old Fig. 9 block (Appendix C)

Find the figure environment containing `fig_connectivity_boundary.png` (caption
starts "Where structural pluralism dies as mixing rises") and delete the whole
`\begin{figure}...\end{figure}` block.

## 3. Fix the two dangling references to it

(a) Main text §4.1, sentence ending "...they retain 94% of the divergence where
fixed trust keeps 10%—Fig.~\ref{fig:res-wiring}, Appendix~\ref{app:reliability}.)"
→ delete "Fig.~\ref{fig:res-wiring}, " so it reads "...keeps 10%—Appendix~\ref{app:reliability}.)"
(their copy may use a comma instead of the em-dash; keep their punctuation).

(b) Appendix C, "The experiments, interpreted" paragraph: "...surviving
cross-community structure (Fig.~\ref{fig:res-wiring}): fixed trust retains..."
→ "...surviving cross-community structure: fixed trust retains..."

## 4. Insert the new paragraph + figure (Appendix C)

Paste this immediately BEFORE the paragraph "Scope, stated plainly." (i.e. where
the old figure block was):

```latex
\paragraph{The transition's shape is the wiring's signature.}
A final run replays the Kuhnian transition of \S\ref{sec:res-kuhn} where it must
\emph{spread}---$N{=}24$, a vanguard of $4$ keeps full attention on the disconfirming
experiment, the rest self-censor it ($\gamma{=}0.99$, a trickle)---on the four
reference contact graphs (Table~\ref{tab:dials}; $50$ seeds). Every community
completes the transition \citep{zollman2010}, but the \emph{same} world flip arrives
as four different events (Fig.~\ref{fig:res-trajtopo}): the median conservative
converts at $t{=}220/232/453/488$ on complete/wheel/community/cycle, inter-agent
spread $4/11/57/102$ steps. The wheel keeps pace with the complete graph despite a
$\lambda_2$ an order of magnitude smaller---the vanguard holds its hub: centrality,
not density. The shape of a paradigm shift is a property of the contact graph.

\begin{figure}[h!]
\centering
\includegraphics[width=0.74\linewidth]{fig_kuhn_trajectories_by_topology.png}
\caption{The same Kuhnian transition on four contact graphs, on one canvas. Bold:
conservatives' mean $q(\mathrm{oxygen})$ ($50$ seeds $\pm$ s.d.); thin: one seed's
$20$ conservative agents; the world turns at the dotted line; vanguard nodes are
outlined in the glyphs. Complete and wheel flip together and early; the community
graph converts as a staircase after the bridge delay; the cycle as a travelling
wave.}
\label{fig:res-trajtopo}
\end{figure}
```

NOTE: if their section label for §4.2 is not `sec:res-kuhn`, fix the `\S\ref`.
If `\citep` is undefined there, use their citation macro for zollman2010.

## 5. The one-clause pointer in §4.2 "Boundaries, stated plainly"

"...existence proofs that the persistence was representational, not sweeps of
social topologies" → append: " (the one such sweep---this transition replayed
across four reference contact graphs---is in Appendix~\ref{app:reliability})".

## 6. If the PDF lands on 29 pages (appendix = 13)

Apply these five width changes (content-neutral; this is exactly what brought
the repo copy back to 28):

| figure | old | new |
|---|---|---|
| fig_kuhn_lockin_boundary | 0.85 | 0.72 |
| fig_kuhn_fusion | 0.62 | 0.52 |
| fig_abc_variance | 0.52 | 0.47 |
| fig_schur_generic | 0.72 | 0.60 |
| fig_kuhn_trajectories_by_topology | 0.74 | keep 0.74 |

## 7. Optional (the cold-read panel's one named flaw)

Delete the two never-cited bibliography entries: `catal2024` ("[2] Çatal et al.")
and `albarracin2024protentions` ("[46] shared protentions"). Zero layout risk;
numbering renumbers itself.

## Sanity check before upload

- Page count ≤ 28 with refs at 4pp (12 main / 4 refs / 12 appendix).
- Fig. 9 is now the trajectory figure; caption mentions 50 seeds.
- No `??` in the PDF (search it) — would mean a broken `\ref` from step 3/4.
