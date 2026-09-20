# IWAI communication experiments — camera-ready handoff

The requested suite is implemented, run and independently checked. The original revised combustion model is preserved. No manuscript was edited.

1. Read **[results and interpretation](paper/communication_experiments/RESULTS.md)**.
2. Open **[the captioned figure pack](paper/communication_experiments/figures/figure_pack.pdf)**.
3. Use **[the integration guide](paper/communication_experiments/INTEGRATION_GUIDE.md)** and **[LaTeX figure blocks](paper/communication_experiments/figures/integration_blocks.tex)** to update the paper.
4. Consult **[methods and equation-linked pseudocode](paper/communication_experiments/METHODS.md)** and **[validation](paper/communication_experiments/VALIDATION.md)** for technical details.
5. See **[README](paper/communication_experiments/README.md)** for regeneration and full-data commands.

The main new result is the four-arm communication comparison: model-weight sharing alone produces adoption in the restricted group, while both channels accelerate it and improve that group's prediction. Opposing camp preferences persist over the tested horizon with private model weights, but matching gas reference variance, increasing retention or strengthening conditional exchange removes the effect. Enabling model-weight sharing also removes it, without altering conditional beliefs or local scores.

The compact ZIP contains seven vector figure assets with actual LaTeX Computer Modern fonts, captions/source, all plotting inputs and per-seed summaries, exact protocols, code, reference records and independent audits. It is sufficient for paper writing and figure regeneration. The optional complete-data ZIP adds the full raw and precision histories plus the old network regression arrays. Extract it over this package only if complete raw reanalysis is needed.

Do not infer unrestricted two-channel coexistence, causal-graph discovery, or a universal benefit from diversity. The controls and prediction costs are part of the result. Do not reuse stronger historical claims from archived engine comments.
