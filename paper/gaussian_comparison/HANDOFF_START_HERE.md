# Updated IWAI camera-ready handoff: matched Gaussian comparison

This package adds the requested matched Gaussian / gated Gaussian / candidate comparison and retains the earlier communication study. No manuscript has been edited.

1. Open the **[revised nine-page figure pack](paper/gaussian_comparison/figures/figure_pack.pdf)**. The two new figures are on pages 8–9; the previous seven plots are preserved.
2. Read **[new results and interpretation](paper/gaussian_comparison/RESULTS.md)** and **[integration guidance](paper/gaussian_comparison/INTEGRATION_GUIDE.md)**.
3. Copy the **[new LaTeX figure blocks](paper/gaussian_comparison/figures/integration_blocks.tex)**, vector assets and exact captions as appropriate.
4. Consult **[mathematical methods and pseudocode](paper/gaussian_comparison/METHODS.md)**, **[independent validation](paper/gaussian_comparison/VALIDATION.md)** and **[reproduction instructions](paper/gaussian_comparison/README.md)**.
5. The **[earlier communication results](paper/communication_experiments/RESULTS.md)** and all their source, data reductions and audits remain included.

Plain Gaussians already retain a systematic prediction difference in this driven, anchored setup. The social gate reduces cross-group influence and modestly enlarges the difference while increasing predictive loss. Candidate models with private weights retain opposed choices, with a larger predictive cost for gas observers. Sharing model support removes that opposition. Continuous prediction distance, explicit theory choice and accuracy must remain separate claims.

The comparison uses identical observation streams, retention and base graph settings. A second Gaussian reference construction checks sensitivity to compression of the candidate mixture. All four new conditions ran to 5,000 updates across 50 paired seeds. The original candidate trajectories are reused with exact provenance, not advertised as newly independent confirmations.

All nine figures use actual LaTeX Computer Modern fonts. PDF/SVG are vector manuscript assets; PNGs are previews. Numerical plots use Matplotlib with the orx publication style, not TikZ. The compact package includes all plotting inputs, code, per-seed summaries, captions, reference records and independent checks; no raw-data upload is needed for paper writing or figure regeneration.

The optional new Gaussian raw archive adds `gaussian_raw_data/`. The earlier complete-data archive adds `raw_data/` and `legacy_network/`. Both use this same package-root name, so extracting them into the same parent directory combines the data for complete reanalysis. Representative chunks in the compact package are examples, not substitutes for the full raw inputs. `MANIFEST.json` and `VERIFY_PACKAGE.py` verify the compact payload.
