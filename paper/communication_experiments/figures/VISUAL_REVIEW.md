# Figure review

All seven plots were inspected visually at their native 4.8-inch layout after
orx's collision audit. All labels and equations use real LaTeX Computer Modern.
The compiled caption pack was also inspected, including the largest eight-panel
supplement page. Automated validation separately checks PDF page bounds, embedded
fonts, absence of raster images, exact width, input hashes and bounded data.

The review corrected overlapping legends, an overlong control label and a
clipped supplement axis label. Individual endpoint ECDFs were replaced with
all-value strips because concentrated probabilities at zero and one obscured
the four arms on the frame borders. The strips include every value; only their
horizontal display positions are jittered deterministically. The intervention
main figure now shows the full gas-group continuation and the prespecified first
100 updates; the unchanged mass-group trajectory remains in its prediction
supplement. These are presentation changes and do not change any experiment,
window, seed, metric, numerical table or statistical inference.

The narrow bands in several panels are the actual whole-seed bootstrap
intervals, often smaller than the rendered line or marker. The captions specify
this construction and distinguish descriptive agent strips from inferential
seed intervals. No curve was smoothed and no outcome-based illustrative seed
was selected. The supplementary individual traces use the prespecified seed
10000, regardless of how clearly it exhibits the effect.
