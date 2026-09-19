"""Model libraries -- the reusable science behind each main model.

One module per model (``phlogiston``, ``cosmology``, ...). Pure compute and
builders: no matplotlib, no file I/O, no ``main`` -- so a notebook can import them
without side effects (in particular, importing one never flips the matplotlib
backend). The experiments in ``experiments/`` compose these into runnable, saved
figures with provenance.
"""
