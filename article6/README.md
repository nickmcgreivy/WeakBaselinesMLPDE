## Instructions for running code

Simply type `cd scripts` then type `python reproduce_hoyer_et_al.py` and wait 10 minutes or so for the 5 trials to run.

This produces a plot identical to `replicate_figure3c.png`.

## Instructions for interpreting results

We replicate the result in figure 3c. (Note that this article had a bug in the computation of Godunov flux, which affected the accuracy of WENO5.) See that the DG methods have the same accuracy as the WENO method at 4-8x coarser resolution, which matches the performance of the ML-based solver.