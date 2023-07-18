## Instructions for running code

We recommend not running this code, as it will take multiple hours to run. However, if you would like to run this code, you do so as follows. First, replace the lines in `makefile` that say `BASEDIR=[path here]` and `READWRITE_DIR=[path here]` with the path to the current directory. Then, type 
```
make compute_runtime
make compute_corrcoef
```
Let this code run for a few hours. Once this code runs, type `make plot_accuracy_runtime` to show the resulting cost-accuracy plot. If the runtime was significantly different than my most recent result, the grid resolution labels may not be placed correctly.

## Instructions for interpreting results

We are not the only authors to replicate this result. The article [Dresdner (2022)](https://arxiv.org/abs/2207.00556) replicates the main result on GPU using a stronger pseudo-spectral baseline, and finds that the stronger baseline is faster than the original ML-based solver.

Here, we replicate this result using a DG method on CPU. The resulting cost-accuracy plot is shown in the file `data/runtime_corr.png`. We compare the runtime and resolution of the DG method, and find that the DG method is faster than a PS method for sufficiently low accuracy, but as the accuracy increases the PS method becomes more efficient. We also see (consistent with previous experiments we have conducted) that a DG method gives the same accuracy as the FV method at about 10-11x coarser resolution.