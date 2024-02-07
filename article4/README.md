## Instructions for running code on GPU

This code should be run on GPU, though it will also run on CPU. This requires configuring JAX for GPUs, see [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html). 

This code runs in an hour or two on GPU, and much longer on CPU. To run the code, first replace the lines in `makefile` that say `BASEDIR=[path here]` and `READWRITE_DIR=[path here]` with the path to the current directory. Then, type `make compute_runtime_gputpu`. This will print a list of runtimes for the finite-volume (FV) baseline and the pseudo-spectral (PS) baseline. On an A100 GPU, this outputs
```
ML-CFD FV baseline, nx = 512
runtime per unit time: [0.2381835  0.23868346 0.23633456]
ML-CFD FV baseline, nx = 1024
runtime per unit time: [1.49418926 1.49033809 1.49134207]
ML-CFD FV baseline, nx = 2048
runtime per unit time: [11.36248708 11.40039062 11.36248732]
ML-CFD FV baseline, nx = 4096
runtime per unit time: [91.66669345 92.14138103 91.71203279]
ML-CFD FV baseline, nx = 8192
runtime per unit time: [840.11879325 841.74085355 840.39129853]
ML-CFD PS baseline, nx = 64
runtime per unit time: [0.08551478 0.08573127 0.07165051]
ML-CFD PS baseline, nx = 128
runtime per unit time: [0.15832877 0.14401126 0.14344049]
ML-CFD PS baseline, nx = 256
runtime per unit time: [0.31190848 0.3115592  0.31158614]
ML-CFD PS baseline, nx = 512
runtime per unit time: [1.03669143 1.03707528 1.03646612]
ML-CFD PS baseline, nx = 1024
runtime per unit time: [7.2787168  7.2734499  7.27894807]
```

## Instructions for interpreting code

We would like to replicate figure 1a of article 4 using stronger pseudo-spectral and/or DG baselines.

[Dresdner (2022)](https://arxiv.org/abs/2207.00556) replicates the result of figure 2 on GPU using a stronger pseudo-spectral (PS) baseline, and in figure 6 finds that the PS baseline achieves comparable accuracy to a FV method at 8x coarser resolution. They don't report the runtime on GPU or TPU of these methods, but they do conclude that "In contrast to prior work which showed computational speed-ups of up to 1-2 orders of magnitude over baseline finite volume (Kochkov et al., 2021) ... overall there is little potential for accelerating smooth, periodic 2D turbulence beyond traditional spectral solvers.'' The supplementary cost-accuracy plot in `data/runtime_corr.png` is consistent with figure 6 of Dresdner (2022), though the runtime is on CPU instead of GPU or TPU.

The above results show the runtime of each method at each resolution on GPU. Using the results of figure 6 of Dresdner (2022), implying that the PS baseline has comparable accuracy to the FV baseline at 8x coarser resolution, and the above results, we can replicate figure 1a.

## Instructions for running supplementary code to produce cost-accuracy plots on CPU

We recommend not running this code, as it will take multiple hours to run. However, if you would like to run this code, you do so as follows. As before, replace the lines in `makefile` that say `BASEDIR=[path here]` and `READWRITE_DIR=[path here]` with the path to the current directory. Then, type 
```
make compute_runtime
make compute_corrcoef
```
Let this code run on CPU for a few hours. Once this code runs, type `make plot_accuracy_runtime` to show the resulting cost-accuracy plot. If the runtime was significantly different than our most recent result, the grid resolution labels may not be placed correctly.

## Instructions for interpreting supplementary code

We replicate the benchmark problem in figure 2 using multiple baselines on CPU. The resulting cost-accuracy plot is shown in the file `data/runtime_corr.png`. We compare the runtime and resolution of the DG method, and find that the DG method is faster than a PS method for sufficiently low accuracy, but as the accuracy increases the PS method becomes more efficient. We also see (consistent with previous experiments we have conducted) that a DG method gives the same accuracy as the FV method at about 10-11x coarser resolution.
