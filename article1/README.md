## Instructions for running code

Simply type `python fnoli2021.py`, and wait a minute or so.

On my 2015 MacBook with 2 cores, this produces

```
cpu
nu is 0.001
order = 2, t_runtime = 50.0, nx = 7
runtimes: [0.05060792 0.05051088 0.04995608 0.05088711 0.05038285]
nx = 7, errors = 0.00804762675652948
cpu
nu is 0.0001
order = 2, t_runtime = 30.0, nx = 7
runtimes: [0.05463719 0.05493617 0.05485177 0.05511308 0.0549469 ]
nx = 7, errors = 0.09442633076966393
```

## Instructions for interpreting results

The first and sixth lines (`cpu`) are the device, either CPU or GPU. Because of the small tensor sizes and the higher efficiency of LU decomposition on CPU, this code is faster on CPU than on GPU.

We replicate two of the experiments in table 1. The `runtimes` lists the runtime of each of 5 trials using a DG method with `nx=ny=7` and second-order polynomial basis functions, while `errors` lists the percentage error (see line 2977 of `fnoli2021.py`) averaged over space, time, and the 5 trials. We use a 14x14 DG method as the ground truth solution, though increasing the resolution doesn't significantly change the result of `errors`.

The error of each experiment is comparable to the FNO-3D error in table 1 (0.0086 for nu = 0.001 and 0.0820 for nu = 0.0001). The runtime is about 0.05s, 10x slower than the 0.005s of FNO-3D on GPU. We ran this code on a (now broken) 2016 or 2017 MacBook pro and the runtime was about 0.035s, 7x slower than the 0.005s of FNO-3D on GPU.