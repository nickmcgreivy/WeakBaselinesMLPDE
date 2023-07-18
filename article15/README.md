## Instructions for running code

For the first replication, type `python burgers_replicate.py`, and wait a minute or so. For the second replication, do the same but type `python wave_equation_dirichlet.py`.

On my 2015 MacBook with 2 cores, this produces

```
E1 WENO5: nx = 40, Runtime = 0.00035
E1 WENO5: nx = 50, Runtime = 0.00044
E1 WENO5: nx = 100, Runtime = 0.00162
E1 WENO5: nx = 40, Accumulated MSE = 0.450
E1 WENO5: nx = 50, Accumulated MSE = 0.516
E1 WENO5: nx = 100, Accumulated MSE = 0.172
We are now beginning experiment 2.
E2 WENO5: nx = 40, Runtime = 0.00036
E2 WENO5: nx = 50, Runtime = 0.00054
E2 WENO5: nx = 100, Runtime = 0.00183
E2 WENO5: nx = 40, Accumulated MSE = 0.270
E2 WENO5: nx = 50, Accumulated MSE = 0.261
E2 WENO5: nx = 100, Accumulated MSE = 0.081
```

and a figure identical to `WE1_reproduction.png`.

## Instructions for interpreting results

Comparing the runtime and accumulated MSE for E1, E2, and W1 with tables 1 and 2, we see that this stronger baseline is orders of magnitude faster than the ML-based solver at constant accumulated MSE.