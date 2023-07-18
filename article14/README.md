## Instructions for running code

Simply type python `python test_poisson_runtime.py`, and wait a minute or so.

On my 2015 MacBook with 2 cores, this produces

```
Time to solve Poisson equation using LU decomposition on 32x32 grid is 9.393692016601562e-05
Time to solve Poisson equation using LU decomposition on 64x64 grid is 0.0005288124084472656
Time to solve Poisson equation using LU decomposition on 128x128 grid is 0.0026439428329467773
Time to solve Poisson equation using LU decomposition on 256x256 grid is 0.011600494384765625
```

## Instructions for interpreting results

Although we are solving on a square domain with periodic boundary conditions, changing to non-periodic boundary conditions will have very little influence on the time to solve Poisson's equation. The square domain with non-periodic boundary conditions (figure 2a) takes almost 20 seconds to solve a 256x256 problem, while doing so with an LU decomposition takes 11.6 millseconds with error of zero (modulo floating point errors). LU decomposition is over 3 orders of magnitude faster than FEniCS is in this case.