## Instructions for running code

Simply type `test_poisson_runtime.py`, and wait a minute or so.

On my 2015 MacBook with 2 cores, this produces

```
Time to solve Poisson equation using LU decomposition on 32x32 grid is 9.846687316894531e-05
Time to solve Poisson equation using LU decomposition on 64x64 grid is 0.0004069805145263672
Time to solve Poisson equation using LU decomposition on 128x128 grid is 0.002680659294128418
Time to solve Poisson equation using LU decomposition on 256x256 grid is 0.01187598705291748
```

## Instructions for interpreting results

Consider table 2: the strongest baseline to solve Poisson's equation on a square grid takes 90ms, 189ms, 299ms, and 425ms for grid sizes of 33 × 33, 65 × 65, 129 × 129, and 257 × 257 respectively. We implement Poisson’s equation on a square domain using a direct solve (LU decomposition) and find that the direct solve takes 0.1ms, 0.45ms, 3ms, and 11.2ms for grid sizes of 32 × 32, 64 × 64, 128 × 128, and 256 × 256 respectively. LU decomposition is between 500 and 35 times faster than multigrid for these problem sizes. Multigrid a weak baseline relative to direct methods for sufficiently small problems.