## Instructions for running code

Simply type `python fnoli2021.py`, and wait a minute or so.

On my 2015 MacBook with 2 cores, this produces



## Instructions for interpreting results

Consider table 2: the strongest baseline to solve Poisson's equation on a square grid takes 90ms, 189ms, 299ms, and 425ms for grid sizes of 33 × 33, 65 × 65, 129 × 129, and 257 × 257 respectively. We implement Poisson’s equation on a square domain using a direct solve (LU decomposition) and find that the direct solve takes 0.1ms, 0.45ms, 3ms, and 11.2ms for grid sizes of 32 × 32, 64 × 64, 128 × 128, and 256 × 256 respectively. LU decomposition is between 500 and 35 times faster than multigrid for these problem sizes. Multigrid a weak baseline relative to direct methods for sufficiently small problems.