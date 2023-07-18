## Instructions for running code

Simply type python `wang2021.py`, and wait a minute or so.

On my 2015 MacBook with 2 cores, this produces

```
nx: 25 Mean relative L2 error: 0.04358763247728348
nx: 50 Mean relative L2 error: 0.022661762312054634
nx: 100 Mean relative L2 error: 0.010235977359116077
nx: 25, runtime: 2.2108554840087915e-05
nx: 50, runtime: 0.00011061668395996108
nx: 100, runtime: 0.0004167342185974122
# PDEs solved: 1 runtime: 0.0009201288223266602
# PDEs solved: 5 runtime: 0.014868807792663571
# PDEs solved: 10 runtime: 0.023086071014404297
# PDEs solved: 20 runtime: 0.03454957008361817
# PDEs solved: 50 runtime: 0.06627278327941895
# PDEs solved: 100 runtime: 0.11201057434082033
# PDEs solved: 200 runtime: 0.220036768913269
# PDEs solved: 500 runtime: 0.5412244319915771
# PDEs solved: 1000 runtime: 0.9599107980728149
```

## Instructions for interpreting results

The minimum error in figure 11a is a little over 1%. We achieve 1% error with nx=100, which has a runtime of 4.2e-4 seconds, about an order of magnitude faster than the time to solve 1 PDE with the ML-based solver in figure 11b. The time to solve 1000 PDEs at grid resolution nx=100 on CPU is 0.95 seconds, slower than in figure 11b. However, when we run on GPU, we find this time is only 1.2e-2s, over an order of magnitude faster than the ML-based solver on GPU.