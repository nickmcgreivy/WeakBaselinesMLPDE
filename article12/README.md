## Instructions for running code

Simply type python `pinoli2022.py`, and wait a minute or two.

On my 2015 MacBook with 2 cores, this produces
```
cpu
nu is 0.05
order = 2, t_runtime = 50.0, nx = 3
runtimes: [0.034024   0.03412199 0.03416014 0.033674   0.03401804]
nx = 3, errors = 0.025966695723438644
```

## Instructions for interpreting results

The error in this case is 2.5%, about the same as the 2.87% error of PINO. The runtime is, on average, 0.034s which is about 7x slower than the PINO method which runs in 0.005s.  