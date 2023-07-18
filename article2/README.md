## Instructions for running code

Simply type python `deeponetslulu2021.py`, and wait a minute or so.

On my 2015 MacBook with 2 cores, this produces

```
nt is 130
Average runtime is 6.243228912353516e-05
```

and two figures.

## Instructions for interpreting results

Consider the fluids-relevant PDE in table S5, the advection equation. There are 4 cases of 1D advection and one case of advection-diffusion. Case I is the simplest to replicate, so we focus on case I.

The left figure shows the initial condition, which is similar to the initial condition in figure S11. We run until t=1 using 130 timesteps, which (averaged over 100 trials) takes 6e-5 seconds. The right figure shows the absolute error in space and time, which is comparable to or lower than the error in figure S11. 

In other words, we achieved similar error with almost an order of magnitude lower runtime than the 3.7e-4 seconds listed in table S5.