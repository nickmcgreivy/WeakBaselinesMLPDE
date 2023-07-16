import jax.numpy as np
from jax.lax import scan
from functools import partial
from jax import vmap, jit
from dataclasses import dataclass  # Needs Python 3.7 or higher
from jax.config import config

config.update("jax_enable_x64", True)

from fluxdg import Flux
from timederivativedg import time_derivative_DG_1D_burgers
import sys
sys.path.append("..")
from rungekutta import ssp_rk3


def _scan(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), None


def _scan_output(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), a_f



def simulate_1D(
    a0,
    t0,
    p,
    flux,
    nx,
    dx,
    dt,
    leg_poly,
    nt,
    output=False,
    forcing_func=None,
    nu=0.0,
    rk=ssp_rk3
):

    dadt = lambda a, t: time_derivative_DG_1D_burgers(
        a,
        t,
        p,
        flux,
        nx,
        dx,
        leg_poly,
        forcing_func=forcing_func,
        nu=nu,
    )

    rk_F = lambda a, t: rk(a, t, dadt, dt)


    if output:
        scanf = jit(lambda sol, x: _scan_output(sol, x, rk_F))
        _, data = scan(scanf, (a0, t0), None, length=nt)
        return data
    else:
        scanf = jit(lambda sol, x: _scan(sol, x, rk_F))
        (a_f, t_f), _ = scan(scanf, (a0, t0), None, length=nt)
        return (a_f, t_f)
