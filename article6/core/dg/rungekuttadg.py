import jax.numpy as np
from limiter import min_mod_limiter


def forward_euler(a_n, t_n, F, dt, limit=False):
    a_1 = a_n + dt * F(a_n, t_n)
    if limit:
        a_1 = min_mod_limiter(a_1)
    return a_1, t_n + dt


def ssp_rk2(a_n, t_n, F, dt, limit=False):
    """
    Takes a set of coefficients a_n, and outputs
    a set of coefficients a_{n+1} using a strong-stability
    preserving RK2 method.

    Uses the equations
    a_1 = a_n + dt * F(a_n, t_n)
    a_{n+1} = 1/2 a_n + 1/2 a_1 + 1/2 * dt * F(a_1, t_n + dt)

    Inputs
    a_n: value of vector at beginning of timestep
    t_n: time at beginning of timestep
    F: da/dt = F(a, t), vector function
    dt: timestep

    Outputs
    a_{n+1}: value of vector at end of timestep
    t_{n+1}: time at end of timestep

    """
    a_1 = a_n + dt * F(a_n, t_n)
    if limit:
        a_1 = min_mod_limiter(a_1)
    a_2 = 0.5 * a_n + 0.5 * a_1 + 0.5 * dt * F(a_1, t_n + dt)
    if limit:
        a_2 = min_mod_limiter(a_2)
    return a_2, t_n + dt


def ssp_rk3(a_n, t_n, F, dt, limit=False):
    """
    Takes a set of coefficients a_n, and outputs
    a set of coefficients a_{n+1} using a strong-stability
    preserving RK3 method.

    Uses the equations
    a_1 = a_n + dt * F(a_n, t_n)
    a_2 = 3/4 a_n + 1/4 * a_1 + 1/4 * dt * F(a_1, t_n + dt)
    a_{n+1} = 1/3 a_n + 2/3 a_2 + 2/3 * dt * F(a_2, t_n + dt/2)

    Inputs
    a_n: value of vector at beginning of timestep
    t_n: time at beginning of timestep
    F: da/dt = F(a, t), vector function
    dt: timestep

    Outputs
    a_{n+1}: value of vector at end of timestep
    t_{n+1}: time at end of timestep

    """
    a_1 = a_n + dt * F(a_n, t_n)
    if limit:
        a_1 = min_mod_limiter(a_1)
    a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * F(a_1, t_n + dt))
    if limit:
        a_2 = min_mod_limiter(a_2)
    a_3 = 1 / 3 * a_n + 2 / 3 * (a_2 + dt * F(a_2, dt + dt / 2))
    if limit:
        a_3 = min_mod_limiter(a_3)
    return a_3, t_n + dt


FUNCTION_MAP = {
    "FE": forward_euler,
    "fe": forward_euler,
    "forward_euler": forward_euler,
    "rk2": ssp_rk2,
    "RK2": ssp_rk2,
    "ssp_rk2": ssp_rk2,
    "rk3": ssp_rk3,
    "RK3": ssp_rk3,
    "ssp_rk3": ssp_rk3,
}
