import jax.numpy as np


def forward_euler(a_n, t_n, F, dt):
    return a_n + dt * F(a_n, t_n), t_n + dt


def ssp_rk2(a_n, t_n, F, dt):
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
    return 0.5 * a_n + 0.5 * a_1 + 0.5 * dt * F(a_1, t_n + dt), t_n + dt


def ssp_rk3(a_n, t_n, F, dt):
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
    dadt1 = F(a_n, t_n)
    a_1 = a_n + dt * dadt1
    dadt2 = F(a_1, t_n + dt)
    a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * dadt2)
    dadt3 = F(a_2, t_n + dt / 2)
    return 1 / 3 * a_n + 2 / 3 * (a_2 + dt * dadt3), t_n + dt




def ssp_rk3_adaptive(a_n, F, dt, H, f_poisson_solve):
    a_1 = a_n + dt * F(a_n, H)
    H_1 = f_poisson_solve(a_1)
    a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * F(a_1, H_1))
    H_2 = f_poisson_solve(a_2)
    return 1 / 3 * a_n + 2 / 3 * (a_2 + dt * F(a_2, H_2))




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
    "ssp_rk3_adaptive": ssp_rk3_adaptive,

}
