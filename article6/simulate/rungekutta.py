def forward_euler(a_n, t_n, F, dt):
    return a_n + dt * F(a_n, t_n), t_n + dt


def ssp_rk2(a_n, t_n, F, dt):
    a_1 = a_n + dt * F(a_n, t_n)
    t_1 = t_n + dt
    return 0.5 * a_n + 0.5 * a_1 + 0.5 * dt * F(a_1, t_1), t_n + dt


def ssp_rk3(a_n, t_n, F, dt):
    a_1 = a_n + dt * F(a_n, t_n)
    t_1 = t_n + dt
    a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * F(a_1, t_1))
    t_2 = t_n + dt / 2
    return 1 / 3 * a_n + 2 / 3 * (a_2 + dt * F(a_2, t_2)), t_n + dt


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
