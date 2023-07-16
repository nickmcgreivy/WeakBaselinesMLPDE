import jax.numpy as jnp
from timederivative import time_derivative_FV_1D_burgers


class BurgersFVSim:
    def __init__(
        self,
        core_params,
        sim_params,
        global_stabilization=False,
        epsilon_gs=0.0,
        G=lambda f, u: jnp.roll(u, -1) - u,
        model=None,
        params=None,
        delta=True,
        omega_max=0.0,
    ):
        self.global_stabilization = global_stabilization
        self.epsilon_gs = epsilon_gs
        self.model = model
        self.params = params
        self.G = G
        self.step_fn = self.get_step_fn(
            core_params, sim_params, model=model, params=params, delta=delta
        )
        self.dt_fn = self.get_dt_fn(core_params, sim_params, omega_max=omega_max)

    def get_step_fn(self, core_params, sim_params, model, params, delta):
        self.F = time_derivative_FV_1D_burgers(
            core_params,
            global_stabilization=self.global_stabilization,
            G=self.G,
            epsilon_gs=self.epsilon_gs,
            model=model,
            params=params,
            delta=delta,
        )

        def step_fn(a, t, dt, forcing_func=None):
            F = lambda a, t: self.F(a, t, forcing_func=forcing_func)
            return sim_params.rk_fn(a, t, F, dt)

        return step_fn

    def get_dt_fn(self, core_params, sim_params, omega_max=0.0):
        epsilon = 1e-4

        def get_dt(a):
            c = jnp.maximum(jnp.max(jnp.abs(a)), epsilon)
            nx = a.shape[0]
            dx = core_params.Lx / nx
            dt_advection = sim_params.cfl_safety * dx / c
            if core_params.nu > 0.0:
                dt_diffusion = sim_params.cfl_safety * dx**2 / core_params.nu
            else:
                dt_diffusion = jnp.inf
            if omega_max > 0.0:
                dt_forcing = 1 / (2 * jnp.pi * omega_max)
            else:
                dt_forcing = jnp.inf
            return jnp.minimum(jnp.minimum(dt_advection, dt_diffusion), dt_forcing)

        return get_dt
