import sys
sys.path.append("../core")
sys.path.append("../simulate")
sys.path.append("../core/dg")

import jax
import jax.numpy as jnp
import numpy as onp
from jax import config, vmap, jit

config.update("jax_enable_x64", True)
import xarray
import seaborn as sns
import matplotlib.pyplot as plt


from flux import Flux
from fluxdg import Flux as FluxDG
from initialconditions import (
    get_a0,
    get_initial_condition_fn,
    get_a,
    forcing_func_sum_of_modes,
)
from simparams import CoreParams, SimulationParams
from legendre import generate_legendre
from simulations import BurgersFVSim
from trajectory import get_trajectory_fn, get_inner_fn
from helper import convert_FV_representation
from helperdg import convert_DG_representation
from simulator import simulate_1D


def plot_fv(a, core_params, color="blue"):
    plot_dg(a[..., None], core_params, color=color)


def plot_fv_trajectory(trajectory, core_params, t_inner, color="blue"):
    plot_dg_trajectory(trajectory[..., None], core_params, t_inner, color=color)


def plot_dg(a, core_params, color="blue"):
    if len(a.shape) == 1:
        p = 1
    else:
        p = a.shape[-1]

    def evalf(x, a, j, dx, leg_poly):
        x_j = dx * (0.5 + j)
        xi = (x - x_j) / (0.5 * dx)
        vmap_polyval = vmap(jnp.polyval, (0, None), -1)
        poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
        return jnp.sum(poly_eval * a, axis=-1)

    NPLOT = [2, 2, 5, 7][p - 1]
    nx = a.shape[0]
    dx = core_params.Lx / nx
    xjs = jnp.arange(nx) * core_params.Lx / nx
    xs = xjs[None, :] + jnp.linspace(0.0, dx, NPLOT)[:, None]
    vmap_eval = vmap(evalf, (1, 0, 0, None, None), 1)

    a_plot = vmap_eval(xs, a, jnp.arange(nx), dx, generate_legendre(p))
    a_plot = a_plot.T.reshape(-1)
    xs = xs.T.reshape(-1)
    coords = {("x"): xs}
    data = xarray.DataArray(a_plot, coords=coords)
    data.plot(color=color)


def plot_dg_trajectory(trajectory, core_params, t_inner, color="blue"):
    p = 1
    NPLOT = [2, 2, 5, 7][p - 1]
    nx = trajectory.shape[1]
    dx = core_params.Lx / nx
    xjs = jnp.arange(nx) * core_params.Lx / nx
    xs = xjs[None, :] + jnp.linspace(0.0, dx, NPLOT)[:, None]

    def get_plot_repr(a):
        def evalf(x, a, j, dx, leg_poly):
            x_j = dx * (0.5 + j)
            xi = (x - x_j) / (0.5 * dx)
            vmap_polyval = vmap(jnp.polyval, (0, None), -1)
            poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
            return jnp.sum(poly_eval * a, axis=-1)

        vmap_eval = vmap(evalf, (1, 0, 0, None, None), 1)
        return vmap_eval(xs, a, jnp.arange(nx), dx, generate_legendre(p)).T

    get_trajectory_plot_repr = vmap(get_plot_repr)
    trajectory_plot = get_trajectory_plot_repr(trajectory)

    outer_steps = trajectory.shape[0]

    trajectory_plot = trajectory_plot.reshape(outer_steps, -1)
    xs = xs.T.reshape(-1)
    coords = {"x": xs, "time": t_inner * jnp.arange(outer_steps)}
    xarray.DataArray(trajectory_plot, dims=["time", "x"], coords=coords).plot(
        col="time", col_wrap=5, color=color
    )


def plot_multiple_fv_trajectories(trajectories, core_params, t_inner):
    plot_multiple_dg_trajectories(
        [trajectory[..., None] for trajectory in trajectories], core_params, t_inner
    )


def plot_multiple_dg_trajectories(trajectories, core_params, t_inner):
    outer_steps = trajectories[0].shape[0]
    nx = trajectories[0].shape[1]
    p = 1
    NPLOT = [2, 2, 5, 7][p - 1]
    dx = core_params.Lx / nx
    xjs = jnp.arange(nx) * core_params.Lx / nx
    xs = xjs[None, :] + jnp.linspace(0.0, dx, NPLOT)[:, None]

    def get_plot_repr(a):
        def evalf(x, a, j, dx, leg_poly):
            x_j = dx * (0.5 + j)
            xi = (x - x_j) / (0.5 * dx)
            vmap_polyval = vmap(jnp.polyval, (0, None), -1)
            poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
            return jnp.sum(poly_eval * a, axis=-1)

        vmap_eval = vmap(evalf, (1, 0, 0, None, None), 1)
        return vmap_eval(xs, a, jnp.arange(nx), dx, generate_legendre(p)).T

    get_trajectory_plot_repr = vmap(get_plot_repr)
    trajectory_plots = []
    for trajectory in trajectories:
        trajectory_plots.append(
            get_trajectory_plot_repr(trajectory).reshape(outer_steps, -1)
        )

    xs = xs.T.reshape(-1)
    coords = {"x": xs, "time": t_inner * jnp.arange(outer_steps)}
    xarray.DataArray(
        trajectory_plots, dims=["stack", "time", "x"], coords=coords
    ).plot.line(col="time", hue="stack", col_wrap=5)


def get_core_params(Lx=1.0, flux="godunov", nu=0.0):
    return CoreParams(Lx, flux, nu)


cfl_safety = 0.25

def get_sim_params(name="test", cfl_safety=cfl_safety, rk="ssp_rk3"):
    return SimulationParams(name, cfl_safety, rk)


def l2_norm_trajectory(trajectory):
    return jnp.mean(trajectory**2, axis=1)


# In[ ]:


#################
# HYPERPARAMETERS
#################

init_description = "zeros"
simname = "reproduce"

t_warmup = 5.0

nx_exact = 512
nxs = [16, 32, 64, 128, 256]
nxs_dg = [16, 32, 64, 128, 256]
nxs_rev = [256, 128, 64, 32, 16]
nxs_dg_rev = [256, 128, 64, 32, 16]

omega_max = 0.4
nu = 0.01

key = jax.random.PRNGKey(13)

delta = False

#################
# END HYPERPARAMS
#################


kwargs_init = {
    "min_num_modes": 2,
    "max_num_modes": 6,
    "min_k": 0,
    "max_k": 3,
    "amplitude_max": 1.0,
}
kwargs_forcing = {
    "min_num_modes": 20,
    "max_num_modes": 20,
    "min_k": 3,
    "max_k": 6,
    "amplitude_max": 0.5,
    "omega_max": omega_max,
}
kwargs_sim = {"name": simname, "cfl_safety": cfl_safety, "rk": "ssp_rk3"}

kwargs_core_weno = {"Lx": 2 * jnp.pi, "flux": "weno", "nu": nu}
kwargs_core_god = {"Lx": 2 * jnp.pi, "flux": "godunov", "nu": nu}
kwargs_core_god_bad = {"Lx": 2 * jnp.pi, "flux": "godunovbad", "nu": nu}
kwargs_core_weno_bad = {"Lx": 2 * jnp.pi, "flux": "wenobad", "nu": nu}



sim_params = get_sim_params(**kwargs_sim)

core_params_weno = get_core_params(**kwargs_core_weno)
core_params_god = get_core_params(**kwargs_core_god)
core_params_god_bad = get_core_params(**kwargs_core_god_bad)
core_params_weno_bad = get_core_params(**kwargs_core_weno_bad)

sim_weno = BurgersFVSim(core_params_weno, sim_params, delta=delta, omega_max=omega_max)
sim_god = BurgersFVSim(core_params_god, sim_params, delta=delta, omega_max=omega_max)
sim_god_bad = BurgersFVSim(
    core_params_god_bad, sim_params, delta=delta, omega_max=omega_max
)
sim_weno_bad = BurgersFVSim(
    core_params_weno_bad, sim_params, delta=delta, omega_max=omega_max
)

init_fn = lambda key: get_initial_condition_fn(
    core_params_weno, init_description, key=key, **kwargs_init
)
forcing_fn = forcing_func_sum_of_modes(core_params_weno.Lx, **kwargs_forcing)



key = jax.random.PRNGKey(43)



N = 5

mae_weno = onp.zeros(len(nxs))
mae_god = onp.zeros(len(nxs))
mae_weno_bad = onp.zeros(len(nxs))
mae_god_bad = onp.zeros(len(nxs))
mae_zeros = onp.zeros(len(nxs))
mae_dg2 = onp.zeros(len(nxs))
mae_dg3 = onp.zeros(len(nxs))


def mae_loss(v, v_ex):
    diff = v - v_ex
    return jnp.mean(jnp.absolute(diff))


t_inner = 0.1
outer_steps = 150
outer_steps_warmup = int(t_warmup / t_inner)
key = jax.random.PRNGKey(16)


vmap_convert = vmap(convert_FV_representation, (0, None, None), 0)


for n in range(N):
    print(n)

    key, key1, key2 = jax.random.split(key, 3)

    f_init = get_initial_condition_fn(
        core_params_weno, init_description, key=key1, **kwargs_init
    )
    f_forcing = forcing_fn(key2)

    step_fn = lambda a, t, dt: sim_weno.step_fn(a, t, dt, forcing_func=f_forcing)
    inner_fn = get_inner_fn(step_fn, sim_weno.dt_fn, t_inner)

    trajectory_fn_warmup = get_trajectory_fn(
        inner_fn, outer_steps_warmup, start_with_input=False
    )
    trajectory_fn_weno = get_trajectory_fn(inner_fn, outer_steps)

    t0_init = 0.0
    a0_init = get_a0(f_init, core_params_weno, nx_exact)
    x0_init = (a0_init, t0_init)

    # warmup
    trajectory_warmup, trajectory_t = trajectory_fn_warmup(x0_init)
    a0_exact = trajectory_warmup[-1]
    t0 = trajectory_t[-1]
    x0_exact = (a0_exact, t0)

    # exact trajectory
    trajectory_exact, trajectory_exact_t = trajectory_fn_weno(x0_exact)


    for i, nx in enumerate(nxs):
        print(nx)
        a0 = convert_FV_representation(a0_exact, nx, core_params_weno.Lx)
        x0 = (a0, t0)

        # exact trajectory downsampled
        trajectory_exact_ds = vmap_convert(trajectory_exact, nx, core_params_weno.Lx)

        # WENO
        trajectory_weno, _ = trajectory_fn_weno(x0)

        # Godunov
        step_fn = lambda a, t, dt: sim_god.step_fn(a, t, dt, forcing_func=f_forcing)
        inner_fn = get_inner_fn(step_fn, sim_god.dt_fn, t_inner)
        trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
        trajectory_god, _ = trajectory_fn(x0)

        
        # Godunov Bad
        step_fn = lambda a, t, dt: sim_god_bad.step_fn(a, t, dt, forcing_func=f_forcing)
        inner_fn = get_inner_fn(step_fn, sim_god_bad.dt_fn, t_inner)
        trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
        trajectory_god_bad, _ = trajectory_fn(x0)

        # Weno Bad
        step_fn = lambda a, t, dt: sim_weno_bad.step_fn(
            a, t, dt, forcing_func=f_forcing
        )
        inner_fn = get_inner_fn(step_fn, sim_weno_bad.dt_fn, t_inner)
        trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
        trajectory_weno_bad, _ = trajectory_fn(x0)

        mae_weno[i] += mae_loss(trajectory_weno, trajectory_exact_ds) / N
        mae_god[i] += mae_loss(trajectory_god, trajectory_exact_ds) / N
        mae_god_bad[i] += mae_loss(trajectory_god_bad, trajectory_exact_ds) / N
        mae_weno_bad[i] += mae_loss(trajectory_weno_bad, trajectory_exact_ds) / N
        mae_zeros[i] += (
            mae_loss(jnp.zeros(trajectory_exact_ds.shape), trajectory_exact_ds) / N
        )
        

    for i, nx in enumerate(nxs_dg):
        Lx = core_params_weno.Lx

        # p = 3, second-order polynomials
        p_dg = 3
        leg_poly = generate_legendre(p_dg)
        
        print("dg: nx={}".format(nx))

        a0_dg = convert_DG_representation(a0_exact[:,None], p_dg, nx, (Lx / nx), (Lx / nx_exact), generate_legendre(1))
        trajectory_exact_dg = vmap(lambda a0: convert_DG_representation(a0[:,None], p_dg, nx, (Lx / nx), (Lx / nx_exact), generate_legendre(1)))(trajectory_exact)

        trajectory_dg = a0_dg[None]
        a = a0_dg
        t = t0
        dx = Lx / nx

        if nx == 256:
            cfl = 0.1
        else:
            cfl = cfl_safety
        dt_i = cfl * dx / (2 * p_dg - 1)
        nt = int(t_inner / dt_i) + 1
        dt = t_inner / nt

        sim_f = jit(lambda a, t: simulate_1D(
                a,
                t,
                p_dg,
                FluxDG.GODUNOV,
                nx,
                dx,
                dt,
                leg_poly,
                nt,
                forcing_func=f_forcing,
                nu=nu,
                output=False,
            ))
        for _ in range(outer_steps-1):
            a, t = sim_f(a,t)
            trajectory_dg = jnp.concatenate((trajectory_dg, a[None]),axis=0)

        mae_dg2[i] += mae_loss(trajectory_dg[...,0], trajectory_exact_dg[...,0]) / N


        # p = 4, third-order polynomials
        p_dg = 4
        leg_poly = generate_legendre(p_dg)
        
        print("dg: nx={}".format(nx))

        a0_dg = convert_DG_representation(a0_exact[:,None], p_dg, nx, (Lx / nx), (Lx / nx_exact), generate_legendre(1))
        trajectory_exact_dg = vmap(lambda a0: convert_DG_representation(a0[:,None], p_dg, nx, (Lx / nx), (Lx / nx_exact), generate_legendre(1)))(trajectory_exact)

        trajectory_dg = a0_dg[None]
        a = a0_dg
        t = t0
        dx = Lx / nx

        if nx == 256:
            cfl = 0.05
        elif nx == 128:
            cfl = 0.1
        else:
            cfl = cfl_safety
        dt_i = cfl * dx / (2 * p_dg - 1)
        nt = int(t_inner / dt_i) + 1
        dt = t_inner / nt

        sim_f = jit(lambda a, t: simulate_1D(
                a,
                t,
                p_dg,
                FluxDG.GODUNOV,
                nx,
                dx,
                dt,
                leg_poly,
                nt,
                forcing_func=f_forcing,
                nu=nu,
                output=False,
            ))
        for _ in range(outer_steps-1):
            a, t = sim_f(a,t)
            trajectory_dg = jnp.concatenate((trajectory_dg, a[None]),axis=0)

        mae_dg3[i] += mae_loss(trajectory_dg[...,0], trajectory_exact_dg[...,0]) / N





"""
maes = jnp.asarray(
    [
        mae_weno,
        mae_god,
        mae_god_bad,
        mae_weno_bad,
        mae_zeros,
    ]
)

with open("maes.npy", "wb") as f:
    onp.save(f, maes)


# In[ ]:


with open("maes.npy", "rb") as f:
    maes = onp.load(f, allow_pickle=True)
print(maes)

(
    mae_weno,
    mae_god,
    mae_god_bad,
    mae_weno_bad,
    mae_zeros,
) = maes
"""

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

print(mae_weno)
print(mae_god)
print(mae_god_bad)
print(mae_weno_bad)
print(mae_zeros)
print(mae_dg2)
print(mae_dg3)

fig, axs = plt.subplots(1, 1, figsize=(7, 3.25))
axs.spines["top"].set_visible(False)
axs.spines["right"].set_visible(False)
linewidth = 2

maes = [mae_god, mae_weno, mae_dg2, mae_dg3]
labels = ["1st Order", "WENO", "DG polyOrder2", "DG polyOrder3"]
colors = ["#1f77b4", "purple", "black", "red"]
linestyles = ["solid", "solid", "solid", "solid"]
markers = ["^", "*", "s", "s"]
markersize = ["12", "12", "8", "8"]

for k, mae in enumerate(maes):
    plt.loglog(
        nxs_rev,
        jnp.nan_to_num(jnp.asarray(mae), nan=1e2),
        color=colors[k],
        linewidth=linewidth,
        linestyle=linestyles[k],
        markersize=markersize[k],
        marker=markers[k],
    )



axs.set_xticks([16, 32, 64, 128, 256])
axs.set_xticklabels(["2", "4", "8", "16", "32"], fontsize=18)
axs.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
axs.set_yticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"], fontsize=18)
axs.set_ylabel("Mean absolute error", fontsize=18)
axs.set_xlabel("Resample Factor", fontsize=18)
axs.tick_params(axis="x", which="minor", bottom=False)

handles = []
for k, mae in enumerate(maes):
    handles.append(
        mlines.Line2D(
            [],
            [],
            color=colors[k],
            linewidth=linewidth,
            linestyle=linestyles[k],
            label=labels[k],
            marker=markers[k],
            markersize=markersize[k],
        )
    )


plt.ylim([1e-4 - 1e-5, 3e-1])
axs.legend(handles=handles, loc=(0.98, 0.1), prop={"size": 16}, frameon=False)

fig.tight_layout()


plt.show()
