import jax.numpy as np
import numpy as onp
from functools import partial
from time import time
from jax import config, jit
import jax_cfd.base as cfd
config.update("jax_enable_x64", True)

from arguments import get_args
from initial_conditions import get_initial_condition_FNO, f_init_CNO
from flux import Flux
from rungekutta import FUNCTION_MAP
from helper import f_to_DG, _evalf_2D_integrate, legendre_inner_product, inner_prod_with_legendre, convert_DG_representation, vorticity_to_velocity
from poissonsolver import get_poisson_solver
from poissonbracket import get_poisson_bracket
from diffusion import get_diffusion_func
from simulate import simulate_2D

from basisfunctions import legendre_poly
import matplotlib.pyplot as plt
import seaborn as sns

def plot_DG_basis(
    axs, Lx, Ly, order, zeta, title="", plotting_density=4
):
    nx, ny = zeta.shape[0:2]
    factor = order * plotting_density + 1
    num_elem = zeta.shape[-1]
    basis = legendre_poly(order)
    x = onp.linspace(-1, 1, factor + 1)[:-1] + 1 / factor
    y = onp.linspace(-1, 1, factor + 1)[:-1] + 1 / factor

    basis_x = onp.zeros((factor, factor, num_elem))
    for i in range(factor):
        for j in range(factor):
            for k in range(num_elem):
                basis_x[i, j, k] = basis[k].subs("x", x[i]).subs("y", y[j])
    Nx_plot = nx * factor
    Ny_plot = ny * factor
    output = onp.zeros((Nx_plot, Ny_plot))
    for i in range(nx):
        for j in range(ny):
            output[
                i * factor : (i + 1) * factor, j * factor : (j + 1) * factor
            ] = onp.sum(basis_x * zeta[i, j, None, None, :], axis=-1)
    
    x_plot = np.linspace(0, Lx, Nx_plot + 1)
    y_plot = np.linspace(0, Ly, Ny_plot + 1)
    pcm = axs.pcolormesh(
        x_plot,
        y_plot,
        output.T,
        shading="flat",
        cmap='jet',
        vmin=-1, 
        vmax=1
    )
    axs.set_xlim([0, Lx])
    axs.set_ylim([0, Ly])
    axs.set_xticks([0, Lx])
    axs.set_yticks([0, Ly])
    axs.set_title(title)
    return pcm



################
# PARAMETERS OF SIMULATION
################

Lx = 1.0
Ly = 1.0

order_exact = 2
order = 0

nx_exact = 60
ny_exact = 60
forcing_coefficient = 0.1
runge_kutta = "ssp_rk3"
nxs_dg = [8, 16]
t0 = 0.0

N_compute_runtime = 3 # change to 5 or 10
N_compute_error = 0 # change to 5 or 10

t_runtime = 5.0
t_chunk = 1.0
outer_steps = int(t_runtime)
cfl_safety = 30.0 # misnomer
cfl_safety_exact = 5.0
viscosity = 1e-3


################
# END PARAMETERS
################


################
# HELPER FUNCTIONS
################


def compute_percent_error_l1(a, a_exact):
    return np.linalg.norm((a[:,:,0]-a_exact[:,:,0]), ord=1) / np.linalg.norm((a_exact[:,:,0]), ord=1)

def concatenate_vorticity(v0, trajectory):
    return np.concatenate((v0[None], trajectory), axis=0)

def get_forcing_FNO(order, nx, ny):
    ff = lambda x, y, t: forcing_coefficient * (np.sin( 2 * np.pi * (x + y) ) + np.cos( 2 * np.pi * (x + y) ))
    y_term = inner_prod_with_legendre(nx, ny, Lx, Ly, order, ff, 0.0, n = 8)
    return lambda zeta: y_term

def get_inner_steps_dt_DG(nx, ny, order, cfl_safety, T):
    dx = Lx / (nx)
    dy = Ly / (ny)
    dt_i = cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)
    inner_steps = int(T // dt_i) + 1
    dt = T / inner_steps
    return inner_steps, dt

def get_dg_step_fn(args, nx, ny, order, T, cfl_safety=cfl_safety):
    if order == 0:
        flux = Flux.VANLEER
    else:
        flux = Flux.UPWIND
    
    f_poisson_bracket = get_poisson_bracket(args.poisson_dir, order, flux)
    f_poisson_solve = get_poisson_solver(args.poisson_dir, nx, ny, Lx, Ly, order)
    f_phi = lambda zeta, t: f_poisson_solve(zeta)
    f_diffusion = get_diffusion_func(order, Lx, Ly, viscosity)
    f_forcing_sim = get_forcing_FNO(order, nx, ny)
    
    inner_steps, dt = get_inner_steps_dt_DG(nx, ny, order, cfl_safety, T)

    @jit
    def simulate(a_i):
        a, _ = simulate_2D(a_i, t0, nx, ny, Lx, Ly, order, dt, inner_steps, 
                           f_poisson_bracket, f_phi, a_data=None, output=False, f_diffusion=f_diffusion,
                            f_forcing=f_forcing_sim, rk=FUNCTION_MAP[runge_kutta])
        return a
    return simulate


def get_trajectory_fn(step_fn, outer_steps):
    rollout_fn = jit(cfd.funcutils.trajectory(step_fn, outer_steps))
    
    def get_rollout(v0):
        _, trajectory = rollout_fn(v0)
        return trajectory
    
    return get_rollout


def print_runtime(args):

    a0 = get_initial_condition_FNO()

    for nx in nxs_dg:
        ny = nx

        a_i = convert_DG_representation(a0[None], order, 0, nx, ny, Lx, Ly, n=8)[0]

        step_fn = get_dg_step_fn(args, nx, ny, order, t_runtime)
        rollout_fn = get_trajectory_fn(step_fn, 1)

        a_final = rollout_fn(a_i)
        a_final.block_until_ready()
        times = onp.zeros(N_compute_runtime)
        for n in range(N_compute_runtime):
            t1 = time()
            a_final = rollout_fn(a_i)
            a_final.block_until_ready()
            t2 = time()
            times[n] = t2 - t1

        print("order = {}, t_runtime = {}, nx = {}".format(order, t_runtime, nx))
        print("runtimes: {}".format(times))


def print_errors(args):

    errors = onp.zeros((len(nxs_dg), outer_steps+1))

    for _ in range(N_compute_error):

        a0 = get_initial_condition_FNO()

        a_i_exact = convert_DG_representation(a0[None], order_exact, 0, nx_exact, ny_exact, Lx, Ly, n=8)[0]
        exact_step_fn = get_dg_step_fn(args, nx_exact, ny_exact, order_exact, t_chunk, cfl_safety = cfl_safety_exact)
        exact_rollout_fn = get_trajectory_fn(exact_step_fn, outer_steps)
        exact_trajectory = exact_rollout_fn(a_i_exact)
        exact_trajectory = concatenate_vorticity(a_i_exact, exact_trajectory)

        if np.isnan(exact_trajectory).any():
            print("NaN in exact trajectory")
            raise Exception



        for n, nx in enumerate(nxs_dg):
            ny = nx

            a_i = convert_DG_representation(a_i_exact[None], order, order_exact, nx, ny, Lx, Ly, n=8)[0]
            step_fn = get_dg_step_fn(args, nx, ny, order, t_chunk)
            rollout_fn = get_trajectory_fn(step_fn, outer_steps)
            trajectory = rollout_fn(a_i)
            trajectory = concatenate_vorticity(a_i, trajectory)

            if np.isnan(trajectory).any():
                print("NaN in trajectory for nx={}")
                raise Exception


            ##### only evaluating error at T=5, j=-1

            for j in range(outer_steps+1):
                a_ex = convert_DG_representation(exact_trajectory[j][None], order, order_exact, nx, ny, Lx, Ly, n=8)[0]
                errors[n, j] += compute_percent_error_l1(trajectory[j], a_ex) / N_compute_error

            
            fig, axs = plt.subplots(2,2,figsize=(6, 6))

            plot_DG_basis(axs[0,0], Lx, Ly, order, exact_trajectory[0], title="Exact Trajectory t=0")
            plot_DG_basis(axs[0,1], Lx, Ly, order, exact_trajectory[-1], title="Exact Trajectory t={}".format(int(t_runtime)))
            plot_DG_basis(axs[1,0], Lx, Ly, order, trajectory[0], title="Low-resolution ({}x{}), t=0".format(nx, ny))
            pcm = plot_DG_basis(axs[1,1], Lx, Ly, order, trajectory[-1], title="Low-resolution ({}x{}), t={}".format(nx, ny, int(t_runtime)))
            fig.colorbar(pcm, ax=axs, extend="max")
            #fig.tight_layout()
            plt.show()
            

    print("nxs: {}".format(nxs_dg))
    print("Mean errors: {}".format(np.mean(errors, axis=-1)))
    for j in range(outer_steps+1):
        print("Percent L1 errors at t={} are {}".format(j, errors[:, j]))

def main():
    args = get_args()

    from jax.lib import xla_bridge
    device = xla_bridge.get_backend().platform
    print(device)

    print_runtime(args)
    #print_errors(args)

if __name__ == '__main__':
    main()
    
