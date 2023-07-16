import jax.numpy as np
import numpy as onp
from functools import partial
from time import time
from jax import config, jit
import jax
config.update("jax_enable_x64", True)

from arguments import get_args
from initial_conditions import get_initial_condition_FNO
from flux import Flux
from rungekutta import FUNCTION_MAP
from helper import f_to_DG, _evalf_2D_integrate, legendre_inner_product, inner_prod_with_legendre, convert_DG_representation, vorticity_to_velocity
from poissonsolver import get_poisson_solver
from poissonbracket import get_poisson_bracket, load_alpha_right_matrix_twice, load_alpha_top_matrix_twice
from diffusion import get_diffusion_func
from simulate import simulate_2D
from trajectory import get_inner_fn
from trajectory import get_trajectory_fn as get_traj_fn
from timederivative import time_derivative_2d_navier_stokes

################
# PARAMETERS OF SIMULATION
################

Lx = 1.0
Ly = 1.0
order_exact = 2
order = order_exact
nx_exact = 16
ny_exact = nx_exact
forcing_coefficient = 0.1
runge_kutta = "ssp_rk3"
nxs_dg = [8]
t0 = 0.0

N_compute_runtime = 5
N_test = 5 # change to 5 or 10

"""
t_runtime = 50.0
cfl_safety = 10.0
cfl_safety_adaptive = 0.28 * (2 * order + 1)
cfl_safety_exact = 3.0
cfl_safety_scaled = [10.0, 10.0]
Re = 1e3
"""
t_runtime = 30.0
cfl_safety = 6.0
cfl_safety_adaptive = 0.36 * 5
cfl_safety_exact = 2.0
cfl_safety_scaled = [6.0, 6.0]
Re = 1e4

viscosity = 1/Re
t_chunk = 1.0
outer_steps = int(t_runtime)

################
# END PARAMETERS
################


################
# HELPER FUNCTIONS
################


def compute_percent_error(a1, a2):
    return np.linalg.norm(((a1[:,:,0]-a2[:,:,0]))) / np.linalg.norm((a2[:,:,0]))

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



def trajectory_fn(inner_fn, steps, start_with_input=True):
    def step(carry_in, _):
        carry_out = inner_fn(carry_in)
        frame = carry_in if start_with_input else carry_out
        return carry_out, frame

    @jax.jit
    def multistep(x_init):
        return jax.lax.scan(step, x_init, xs=None, length=steps)

    return multistep


def get_trajectory_fn(inner_fn, outer_steps, start_with_input=False):
    rollout_fn = trajectory_fn(inner_fn, outer_steps, start_with_input=start_with_input)

    @jax.jit
    def get_rollout(x_init):
        _, trajectory = rollout_fn(x_init)
        return trajectory

    return get_rollout


def get_trajectory_fn_adaptive(adaptive_step_fn, dt_fn, t_inner, outer_steps, f_poisson_solve):
    """
    adaptive_step_fn should be lambda a, dt: rk(a, F, dt)
    """
    inner_fn = get_inner_fn(adaptive_step_fn, dt_fn, t_inner, f_poisson_solve)
    traj_fn = get_traj_fn(inner_fn, outer_steps, start_with_input=False)
    return traj_fn

def get_dt_fn(args, nx, ny, order, cfl, t_runtime):

    dx = Lx / nx
    dy = Ly / ny

    R = load_alpha_right_matrix_twice(args.poisson_dir, order) / dy
    T = load_alpha_top_matrix_twice(args.poisson_dir, order) / dx

    def get_alpha(H):
        alpha_R = H @ R.T
        alpha_T = H @ T.T
        return alpha_R, alpha_T

    def get_max(H):
        alpha_R, alpha_T = get_alpha(H)
        max_R = np.amax(np.abs(alpha_R))
        max_T = np.amax(np.abs(alpha_T))
        return max_R, max_T

    def dt_fn(H):
        max_R, max_T = get_max(H)
        return cfl * (dx * dy) / (max_R * dy + max_T * dx) / (2 * order + 1)

    return dt_fn


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


def print_runtime_adaptive(args):
    a0 = get_initial_condition_FNO()
    flux = Flux.UPWIND

    for nx in nxs_dg:
        ny = nx

        a_i = convert_DG_representation(a0[None], order, 0, nx, ny, Lx, Ly, n=8)[0]

        f_poisson_bracket = get_poisson_bracket(args.poisson_dir, order, flux)
        f_poisson_solve = get_poisson_solver(args.poisson_dir, nx, ny, Lx, Ly, order)
        f_phi = lambda zeta, t: f_poisson_solve(zeta)
        f_diffusion = get_diffusion_func(order, Lx, Ly, viscosity)
        f_forcing_sim = get_forcing_FNO(order, nx, ny)
        dx = Lx / nx
        dy = Ly / ny
        leg_ip = np.asarray(legendre_inner_product(order))
        denominator = leg_ip * dx * dy
        F = lambda a, H: time_derivative_2d_navier_stokes(a, None, f_poisson_bracket, lambda zeta, t: H, denominator, f_forcing=f_forcing_sim, f_diffusion=f_diffusion)
        rk_fn = FUNCTION_MAP["ssp_rk3_adaptive"]
        adaptive_step_fn = lambda a, dt, H: rk_fn(a, F, dt, H, f_poisson_solve)

        dt_fn = get_dt_fn(args, nx, ny, order, cfl_safety_adaptive, t_runtime)

        rollout_fn_adaptive = get_trajectory_fn_adaptive(adaptive_step_fn, dt_fn, t_runtime, 1, f_poisson_solve)

        a0 = get_initial_condition_FNO()
        a_i = convert_DG_representation(a0[None], order, 0, nx, ny, Lx, Ly, n=8)[0]

        a_final = rollout_fn_adaptive(a_i)
        a_final.block_until_ready()
        times = onp.zeros(N_compute_runtime)
        for n in range(N_compute_runtime):
            a0 = get_initial_condition_FNO()
            a_i = convert_DG_representation(a0[None], order, 0, nx, ny, Lx, Ly, n=8)[0]

            t1 = time()
            a_final = rollout_fn_adaptive(a_i)
            a_final.block_until_ready()
            t2 = time()

            times[n] = t2 - t1

        print("order = {}, t_runtime = {}, nx = {}".format(order, t_runtime, nx))
        print("runtimes: {}".format(times))



def print_runtime_scaled(args):

    a0 = get_initial_condition_FNO()

    for nx in nxs_dg:
        ny = nx

        a_i = convert_DG_representation(a0[None], order, 0, nx, ny, Lx, Ly, n=8)[0]

        @jax.jit
        def rollout_fn(a0):
            step_fn_one = get_dg_step_fn(args, nx, ny, order, t_runtime/len(cfl_safety_scaled), cfl_safety=cfl_safety_scaled[0])
            step_fn_two = get_dg_step_fn(args, nx, ny, order, t_runtime/len(cfl_safety_scaled), cfl_safety=cfl_safety_scaled[1])
            rollout_fn_one = get_trajectory_fn(step_fn_one, 1)
            rollout_fn_two = get_trajectory_fn(step_fn_two, 1)
            a_one = rollout_fn_one(a0)[-1]
            assert a_one.shape == a0.shape
            a_two = rollout_fn_two(a_one)
            return a_two

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

    for _ in range(N_test):
        a0 = get_initial_condition_FNO()

        a_i = convert_DG_representation(a0[None], order_exact, 0, nx_exact, ny_exact, Lx, Ly, n=8)[0]
        exact_step_fn = get_dg_step_fn(args, nx_exact, ny_exact, order_exact, t_chunk, cfl_safety = cfl_safety_exact)
        exact_rollout_fn = get_trajectory_fn(exact_step_fn, outer_steps)
        exact_trajectory = exact_rollout_fn(a_i)
        exact_trajectory = concatenate_vorticity(a_i, exact_trajectory)

        if np.isnan(exact_trajectory).any():
            print("NaN in exact trajectory")
            raise Exception


        for n, nx in enumerate(nxs_dg):
            ny = nx

            a_i = convert_DG_representation(a0[None], order, 0, nx, ny, Lx, Ly, n=8)[0]
            step_fn = get_dg_step_fn(args, nx, ny, order, t_chunk)
            rollout_fn = get_trajectory_fn(step_fn, outer_steps)
            trajectory = rollout_fn(a_i)
            trajectory = concatenate_vorticity(a_i, trajectory)

            if np.isnan(trajectory).any():
                print("NaN in trajectory for nx={}")
                raise Exception

            for j in range(outer_steps+1):
                a_ex = convert_DG_representation(exact_trajectory[j][None], order, order_exact, nx, ny, Lx, Ly, n=8)[0]
                errors[n, j] += compute_percent_error(trajectory[j], a_ex) / N_test
    print("nxs: {}".format(nxs_dg))
    print("Mean errors: {}".format(np.mean(errors, axis=-1)))


def print_errors_adaptive(args):
    flux = Flux.UPWIND
    errors = onp.zeros((len(nxs_dg), outer_steps+1))

    for _ in range(N_test):
        a0 = get_initial_condition_FNO()
        a_i = convert_DG_representation(a0[None], order_exact, 0, nx_exact, ny_exact, Lx, Ly, n=8)[0]
        
        exact_step_fn = get_dg_step_fn(args, nx_exact, ny_exact, order_exact, t_chunk, cfl_safety = cfl_safety_exact)
        exact_rollout_fn = get_trajectory_fn(exact_step_fn, outer_steps)
        exact_trajectory = exact_rollout_fn(a_i)
        exact_trajectory = concatenate_vorticity(a_i, exact_trajectory)

        if np.isnan(exact_trajectory).any():
            print("NaN in exact trajectory")
            raise Exception


        for n, nx in enumerate(nxs_dg):
            ny = nx

            a_i = convert_DG_representation(a0[None], order, 0, nx, ny, Lx, Ly, n=8)[0]

            f_poisson_bracket = get_poisson_bracket(args.poisson_dir, order, flux)
            f_poisson_solve = get_poisson_solver(args.poisson_dir, nx, ny, Lx, Ly, order)
            f_phi = lambda zeta, t: f_poisson_solve(zeta)
            f_diffusion = get_diffusion_func(order, Lx, Ly, viscosity)
            f_forcing_sim = get_forcing_FNO(order, nx, ny)
            dx = Lx / nx
            dy = Ly / ny
            leg_ip = np.asarray(legendre_inner_product(order))
            denominator = leg_ip * dx * dy
            F = lambda a, H: time_derivative_2d_navier_stokes(a, None, f_poisson_bracket, lambda zeta, t: H, denominator, f_forcing=f_forcing_sim, f_diffusion=f_diffusion)
            rk_fn = FUNCTION_MAP["ssp_rk3_adaptive"]
            adaptive_step_fn = lambda a, dt, H: rk_fn(a, F, dt, H, f_poisson_solve)

            dt_fn = get_dt_fn(args, nx, ny, order, cfl_safety_adaptive, t_runtime)

            rollout_fn_adaptive = get_trajectory_fn_adaptive(adaptive_step_fn, dt_fn, t_chunk, outer_steps, f_poisson_solve)

            trajectory = rollout_fn_adaptive(a_i)
            trajectory = concatenate_vorticity(a_i, trajectory)

            if np.isnan(trajectory).any():
                print("NaN in trajectory for nx={}".format(nx))
                raise Exception

            for j in range(outer_steps+1):
                a_ex = convert_DG_representation(exact_trajectory[j][None], order, order_exact, nx, ny, Lx, Ly, n=8)[0]
                errors[n, j] += compute_percent_error(trajectory[j], a_ex) / N_test


    print("nxs: {}".format(nxs_dg))
    print("Mean errors: {}".format(np.mean(errors, axis=-1)))




def print_errors_scaled(args):
    errors = onp.zeros((len(nxs_dg), outer_steps+1))

    for _ in range(N_test):
        a0 = get_initial_condition_FNO()

        a_i = convert_DG_representation(a0[None], order_exact, 0, nx_exact, ny_exact, Lx, Ly, n=8)[0]
        exact_step_fn = get_dg_step_fn(args, nx_exact, ny_exact, order_exact, t_chunk, cfl_safety = cfl_safety_exact)
        exact_rollout_fn = get_trajectory_fn(exact_step_fn, outer_steps)
        exact_trajectory = exact_rollout_fn(a_i)
        exact_trajectory = concatenate_vorticity(a_i, exact_trajectory)

        if np.isnan(exact_trajectory).any():
            print("NaN in exact trajectory")
            raise Exception


        for n, nx in enumerate(nxs_dg):
            ny = nx

            a_i = convert_DG_representation(a0[None], order, 0, nx, ny, Lx, Ly, n=8)[0]
            step_fn_one = get_dg_step_fn(args, nx, ny, order, t_chunk/2, cfl_safety=cfl_safety_scaled[0])
            step_fn_two = get_dg_step_fn(args, nx, ny, order, t_chunk/2, cfl_safety=cfl_safety_scaled[1])

            @jax.jit
            def step_fn(a0):
                a_one = step_fn_one(a0)
                return step_fn_two(a_one)

            rollout_fn = get_trajectory_fn(step_fn, outer_steps)

            trajectory = rollout_fn(a_i)
            trajectory = concatenate_vorticity(a_i, trajectory)

            if np.isnan(trajectory).any():
                print("NaN in trajectory for nx={}")
                raise Exception

            for j in range(outer_steps+1):
                a_ex = convert_DG_representation(exact_trajectory[j][None], order, order_exact, nx, ny, Lx, Ly, n=8)[0]
                errors[n, j] += compute_percent_error(trajectory[j], a_ex) / N_test
    print("nxs: {}".format(nxs_dg))
    print("Mean errors: {}".format(np.mean(errors, axis=-1)))



def main():
    args = get_args()

    from jax.lib import xla_bridge
    device = xla_bridge.get_backend().platform
    print(device)
    print("nu is {}".format(viscosity))

    print_runtime_scaled(args)
    #print_runtime_adaptive(args)
    print_runtime(args)
    print_errors_scaled(args)
    #print_errors_adaptive(args)
    print_errors(args)

if __name__ == '__main__':
    main()
    
