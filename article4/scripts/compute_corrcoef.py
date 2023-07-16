import jax.numpy as np
import jax
from jax import jit, config
from functools import partial
import h5py
import jax_cfd.base as cfd
from jax_cfd.base import boundaries
from jax_cfd.base import forcings
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
config.update("jax_enable_x64", True)

from arguments import get_args
from flux import Flux
from helper import f_to_DG, _evalf_2D_integrate, legendre_inner_product, inner_prod_with_legendre, convert_DG_representation, vorticity_to_velocity
from poissonsolver import get_poisson_solver
from poissonbracket import get_poisson_bracket
from diffusion import get_diffusion_func
from simulate import simulate_2D
from rungekutta import FUNCTION_MAP
from initial_conditions import f_init_MLCFD

################
# PARAMETERS OF SIMULATION
################

PI = np.pi
Lx = 2 * PI
Ly = 2 * PI
t0 = 0.0
runge_kutta = "ssp_rk3"

density = 1.
max_velocity = 7.0
ic_wavenumber = 2
Re = 1000
viscosity = 1/Re
forcing_coefficient = 1.0
damping_coefficient = 0.1
exact_flux = Flux.UPWIND


nx_max = ny_max = 512
nx_exact = 192
ny_exact = 192
order_exact = 2
orders = [0, 1, 2]
nxs_dg = [[16, 32, 64, 128, 256], [8, 16, 32, 48, 64, 96, 128, 192], [8, 16, 24, 32, 48, 64, 96, 128]]
nxs_fv_baseline = [16, 32, 64, 128, 256, 512]
nxs_ps_baseline = [8, 16, 32, 64, 128, 256]
nx_burn_in = ny_burn_in = nx_max


"""
nx_max = ny_max = 128
nx_exact = 24
ny_exact = 24
order_exact = 2
orders = [0, 1, 2]
nxs_dg = [[],[],[]]
nxs_fv_baseline = [16, 32, 64]
nxs_ps_baseline = [32]
nx_burn_in = ny_burn_in = nx_max
"""

cfl_safety_exact = 0.3
cfl_safety_dg = 0.3
cfl_safety_cfd = 0.5

t_final = 10.0
outer_steps = int(t_final * 10)
t_chunk = t_final / outer_steps
N_test = 1 # change to 5 or 10

t_burn_in = 10.0


################
# END PARAMETERS
################


################
# WRITE TO FILE
################

def create_corr_file(n, args):
    # 
    f = h5py.File(
        "{}/data/corr_run{}_fv.hdf5".format(args.read_write_dir, n),
        "w",
    )
    for nx in nxs_fv_baseline:
        dset_new = f.create_dataset(str(nx), (outer_steps+1,), dtype="float64")
    f.close()

    f = h5py.File(
        "{}/data/corr_run{}_ps.hdf5".format(args.read_write_dir, n),
        "w",
    )
    for nx in nxs_ps_baseline:
        dset_new = f.create_dataset(str(nx), (outer_steps+1,), dtype="float64")
    f.close()

    for o, order in enumerate(orders):
        f = h5py.File(
            "{}/data/corr_run{}_order{}.hdf5".format(args.read_write_dir, n, order),
            "w",
        )
        for nx in nxs_dg[o]:
            dset_new = f.create_dataset(str(nx), (outer_steps+1,), dtype="float64")
        f.close()


def write_corr_file(n, args, name, nx, j, value):
    f = h5py.File(
        "{}/data/corr_run{}_{}.hdf5".format(args.read_write_dir, n, name),
        "r+",
    )
    f[str(nx)][j] = value
    f.close()


#############
# HELPER FUNCTIONS
#############


def kolmogorov_forcing_cfd(grid, scale, k):
    offsets = grid.cell_faces

    y = grid.mesh(offsets[0])[1]
    u = scale * grids.GridArray(np.sin(k * y), offsets[0], grid)

    if grid.ndim == 2:
        v = grids.GridArray(np.zeros_like(u.data), offsets[1], grid)
        f = (u, v)
    else:
        raise NotImplementedError

    def forcing(v):
        del v
        return f
    return forcing


def get_forcing_cfd(nx, ny):
    grid = get_grid(nx, ny)
    f_constant_term = kolmogorov_forcing_cfd(grid, forcing_coefficient, 4)
    def f_forcing(v):
        return tuple(c_i - damping_coefficient * v_i.array for c_i, v_i in zip(f_constant_term(v), v))
    return f_forcing


def get_forcing_ps(nx, ny):
    #offsets = ((0, 0), (0, 0))
    offsets = None
    k=4
    forcing_fn = lambda grid: forcings.kolmogorov_forcing(grid, scale = forcing_coefficient, k=k, offsets=offsets)
    return forcing_fn


def get_velocity_cfd(u_x, u_y):
    assert u_x.shape == u_y.shape
    bcs = boundaries.periodic_boundary_conditions(2)
  
    grid = get_grid(u_x.shape[0], u_x.shape[1])
    u_x = grids.GridVariable(grids.GridArray(u_x, grid.cell_faces[0], grid=grid), bcs)
    u_y = grids.GridVariable(grids.GridArray(u_y, grid.cell_faces[1], grid=grid), bcs)
    return (u_x, u_y)


def get_trajectory_fn(step_fn, outer_steps):
    rollout_fn = jax.jit(cfd.funcutils.trajectory(step_fn, outer_steps))
    
    def get_rollout(v0):
        _, trajectory = rollout_fn(v0)
        return trajectory
    
    return get_rollout


def concatenate_vorticity(v0, trajectory):
    return np.concatenate((v0[None], trajectory), axis=0)


def concatenate_velocity(u0, trajectory):
    return (np.concatenate((u0[0].data[None], trajectory[0].data),axis=0), np.concatenate((u0[1].data[None], trajectory[1].data),axis=0))


def downsample_ux(u_x, F):
  nx, ny = u_x.shape
  assert nx % F == 0
  return np.mean(u_x[F-1::F,:].reshape(nx // F, ny // F, F), axis=2)
    

def downsample_uy(u_y, F):
  nx, ny = u_y.shape
  assert ny % F == 0
  return np.mean(u_y[:, F-1::F].reshape(nx // F, F, ny // F), axis=1)


def get_dt_cfd(nx, ny):
    grid = get_grid(nx, ny)
    return cfd.equations.stable_time_step(max_velocity, cfl_safety_cfd, viscosity, grid)


def get_grid(nx, ny):
    return grids.Grid((nx, ny), domain=((0, Lx), (0, Ly)))


def get_forcing_dg(order, nx, ny):
    leg_ip = np.asarray(legendre_inner_product(order))
    ff = lambda x, y, t: 4 * (2 * PI / Ly) * np.cos(4 * (2 * PI / Ly) * y)
    y_term = inner_prod_with_legendre(nx, ny, Lx, Ly, order, ff, 0.0, n = 8)
    dx = Lx / nx
    dy = Ly / ny
    return lambda zeta: (forcing_coefficient * y_term - dx * dy * damping_coefficient * zeta * leg_ip)


def get_inner_steps_dt_cfd(nx, ny, cfl_safety, T):
    inner_steps = int(T // get_dt_cfd(nx, ny)) + 1
    dt = T / inner_steps
    return inner_steps, dt


def get_inner_steps_dt_DG(nx, ny, order, cfl_safety, T):
    dx = Lx / (nx)
    dy = Ly / (ny)
    dt_i = cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)
    inner_steps = int(T // dt_i) + 1
    dt = T / inner_steps
    return inner_steps, dt


def shift_down_left(a):
    return (a + np.roll(a, 1, axis=1) + np.roll(a, 1, axis=0) + np.roll(np.roll(a, 1, axis=0), 1, axis=1)) / 4


def vorticity(u):
    return cfd.finite_differences.curl_2d(u).data


def downsample_u(u0, nx_new):
    ux, uy = u0
    nx_old, ny_old = ux.data.shape
    assert nx_old == ny_old
    factor = nx_old // nx_new
    ux_ds, uy_ds = (downsample_ux(ux.data, factor), downsample_uy(uy.data, factor))
    return get_velocity_cfd(ux_ds, uy_ds)


def vorticity_cfd_to_dg(vorticity_cfd, nx_new, ny_new, order_new):
    return convert_DG_representation(vorticity_cfd[...,None][None], order_new, 0, nx_new, ny_new, Lx, Ly)[0]


def get_u0(key):
    grid = grids.Grid((nx_max, ny_max), domain=((0, Lx), (0, Lx)))
    return cfd.initial_conditions.filtered_velocity_field(key, grid, max_velocity, ic_wavenumber)


def get_u0_fv(key, nx, ny):
    assert nx == ny
    u0 = get_u0(key)
    return downsample_u(u0, nx)

def get_u_fv(u, nx, ny):
    assert nx == ny
    return downsample_u(u, nx)


def get_v0_ps(key, nx, ny):
    u0 = get_u0(key)
    v0 = vorticity(u0)
    return vorticity_cfd_to_dg(v0, nx, ny, 0)[...,0]


def get_v_ps(u, nx, ny):
    v = vorticity(u)
    return vorticity_cfd_to_dg(v, nx, ny, 0)[...,0]


def get_v0_dg(key, nx, ny, order):
    u0 = get_u0(key)
    v0 = vorticity(u0)
    return vorticity_cfd_to_dg(v0, nx, ny, order)

def get_v_dg(u, nx, ny, order):
    v = vorticity(u)
    return vorticity_cfd_to_dg(v, nx, ny, order)


def get_dg_step_fn(args, nx, ny, order, T, cfl_safety=cfl_safety_dg):
    if order == 0:
        flux = Flux.VANLEER
    else:
        flux = Flux.UPWIND
    
    f_poisson_bracket = get_poisson_bracket(order, flux)
    f_poisson_solve = get_poisson_solver(nx, ny, Lx, Ly, order)
    f_phi = lambda zeta, t: f_poisson_solve(zeta)
    f_diffusion = get_diffusion_func(order, Lx, Ly, viscosity)
    f_forcing_sim = get_forcing_dg(order, nx, ny)
    
    inner_steps, dt = get_inner_steps_dt_DG(nx, ny, order, cfl_safety, T)

    @jax.jit
    def simulate(a_i):
        a, _ = simulate_2D(a_i, t0, nx, ny, Lx, Ly, order, dt, inner_steps, 
                           f_poisson_bracket, f_phi, a_data=None, output=False, f_diffusion=f_diffusion,
                            f_forcing=f_forcing_sim, rk=FUNCTION_MAP[runge_kutta])
        return a
    return simulate


def get_fv_step_fn(nx, ny, T, cfl_safety=cfl_safety_cfd):
    grid = get_grid(nx, ny)
    inner_steps, dt = get_inner_steps_dt_cfd(nx, ny, cfl_safety, T)
    step_fn = cfd.equations.semi_implicit_navier_stokes(
        density=density, 
        viscosity=viscosity, 
        forcing=get_forcing_cfd(nx, ny),
        dt=dt, 
        grid=grid)
    return jax.jit(cfd.funcutils.repeated(step_fn, inner_steps))


def get_ps_step_fn(nx, ny, T):
    grid = get_grid(nx, ny)
    inner_steps, dt = get_inner_steps_dt_cfd(nx, ny, cfl_safety_cfd, T)
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.NavierStokes2D(viscosity, grid, drag=damping_coefficient, smooth=True, 
            forcing_fn = get_forcing_ps(nx, ny)), dt)
    return jax.jit(cfd.funcutils.repeated(step_fn, inner_steps))


def store_correlation(n, args, exact_trajectory, nx, ny, name, j, v_j):
    v_e = convert_DG_representation(exact_trajectory[j][None], 0, order_exact, nx, ny, Lx, Ly)[0]
    M = np.concatenate([v_j[:,:,0].reshape(-1)[:,None], v_e[:,:,0].reshape(-1)[:,None]],axis=1)
    corrcoeff_j = np.corrcoef(M.T)[0,1]
    print("Run: {}, {}, nx = {}, T = {:.1f}, corr = {}".format(n, name, nx, j * t_chunk, corrcoeff_j))
    write_corr_file(n, args, name, nx, j, corrcoeff_j)


###################
# END HELPER FUNCTIONS
###################


def compute_corrcoef(n, args, key, orders):

    create_corr_file(n, args)


    ########
    # Store exact data
    ########

    exact_step_fn = get_dg_step_fn(args, nx_exact, ny_exact, order_exact, t_chunk, cfl_safety = cfl_safety_exact)
    exact_rollout_fn = get_trajectory_fn(exact_step_fn, outer_steps)

    #v0 = get_v0_dg(key, nx_exact, ny_exact, order_exact)
    u0 = get_u0(key)
    burn_in_step_fn = get_fv_step_fn(nx_burn_in, ny_burn_in, t_burn_in)
    u_burned_in = burn_in_step_fn(u0)
    v_burned_in = get_v_dg(u_burned_in, nx_exact, ny_exact, order_exact)

    exact_trajectory = exact_rollout_fn(-v_burned_in)
    exact_trajectory = concatenate_vorticity(v_burned_in, -exact_trajectory)

    store_corr_fn = lambda nx, ny, name, j, v_j:  store_correlation(n, args, exact_trajectory, nx, ny, name, j, v_j)
    

    ### FV Baseline

    for nx in nxs_fv_baseline:

        ny = nx
        u_fv_0 = get_u_fv(u_burned_in, nx, ny)
        step_fn = get_fv_step_fn(nx, ny, t_chunk)
        rollout_fn = get_trajectory_fn(step_fn, outer_steps)

        trajectory = rollout_fn(u_fv_0)
        trajectory_fv = concatenate_velocity(u_fv_0, trajectory)

        for j in range(outer_steps+1):
            u_j = get_velocity_cfd(trajectory_fv[0][j], trajectory_fv[1][j])
            store_corr_fn(nx, ny, "fv", j, shift_down_left(vorticity(u_j))[...,None])


    ### PS Baseline
    for nx in nxs_ps_baseline:
        ny = nx
        v_ps_0 = get_v_ps(u_burned_in, nx, ny)
        step_fn = get_ps_step_fn(nx, ny, t_chunk)
        rollout_fn = get_trajectory_fn(step_fn, outer_steps)

        v_hat0 = np.fft.rfftn(v_ps_0)
        trajectory_hat = rollout_fn(v_hat0)
        trajectory_hat_ps = concatenate_vorticity(v_hat0, trajectory_hat)
        trajectory_ps = np.fft.irfftn(trajectory_hat_ps, axes=(1,2))

        for j in range(outer_steps+1):
            store_corr_fn(nx, ny, "ps", j, trajectory_ps[j][...,None])

    # DG Baseline

    for o, order in enumerate(orders):
        for nx in nxs_dg[o]:
            ny = nx
            v_dg_0 = get_v_dg(u_burned_in, nx, ny, order)
            step_fn = get_dg_step_fn(args, nx, ny, order, t_chunk)
            rollout_fn = get_trajectory_fn(step_fn, outer_steps)

            trajectory = rollout_fn(-v_dg_0)
            trajectory_dg = concatenate_vorticity(v_dg_0, -trajectory)
            for j in range(outer_steps+1):
                store_corr_fn(nx, ny, "order{}".format(order), j, trajectory_dg[j])


def main():
    args = get_args()

    random_seed = 42
    key = jax.random.PRNGKey(random_seed)

    for n in range(N_test):
        key, _ = jax.random.split(key)
        compute_corrcoef(n, args, key, orders)
    




if __name__ == '__main__':
    main()