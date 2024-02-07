import jax
import jax.numpy as np
import numpy as onp
from jax import jit, vmap, config
import h5py
from functools import partial
from time import time
import jax_cfd.base as cfd
from jax_cfd.base import boundaries
from jax_cfd.base import forcings
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
config.update("jax_enable_x64", True)

from arguments import get_args
from flux import Flux
from initial_conditions import f_init_MLCFD

PI = np.pi



################
# PARAMETERS OF SIMULATION
################

PI = np.pi
Lx = 2 * PI
Ly = 2 * PI
density = 1.
max_velocity = 7.0
ic_wavenumber = 2
t0 = 0.0
Re = 1000
viscosity = 1/Re
forcing_coefficient = 1.0
damping_coefficient = 0.1
runge_kutta = "ssp_rk3"

nx_exact = 192
ny_exact = 192

nxs_fv_baseline = [512, 1024, 2048, 4096, 8192]
nxs_ps_baseline = [64, 128, 256, 512, 1024]    

cfl_safety_cfd = 0.5

N_compute_runtime = 3
t_runtime = 1.0

################
# END PARAMETERS
################


#############
# HELPER FUNCTIONS
#############


def kolmogorov_forcing_cfd(grid, scale, k):
    offsets = grid.cell_faces

    y = grid.mesh(offsets[0])[1]
    u = scale * grids.GridArray(np.sin(k * y), offsets[0], grid)

    if grid.ndim == 2:
        v = grids.GridArray(np.zeros_like(u.data), (1/2, 1), grid)
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
    offsets = ((0, 0), (0, 0))
    k=4
    forcing_fn = lambda grid: forcings.kolmogorov_forcing(grid, scale = forcing_coefficient, k=k, offsets=offsets)
    return forcing_fn

def get_velocity_cfd(u_x, u_y):
    assert u_x.shape == u_y.shape
    bcs = boundaries.periodic_boundary_conditions(2)
  
    grid = grids.Grid(u_x.shape, domain=((0, Lx), (0, Ly)))
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
    return cfd.equations.stable_time_step(max_velocity, cfl_safety_cfd, viscosity, get_grid(nx,ny))

def get_grid(nx, ny):
    return grids.Grid((nx, ny), domain=((0, Lx), (0, Ly)))


def get_inner_steps_dt_cfd(nx, ny, cfl_safety, T):
    inner_steps = int(T // get_dt_cfd(nx, ny)) + 1
    dt = T / inner_steps
    return inner_steps, dt

def vorticity(u):
    return cfd.finite_differences.curl_2d(u).data

def downsample_u(u0, nx_new):
    ux, uy = u0
    nx_old, ny_old = ux.data.shape
    assert nx_old == ny_old
    factor = nx_old // nx_new
    ux_ds, uy_ds = (downsample_ux(ux.data, factor), downsample_uy(uy.data, factor))
    return get_velocity_cfd(ux_ds, uy_ds)

def get_u0(key):
    nx_max = nxs_fv_baseline[-1]
    ny_max = nx_max
    grid = grids.Grid((nx_max, ny_max), domain=((0, Lx), (0, Lx)))
    return cfd.initial_conditions.filtered_velocity_field(key, grid, max_velocity, ic_wavenumber)

def get_u0_fv(key, nx, ny):
    assert nx == ny
    u0 = get_u0(key)
    return downsample_u(u0, nx)

def get_v0_ps(key, nx, ny):
    u0_ds = get_u0_fv(key, nx, ny)
    return vorticity(u0_ds)

def get_fv_step_fn(nx, ny, T):
    grid = get_grid(nx, ny)
    inner_steps, dt = get_inner_steps_dt_cfd(nx, ny, cfl_safety_cfd, T)
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


def create_datasets(args, device):
    f = h5py.File(
      "{}/data/{}_fv.hdf5".format(args.read_write_dir, device),
      "w",
    )
    for nx in nxs_fv_baseline:
        dset_a = f.create_dataset(str(nx), (1,), dtype="float64")
    f.close()

    f = h5py.File(
      "{}/data/{}_ps.hdf5".format(args.read_write_dir, device),
      "w",
    )
    for nx in nxs_ps_baseline:
        dset_a = f.create_dataset(str(nx), (1,), dtype="float64")
    f.close()


###################
# END HELPER FUNCTIONS
###################



def compute_runtime(args, key, device):

    create_datasets(args, device)

    ### FV Baseline

    for nx in nxs_fv_baseline:
        ny = nx
        u0 = get_u0_fv(key, nx, ny)
        step_fn = get_fv_step_fn(nx, ny, t_runtime)
        rollout_fn = get_trajectory_fn(step_fn, 1)

        trajectory = rollout_fn(u0)
        trajectory[0].array.data.block_until_ready()
        times = onp.zeros(N_compute_runtime)
        for n in range(N_compute_runtime):
            t1 = time()
            trajectory = rollout_fn(u0)
            trajectory[0].array.data.block_until_ready()
            t2 = time()
            times[n] = t2 - t1


        print("ML-CFD FV baseline, nx = {}".format(nx))
        print("runtime per unit time: {}".format(times / t_runtime))
        f = h5py.File(
          "{}/data/{}_fv.hdf5".format(args.read_write_dir, device),
          "r+",
        )
        f[str(nx)][0] = np.median(times / t_runtime)
        f.close()


    ### PS Baseline
    for nx in nxs_ps_baseline:
        ny = nx
        v0 = get_v0_ps(key, nx, ny)
        v_hat0 = np.fft.rfftn(v0)
        step_fn = get_ps_step_fn(nx, ny, t_runtime)
        rollout_fn = get_trajectory_fn(step_fn, 1)
        
        trajectory_hat = rollout_fn(v_hat0).block_until_ready()
        times = onp.zeros(N_compute_runtime)
        for n in range(N_compute_runtime):
            t1 = time()
            trajectory_hat = rollout_fn(v_hat0).block_until_ready()
            t2 = time()
            times[n] = t2 - t1

        print("ML-CFD PS baseline, nx = {}".format(nx))
        print("runtime per unit time: {}".format(times / t_runtime))
        f = h5py.File(
          "{}/data/{}_ps.hdf5".format(args.read_write_dir, device),
          "r+",
        )
        f[str(nx)][0] = np.median(times / t_runtime)
        f.close()


def main():
  args = get_args()


  from jax.lib import xla_bridge
  device = xla_bridge.get_backend().platform
  print("Device should be GPU, device is {}".format(device))

  key = jax.random.PRNGKey(42)

  compute_runtime(args, key, device)
  

if __name__ == '__main__':
  main()
