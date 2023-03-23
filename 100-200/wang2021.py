import jax.numpy as jnp
from jax import vmap, jit
from jax.lax import scan
import matplotlib.pyplot as plt
from jax.config import config
import torch
import math
import numpy as onp
import matplotlib as mpl
from functools import partial
mpl.rcParams.update(mpl.rcParamsDefault)
config.update("jax_enable_x64", True)
vmap_polyval = vmap(jnp.polyval, (0, None), -1)


PI = jnp.pi

class GaussianRF(object):

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        else:
            raise Exception

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real



def get_initial_condition_GRF(nx=100):
    GRF = GaussianRF(1, nx, alpha=4, tau=5, sigma=5**4)
    return jnp.asarray(GRF.sample(1)[0])



def downsample(a, SCALE):
    return jnp.mean(a.reshape(-1, SCALE), axis=-1)




def ssp_rk3(a_n, t_n, F, dt):
    a_1 = a_n + dt * F(a_n, t_n)
    a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * F(a_1, t_n + dt))
    a_3 = 1 / 3 * a_n + 2 / 3 * (a_2 + dt * F(a_2, dt + dt / 2))
    return a_3, t_n + dt


def _scan(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), None


def _scan_output(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), a_f



def _godunov_flux_1D_burgers(a):
    u_left = a
    u_right = jnp.roll(a, -1)
    zero_out = 0.5 * jnp.abs(jnp.sign(u_left) + jnp.sign(u_right))
    compare = jnp.less(u_left, u_right)
    F = lambda u: u**2 / 2
    return compare * zero_out * jnp.minimum(F(u_left), F(u_right)) + (
        1 - compare
    ) * jnp.maximum(F(u_left), F(u_right))


def time_derivative_1D_burgers(a, t, nx, dx, nu):
    flux_right = _godunov_flux_1D_burgers(a)
    flux_right = flux_right - (nu / dx) * (jnp.roll(a, -1) - a)

    flux_left = jnp.roll(flux_right, 1)
    return (flux_left - flux_right) / dx


@partial(jit, static_argnums=(2,5,7))
def simulate_1D(a0, t0, nx, dx, dt, nt, nu, output=False):
    dadt = lambda a, t: time_derivative_1D_burgers(a, t, nx, dx, nu)
    rk_F = lambda a, t: ssp_rk3(a, t, dadt, dt)

    if output:
        scanf = jit(lambda sol, x: _scan_output(sol, x, rk_F))
        _, data = scan(scanf, (a0, t0), None, length=nt)
        return data
    else:
        scanf = jit(lambda sol, x: _scan(sol, x, rk_F))
        (a_f, t_f), _ = scan(scanf, (a0, t0), None, length=nt)
        return (a_f, t_f)


def plot_subfig(
    a, subfig, L, color="blue", linewidth=0.5, linestyle="solid", label=None
):
    def evalf(x, a, j, dx):
        x_j = dx * (0.5 + j)
        xi = (x - x_j) / (0.5 * dx)
        vmap_polyval = vmap(jnp.polyval, (0, None), -1)
        poly_eval = vmap_polyval(jnp.asarray([[1.0]]), xi)
        return jnp.sum(poly_eval * a, axis=-1)

    nx = a.shape[0]
    dx = L / nx
    xjs = jnp.arange(nx) * L / nx
    xs = xjs[None, :] + jnp.linspace(0.0, dx, 10)[:, None]
    vmap_eval = vmap(evalf, (1, 0, 0, None), 1)
    subfig.plot(
        xs,
        vmap_eval(xs, a, jnp.arange(nx), dx),
        color=color,
        linewidth=linewidth,
        label=label,
        linestyle=linestyle,
    )
    return


def relative_l2_error(a, a_ex):
    assert len(a.shape) == 1
    assert len(a_ex.shape) == 1
    diff = a - a_ex
    return jnp.sqrt(jnp.sum(diff**2) / jnp.sum(a_ex**2))

trajectory_relative_l2_error = jit(vmap(relative_l2_error, (0, 0)))


#############
# HYPERPARAMS
#############


Tf = 1.0
L = 1.0
nu = 0.01
cfl_safety = 0.5
alpha_max = 0.5
t0 = 0.0
N_test = 1000
N_time = 100

nx_exact = 400
nxs = [25, 50, 100]
DWS = [8*16, 4*8, 4*4]

dx_exact = L / nx_exact
dt_exact = min(cfl_safety * dx_exact / alpha_max, cfl_safety * dx_exact**2 / nu)
nt_exact = Tf // dt_exact
nt_exact = (DWS[0] - (nt_exact % DWS[0])) + nt_exact
for DW in DWS:
    assert nt_exact % DW == 0
dt_exact = Tf / nt_exact


nts = [int(nt_exact // DW) for DW in DWS]
dts = [dt_exact * DW for DW in DWS]
dxs = [L / nx for nx in nxs]





################
# COMPUTE ERRORS
################

relative_errors = [0.0, 0.0, 0.0]

for i in range(N_test):

    a0_exact = get_initial_condition_GRF(nx = nx_exact)
    traj_exact = simulate_1D(a0_exact, t0, nx_exact, dx_exact, dt_exact, nt_exact, nu, output=True)
    traj_exact = jnp.concatenate((a0_exact[None], traj_exact), axis=0)

    for i, nx in enumerate(nxs):
        DS = nx_exact // nx
        a0 = downsample(a0_exact, DS)
        assert a0.shape[0] == nx
        dx = dxs[i]
        nt = nts[i]
        dt = dts[i]
        DW = DWS[i]

        traj = simulate_1D(a0, t0, nx, dx, dt, nt, nu, output=True)
        traj = jnp.concatenate((a0[None], traj), axis=0)

        traj_exact_ds = vmap(downsample, (0, None))(traj_exact, DS)[::DW]

        assert traj_exact_ds.shape[0] == traj.shape[0]
        assert traj_exact_ds.shape[1] == traj.shape[1]

        traj_l2_error = jnp.mean(trajectory_relative_l2_error(traj, traj_exact_ds))

        relative_errors[i] += traj_l2_error / N_test

for i, nx in enumerate(nxs):
    print("nx: {} Mean relative L2 error: {}".format(nx, relative_errors[i]))


################
# COMPUTE RUNTIME
################

from time import time



a0_exact = get_initial_condition_GRF(nx = nx_exact)


for i, nx in enumerate(nxs):
        DS = nx_exact // nx
        a0 = downsample(a0_exact, DS)
        assert a0.shape[0] == nx
        dx = dxs[i]
        nt = nts[i]
        dt = dts[i]
        DW = DWS[i]

        @jit
        def sim_f(a0):
            a_f, _ = simulate_1D(a0, t0, nx, dx, dt, nt, nu, output=False)
            return a_f

        a_f = sim_f(a0).block_until_ready()

        ave_time = 0.0

        for _ in range(N_time):
            t1 = time()
            a_f = sim_f(a0).block_until_ready()
            t2 = time()
            ave_time += (t2 - t1) / N_time

        print("nx: {}, runtime: {}".format(nx, ave_time))



################
# COMPUTE RUNTIME VS NUM SIMS
################



nsims = [1, 5, 10, 20, 50, 100]#, 200, 500, 1000]



for nsim in nsims:

    i = -1
    nx = nxs[i]
    dx = dxs[i]
    nt = nts[i]
    dt = dts[i]
    DW = DWS[i]

    a0s = onp.zeros((nsim, nx))

    for i in range(nsim):
        a0_exact = get_initial_condition_GRF(nx = nx_exact)
        a0_ds = downsample(a0_exact, DS)
        a0s[i] = a0_ds

    a0s = jnp.asarray(a0s)

    @jit
    @vmap
    def sim_f(a0):
        a_f, _ = simulate_1D(a0, t0, nx, dx, dt, nt, nu, output=False)
        return a_f

    afs = sim_f(a0s).block_until_ready()

    ave_time = 0.0
    N_time = 10
    for _ in range(N_time):
        t1 = time()
        a_f = sim_f(a0s).block_until_ready()
        t2 = time()
        ave_time += (t2 - t1) / N_time

    print("# PDEs solved: {} runtime: {}".format(nsim, ave_time))


"""
fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, squeeze=True, figsize=(8, 3))

plot_subfig(
    a0_exact,
    axs[0],
    L,
    color="grey",
    label="Initial\nconditions",
    linewidth=1.2,
)

plot_subfig(
    traj_exact_ds[-1],
    axs[1],
    L,
    color="grey",
    label="Initial\nconditions",
    linewidth=1.2,
)

plt.show()
"""