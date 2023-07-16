import jax.numpy as np
from functools import partial
import numpy as onp
from basisfunctions import (
    legendre_npbasis,
    node_locations,
    legendre_inner_product,
    legendre_poly,
)
from jax import vmap, hessian, jit


def _trapezoidal_integration(f, xi, xf, yi, yf, n=None):
    return (xf - xi) * (yf - yi) * (f(xi, yi) + f(xf, yi) + f(xi, yf) + f(xf, yf)) / 4


def _2d_fixed_quad(f, xi, xf, yi, yf, n=3):
    """
    Takes a 2D-valued function of two 2D inputs
    f(x,y) and four scalars xi, xf, yi, yf, and
    integrates f over the 2D domain to order n.
    """

    w_1d = {
        1: np.asarray([2.0]),
        2: np.asarray([1.0, 1.0]),
        3: np.asarray(
            [
                0.5555555555555555555556,
                0.8888888888888888888889,
                0.555555555555555555556,
            ]
        ),
        4: np.asarray(
            [
                0.3478548451374538573731,
                0.6521451548625461426269,
                0.6521451548625461426269,
                0.3478548451374538573731,
            ]
        ),
        5: np.asarray(
            [
                0.2369268850561890875143,
                0.4786286704993664680413,
                0.5688888888888888888889,
                0.4786286704993664680413,
                0.2369268850561890875143,
            ]
        ),
        6: np.asarray(
            [
                0.1713244923791703450403,
                0.3607615730481386075698,
                0.4679139345726910473899,
                0.4679139345726910473899,
                0.3607615730481386075698,
                0.1713244923791703450403,
            ]
        ),
        7: np.asarray(
            [
                0.1294849661688696932706,
                0.2797053914892766679015,
                0.38183005050511894495,
                0.417959183673469387755,
                0.38183005050511894495,
                0.279705391489276667901,
                0.129484966168869693271,
            ]
        ),
        8: np.asarray(
            [
                0.1012285362903762591525,
                0.2223810344533744705444,
                0.313706645877887287338,
                0.3626837833783619829652,
                0.3626837833783619829652,
                0.313706645877887287338,
                0.222381034453374470544,
                0.1012285362903762591525,
            ]
        ),
    }[n]

    xi_i_1d = {
        1: np.asarray([0.0]),
        2: np.asarray([-0.5773502691896257645092, 0.5773502691896257645092]),
        3: np.asarray([-0.7745966692414833770359, 0.0, 0.7745966692414833770359]),
        4: np.asarray(
            [
                -0.861136311594052575224,
                -0.3399810435848562648027,
                0.3399810435848562648027,
                0.861136311594052575224,
            ]
        ),
        5: np.asarray(
            [
                -0.9061798459386639927976,
                -0.5384693101056830910363,
                0.0,
                0.5384693101056830910363,
                0.9061798459386639927976,
            ]
        ),
        6: np.asarray(
            [
                -0.9324695142031520278123,
                -0.661209386466264513661,
                -0.2386191860831969086305,
                0.238619186083196908631,
                0.661209386466264513661,
                0.9324695142031520278123,
            ]
        ),
        7: np.asarray(
            [
                -0.9491079123427585245262,
                -0.7415311855993944398639,
                -0.4058451513773971669066,
                0.0,
                0.4058451513773971669066,
                0.7415311855993944398639,
                0.9491079123427585245262,
            ]
        ),
        8: np.asarray(
            [
                -0.9602898564975362316836,
                -0.7966664774136267395916,
                -0.5255324099163289858177,
                -0.1834346424956498049395,
                0.1834346424956498049395,
                0.5255324099163289858177,
                0.7966664774136267395916,
                0.9602898564975362316836,
            ]
        ),
    }[n]

    x_w, y_w = np.meshgrid(w_1d, w_1d)
    x_w = x_w.reshape(-1)
    y_w = y_w.reshape(-1)
    w_2d = x_w * y_w

    xi_x, xi_y = np.meshgrid(xi_i_1d, xi_i_1d)
    xi_x = xi_x.reshape(-1)
    xi_y = xi_y.reshape(-1)

    x_i = (xf + xi) / 2 + (xf - xi) / 2 * xi_x
    y_i = (yf + yi) / 2 + (yf - yi) / 2 * xi_y
    wprime = w_2d * (xf - xi) * (yf - yi) / 4
    return np.sum(wprime[None, :] * f(x_i, y_i), axis=1)


def evalf_2D(x, y, a, dx, dy, order):
    """
    Returns the value of DG representation of the
    solution at x, y, where x,y is a 2D array of points

    Inputs:
    x, y: 2D array of points
    a: DG representation, (nx, ny, num_elements) ndarray

    Ouputs:
    f: 2d array of points, equal to sum over num_elements polynomials
    """
    j = np.floor(x / dx).astype(int)
    k = np.floor(y / dy).astype(int)
    x_j = dx * (0.5 + j)
    y_k = dy * (0.5 + k)
    xi_x = (x - x_j) / (0.5 * dx)
    xi_y = (y - y_k) / (0.5 * dy)
    f_eval = _eval_legendre(order)
    legendre_val = np.transpose(f_eval(xi_x, xi_y), (1, 2, 0))
    return np.sum(a[j, k, :] * legendre_val, axis=-1)


def _evalf_2D_integrate(x, y, a, dx, dy, order):
    """
    Returns the value of DG representation of the
    solution at x, y, where x and y are a 1d array of points

    Inputs:
    x, y: 1D array of points
    a: DG representation, (nx, ny, num_elements) ndarray

    Ouputs:
    f: 2d array of points, equal to sum over num_elements polynomials
    """
    j = np.floor(x / dx).astype(int)
    k = np.floor(y / dy).astype(int)
    x_j = dx * (0.5 + j)
    y_k = dy * (0.5 + k)
    xi_x = (x - x_j) / (0.5 * dx)
    xi_y = (y - y_k) / (0.5 * dy)
    f_eval = _eval_legendre(order)
    return np.sum(a[j, k, :] * f_eval(xi_x, xi_y).T, axis=-1)


def _eval_legendre(order):
    """
    Takes two 1D vectors xi_x and xi_y, outputs
    the 2D legendre basis at (xi_x, xi_y)
    """
    polybasis = legendre_npbasis(order)  # (order+1, k, 2) matrix
    _vmap_polyval = vmap(np.polyval, (1, None), 0)

    def f(xi_x, xi_y):
        return _vmap_polyval(polybasis[:, :, 0], xi_x) * _vmap_polyval(
            polybasis[:, :, 1], xi_y
        )  # (k,) array

    return f


def inner_prod_with_legendre(nx, ny, Lx, Ly, order, func, t, quad_func=_2d_fixed_quad, n=5):
    dx = Lx / nx
    dy = Ly / ny

    i = np.arange(nx)
    x_i = dx * i
    x_f = dx * (i + 1)
    j = np.arange(ny)
    y_i = dy * j
    y_f = dy * (j + 1)

    def xi_x(x):
        k = np.floor(x / dx)
        x_k = dx * (0.5 + k)
        return (x - x_k) / (0.5 * dx)

    def xi_y(y):
        k = np.floor(y / dy)
        y_k = dy * (0.5 + k)
        return (y - y_k) / (0.5 * dy)

    quad_lambda = lambda f, xi, xf, yi, yf: quad_func(f, xi, xf, yi, yf, n=n)

    _vmap_integrate = vmap(
        vmap(quad_lambda, (None, 0, 0, None, None), 0), (None, None, None, 0, 0), 1
    )
    to_int_func = lambda x, y: func(x, y, t) * _eval_legendre(order)(xi_x(x), xi_y(y))
    return _vmap_integrate(to_int_func, x_i, x_f, y_i, y_f)

def f_to_DG(nx, ny, Lx, Ly, order, func, t, quad_func=_2d_fixed_quad, n=5):
    """
    Takes a function f of type lambda x, y, t: f(x,y,t) and
    generates the DG representation of the solution, an
    array of size (nx, ny, p).

    Computes the inner product of f with p Legendre polynomials
    over nx regions, to produce an array of size (nx, p)

    Inputs
    f: lambda x, y, t: f(x, y, t), the value of f
    t: the current time

    Outputs
    a0: The DG representation of f(x, y, t) at t=t
    """
    inner_prod = legendre_inner_product(order)
    dx = Lx / nx
    dy = Ly / ny

    return inner_prod_with_legendre(nx, ny, Lx, Ly, order, func, t, quad_func=quad_func, n=n) / (
        inner_prod[None, None, :] * dx * dy
    )


def f_to_source(nx, ny, Lx, Ly, order, func, t, quad_func=_2d_fixed_quad, n=5):
    repr_dg = f_to_DG(nx, ny, Lx, Ly, order, func, t, quad_func=quad_func, n=n)
    return repr_dg.at[:, :, 0].add(-np.mean(repr_dg[:, :, 0]))


def f_to_FE(nx, ny, Lx, Ly, order, func, t):
    dx = Lx / nx
    dy = Ly / ny
    i = np.arange(nx)
    x_i = dx * i + dx / 2
    j = np.arange(ny)
    y_i = dy * j + dx / 2
    nodes = np.asarray(node_locations(order), dtype=float)

    x_eval = (
        np.ones((nx, ny, nodes.shape[0])) * x_i[:, None, None]
        + nodes[None, None, :, 0] * dx / 2
    )
    y_eval = (
        np.ones((nx, ny, nodes.shape[0])) * y_i[None, :, None]
        + nodes[None, None, :, 1] * dy / 2
    )
    FE_repr = np.zeros((nx, ny, nodes.shape[0]))

    _vmap_evaluate = vmap(vmap(vmap(func, (0, 0, None)), (0, 0, None)), (0, 0, None))
    return _vmap_evaluate(x_eval, y_eval, t)

@partial(
    jit,
    static_argnums=(
        1,
        2,
        3,
        4,
        7,
    ),
)
def convert_DG_representation(
    a, order_new, order_high, nx_new, ny_new, Lx, Ly, n = 8
):
    """
    Inputs:
    a: (nt, nx, ny, num_elements(order_high))

    Outputs:
    a_converted: (nt, nx_new, ny_new, num_elements(order_new))
    """
    _, nx_high, ny_high = a.shape[0:3]
    dx_high = Lx / nx_high
    dy_high = Ly / ny_high

    def convert_repr(a):
        def f_high(x, y, t):
            return _evalf_2D_integrate(x, y, a, dx_high, dy_high, order_high)

        t0 = 0.0
        return f_to_DG(nx_new, ny_new, Lx, Ly, order_new, f_high, t0, n=n)

    vmap_convert_repr = vmap(convert_repr)
    return vmap_convert_repr(a)


def vorticity_to_velocity(Lx, Ly, a, f_poisson):
    H = f_poisson(a)
    nx, ny, _ = H.shape
    dx = Lx / nx
    dy = Ly / ny

    u_y = -(H[:,:,2] - H[:,:,3]) / dx
    u_x = (H[:,:,2] - H[:,:,1]) / dy
    return u_x, u_y


def nabla(f):
    """
    Takes a function of type f(x,y) and returns a function del^2 f(x,y)
    """
    H = hessian(f)
    return lambda x, y: np.trace(H(x, y))


def minmod(r):
    return np.maximum(0, np.minimum(1, r))


def minmod_2(z1, z2):
    s = 0.5 * (np.sign(z1) + np.sign(z2))
    return s * np.minimum(np.absolute(z1), np.absolute(z2))


def minmod_3(z1, z2, z3):
    s = (
        0.5
        * (np.sign(z1) + np.sign(z2))
        * np.absolute(0.5 * ((np.sign(z1) + np.sign(z3))))
    )
    return s * np.minimum(np.absolute(z1), np.minimum(np.absolute(z2), np.absolute(z3)))





###### test helper

"""


import jax
import numpy as onp
from initial_conditions import f_init_MLCFD, get_initial_condition_FNO, f_init_CNO
import matplotlib.pyplot as plt
import seaborn as sns

def plot_DG_basis(
    nx, ny, Lx, Ly, order, zeta, plot_lines=False, title="", plotting_density=4
):
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
    fig, axs = plt.subplots(figsize=(5 * onp.sqrt(Lx / Ly) + 1, onp.sqrt(Ly / Lx) * 5))
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
    fig.colorbar(pcm, ax=axs, extend="max")

    if plot_lines:
        fig, axs = plt.subplots(figsize=(5 * onp.sqrt(Lx / Ly), onp.sqrt(Ly / Lx) * 5))
        for j in range(0, Nx_plot, 10):
            axs.plot(x_plot[:-1], output[j, :])
        fig, axs = plt.subplots(figsize=(5 * onp.sqrt(Lx / Ly), onp.sqrt(Ly / Lx) * 5))
        for j in range(0, Ny_plot, 10):
            axs.plot(y_plot[:-1], output[:, j])

def MSE(a, a_ex):
    return np.mean((a-a_ex)**2 / legendre_inner_product(order)[None, None, :])


key = jax.random.PRNGKey(11)
PI = np.pi
f_test = f_init_CNO(key) #lambda x, y, t: np.sin(2 * np.pi * 5.90913910 * x) * np.sin(2 * np.pi * 8.23902901124 * y)
Lx = 1.0
Ly = 1.0
order = 2
nx_exact = 64
ny_exact = nx_exact
nxs = [2, 4, 8, 16, 32]
a_exact = f_to_DG(nx_exact, ny_exact, Lx, Ly, order, f_test, 0.0, n=8)

print(a_exact)

plot_DG_basis(nx_exact, ny_exact, Lx, Ly, order, a_exact, plotting_density=4)
plt.show()




n_final = 8
errors = onp.zeros((len(nxs), n_final))

for i, nx in enumerate(nxs):
    ny = nx

    a_exact_ds = convert_DG_representation(a_exact[None], order, order, nx, ny, Lx, Ly, n=8)[0]

    for n in range(1, n_final+1):

        a = f_to_DG(nx, ny, Lx, Ly, order, f_test, 0.0, n=n)

        errors[i,n-1] = MSE(a, a_exact_ds)

plt.loglog(nxs, errors)
plt.show()

"""



