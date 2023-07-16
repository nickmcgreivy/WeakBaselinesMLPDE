import jax.numpy as np
from jax.lax import scan
import jax
from jax import vmap

from fluxdg import Flux
from polynomial_recovery import recovery_slope
from helperdg import _fixed_quad, inner_prod_with_legendre, map_f_to_DG


def _centered_flux_DG_1D_burgers(a, p):
    """
    Computes the centered flux F(f_{j+1/2}) where
    + is the right side.
    F(f_{j+1/2}) = (f_{j+1/2}^- + f_{j+1/2}^+) / 2
    where + = outside and - = inside.

    Inputs
    a: (nx, p) array

    Outputs
    F: (nx) array equal to f averaged.
    """
    a = np.pad(a, ((0, 1), (0, 0)), "wrap")
    alt = (np.ones(p) * -1) ** np.arange(p)
    u_left = np.sum(a[:-1], axis=-1)
    u_right = np.sum(alt[None, :] * a[1:], axis=-1)
    return ((u_left + u_right) / 2) ** 2 / 2
    # return ((u_left**2 + u_right**2) / 2) # THIS ONE IS BAD ON BURGERS


def _godunov_flux_DG_1D_burgers(a, p):
    """
    Computes the Godunov flux F(f_{j+1/2}) where
    + is the right side.

    Inputs
    a: (nx, p) array

    Outputs
    F: (nx) array equal to the godunov flux
    """
    a = np.pad(a, ((0, 1), (0, 0)), "wrap")
    alt = (np.ones(p) * -1) ** np.arange(p)
    u_left = np.sum(a[:-1], axis=-1)
    u_right = np.sum(alt[None, :] * a[1:], axis=-1)
    zero_out = 0.5 * np.abs(np.sign(u_left) + np.sign(u_right))
    compare = np.less(u_left, u_right)
    F = lambda u: u ** 2 / 2
    return compare * zero_out * np.minimum(F(u_left), F(u_right)) + (
        1 - compare
    ) * np.maximum(F(u_left), F(u_right))


def _flux_term_DG_1D_burgers(a, t, p, flux):
    negonetok = (np.ones(p) * -1) ** np.arange(p)
    if flux == Flux.CENTERED:
        flux_right = _centered_flux_DG_1D_burgers(a, p)
    elif flux == Flux.GODUNOV:
        flux_right = _godunov_flux_DG_1D_burgers(a, p)
    else:
        raise NotImplementedError

    flux_left = np.roll(flux_right, 1, axis=0)
    return negonetok[None, :] * flux_left[:, None] - flux_right[:, None]


def _volume_integral_DG_1D_burgers(a, t, p):
    if p == 1:
        volume_sum = np.zeros(a.shape)
        return volume_sum
    elif p == 2:
        volume_sum = np.zeros(a.shape).at[:, 1].add(1.0 * a[:, 0] * a[:, 0] + 0.3333333333333333 * a[:, 1] * a[:, 1])
        return volume_sum
    elif p == 3:
        volume_sum = np.zeros(a.shape).at[:, 1].add(1.0 * a[:, 0] * a[:, 0] + 0.3333333333333333 * a[:, 1] * a[:, 1] + 0.2 * a[:, 2] * a[:, 2])
        volume_sum = volume_sum.at[:, 2].add(
            1.0 * a[:, 0] * a[:, 1]
            + 1.0 * a[:, 1] * a[:, 0]
            + 0.4 * a[:, 1] * a[:, 2]
            + 0.4 * a[:, 2] * a[:, 1],
        )
        return volume_sum
    elif p == 4:
        volume_sum = np.zeros(a.shape).at[:, 1].add(1.0 * a[:, 0] * a[:, 0]
            + 0.3333333333333333 * a[:, 1] * a[:, 1]
            + 0.2 * a[:, 2] * a[:, 2]
            + 0.14285714285714285 * a[:, 3] * a[:, 3],
        )   
        volume_sum = volume_sum.at[:, 2].add(
            1.0 * a[:, 0] * a[:, 1]
            + 1.0 * a[:, 1] * a[:, 0]
            + 0.4 * a[:, 1] * a[:, 2]
            + 0.4 * a[:, 2] * a[:, 1]
            + 0.2571428571428571 * a[:, 2] * a[:, 3]
            + 0.2571428571428571 * a[:, 3] * a[:, 2],
        )
        volume_sum = volume_sum.at[:, 3].add(
            1.0 * a[:, 0] * a[:, 0]
            + 1.0 * a[:, 0] * a[:, 2]
            + 1.0 * a[:, 1] * a[:, 1]
            + 0.42857142857142855 * a[:, 1] * a[:, 3]
            + 1.0 * a[:, 2] * a[:, 0]
            + 0.4857142857142857 * a[:, 2] * a[:, 2]
            + 0.42857142857142855 * a[:, 3] * a[:, 1]
            + 0.3333333333333333 * a[:, 3] * a[:, 3],
        )
        return volume_sum
    else:
        raise NotImplementedError


def _diffusion_flux_term_DG_1D_burgers(a, t, p, dx, nu):
    negonetok = (np.ones(p) * -1) ** np.arange(p)
    slope_right = recovery_slope(a, p) / dx
    slope_left = np.roll(slope_right, 1)
    return nu * (slope_right[:, None] - negonetok[None, :] * slope_left[:, None])


def _diffusion_volume_integral_DG_1D_burgers(a, t, p, dx, nu):
    coeff = -2 * nu / dx
    if p == 1:
        volume_sum = np.zeros(a.shape)
    elif p == 2:
        volume_sum = np.zeros(a.shape).at[:, 1].add(2.0 * a[:, 1])
    elif p == 3:
        volume_sum = np.zeros(a.shape).at[:, 1].add(2.0 * a[:, 1])
        volume_sum = volume_sum.at[:, 2].add(6.0 * a[:, 2])
    elif p == 4:
        volume_sum = np.zeros(a.shape).at[:, 1].add(2.0 * a[:, 1] + 2.0 * a[:, 3])
        volume_sum = volume_sum.at[:, 2].add(6.0 * a[:, 2])
        volume_sum = volume_sum.at[:, 3].add(2.0 * a[:, 1] + 12.0 * a[:, 3])
    else:
        raise NotImplementedError
    return coeff * volume_sum


def time_derivative_DG_1D_burgers(
    a, t, p, flux, nx, dx, leg_poly, forcing_func=None, nu=0.0
):
    """
    Compute da_j^m/dt given the matrix a_j^m which represents the solution,
    for a given flux. The time-derivative is given by a Galerkin minimization
    of the residual squared, with Legendre polynomial basis functions.
    For the 1D burgers equation
            df/dt + c df/dx = 0
    with f_j = \sum a_j^m \phi_m, the time derivatives equal

    da_j^m/dt = ...

    Inputs
    a: (nx, p) array of coefficients
    t: time, scalar, not used here
    c: speed (scalar)
    flux: Enum, decides which flux will be used for the boundary

    Outputs
    da_j^m/dt: (nx, p) array of time derivatives
    """
    twokplusone = 2 * np.arange(0, p) + 1
    flux_term = _flux_term_DG_1D_burgers(a, t, p, flux)
    volume_integral = _volume_integral_DG_1D_burgers(a, t, p)
    dif_flux_term = _diffusion_flux_term_DG_1D_burgers(a, t, p, dx, nu)
    dif_volume_integral = _diffusion_volume_integral_DG_1D_burgers(a, t, p, dx, nu)
    if forcing_func is not None:
        forcing_term = inner_prod_with_legendre(forcing_func, t, p, nx, dx, leg_poly, n = 10)
    else:
        forcing_term = 0.0
    return (twokplusone[None, :] / dx) * (flux_term + volume_integral + dif_flux_term + dif_volume_integral + forcing_term)
