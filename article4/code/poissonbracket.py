import jax.numpy as np
from functools import lru_cache
import os
from jax import config
import numpy as onp
import time
import jax

from flux import Flux
from helper import f_to_DG, minmod_3
from basisfunctions import (
    create_poisson_bracket_volume_matrix,
    create_poisson_bracket_boundary_matrix_centered,
    get_leftright_alternate,
    get_topbottom_alternate,
    create_poisson_bracket_boundary_matrix_upwind,
    boundary_matrix_inverse,
    interpolation_points,
    alpha_right_matrix,
    alpha_top_matrix,
    zeta_right_minus_matrix,
    zeta_right_plus_matrix,
    zeta_top_minus_matrix,
    zeta_top_plus_matrix,
    alpha_right_matrix_twice,
    alpha_top_matrix_twice,
    zeta_right_minus_matrix_twice,
    zeta_right_plus_matrix_twice,
    zeta_top_minus_matrix_twice,
    zeta_top_plus_matrix_twice,
    get_leftright_alternate,
    get_topbottom_alternate,
    leg_FE_inner_product,
    legendre_inner_product,
    boundary_matrix_inverse_twice,
    legendre_boundary_inner_product,
    change_basis_boundary_to_volume,
    deriv_y_leg_FE_inner_product,
    leg_FE_bottom_integrate,
    leg_FE_top_integrate,
    change_legendre_points_twice,
)

config.update("jax_enable_x64", True)


@lru_cache(maxsize=4)
def load_poisson_volume(order):
    if os.path.exists(
        "../../poissonmatrices/poisson_bracket_volume_{}.npy".format(order)
    ):
        V = onp.load(
            "../../poissonmatrices/poisson_bracket_volume_{}.npy".format(
                order
            )
        )
    else:
        V = create_poisson_bracket_volume_matrix(order)
        onp.save(
            "../../poissonmatrices/poisson_bracket_volume_{}.npy".format(
                order
            ),
            V,
        )
    return V


@lru_cache(maxsize=4)
def load_boundary_matrix_centered(order):
    if os.path.exists(
        "../../poissonmatrices/poisson_bracket_boundary_centered_{}.npy".format(
            order
        )
    ):
        B = onp.load(
            "../../poissonmatrices/poisson_bracket_boundary_centered_{}.npy".format(
                order
            )
        )
    else:
        B = create_poisson_bracket_boundary_matrix_centered(order)
        onp.save(
            "../../poissonmatrices/poisson_bracket_boundary_centered_{}.npy".format(
                order
            ),
            B,
        )
    return B


@lru_cache(maxsize=4)
def load_boundary_matrix_upwind(order):
    if os.path.exists(
        "../../poissonmatrices/poisson_bracket_boundary_upwind_{}.npy".format(
            order
        )
    ):
        B = onp.load(
            "../../poissonmatrices/poisson_bracket_boundary_upwind_{}.npy".format(
                order
            )
        )
    else:
        B = create_poisson_bracket_boundary_matrix_upwind(order)
        onp.save(
            "../../poissonmatrices/poisson_bracket_boundary_upwind_{}.npy".format(
                order
            ),
            B,
        )
    return B


def load_alpha_right_matrix_twice(order):
    if os.path.exists(
        "../../poissonmatrices/alpha_right_matrix_{}.npy".format(order)
    ):
        R = onp.load(
            "../../poissonmatrices/alpha_right_matrix_{}.npy".format(order)
        )
    else:
        R = alpha_right_matrix_twice(order)
        onp.save(
            "../../poissonmatrices/alpha_right_matrix_{}.npy".format(order),
            R,
        )
    return R


def load_alpha_top_matrix_twice(order):
    if os.path.exists(
        "../../poissonmatrices/alpha_top_matrix_{}.npy".format(order)
    ):
        T = onp.load(
            "../../poissonmatrices/alpha_top_matrix_{}.npy".format(order)
        )
    else:
        T = alpha_top_matrix_twice(order)
        onp.save(
            "../../poissonmatrices/alpha_top_matrix_{}.npy".format(order),
            T,
        )
    return T


def load_zeta_right_minus_matrix_twice(order):
    if os.path.exists(
        "../../poissonmatrices/zeta_right_minus_matrix_{}.npy".format(order)
    ):
        Rm = onp.load(
            "../../poissonmatrices/zeta_right_minus_matrix_{}.npy".format(
                order
            )
        )
    else:
        Rm = zeta_right_minus_matrix_twice(order)
        onp.save(
            "../../poissonmatrices/zeta_right_minus_matrix_{}.npy".format(
                order
            ),
            Rm,
        )
    return Rm


def load_zeta_right_plus_matrix_twice(order):
    if os.path.exists(
        "../../poissonmatrices/zeta_right_plus_matrix_{}.npy".format(order)
    ):
        Rp = onp.load(
            "../../poissonmatrices/zeta_right_plus_matrix_{}.npy".format(
                order
            )
        )
    else:
        Rp = zeta_right_plus_matrix_twice(order)
        onp.save(
            "../../poissonmatrices/zeta_right_plus_matrix_{}.npy".format(
                order
            ),
            Rp,
        )
    return Rp


def load_zeta_top_minus_matrix_twice(order):
    if os.path.exists(
        "../../poissonmatrices/zeta_top_minus_matrix_{}.npy".format(order)
    ):
        Tm = onp.load(
            "../../poissonmatrices/zeta_top_minus_matrix_{}.npy".format(
                order
            )
        )
    else:
        Tm = zeta_top_minus_matrix_twice(order)
        onp.save(
            "../../poissonmatrices/zeta_top_minus_matrix_{}.npy".format(
                order
            ),
            Tm,
        )
    return Tm


def load_zeta_top_plus_matrix_twice(order):
    if os.path.exists(
        "../../poissonmatrices/zeta_top_plus_matrix_{}.npy".format(order)
    ):
        Tp = onp.load(
            "../../poissonmatrices/zeta_top_plus_matrix_{}.npy".format(order)
        )
    else:
        Tp = zeta_top_plus_matrix_twice(order)
        onp.save(
            "../../poissonmatrices/zeta_top_plus_matrix_{}.npy".format(
                order
            ),
            Tp,
        )
    return Tp


def load_boundary_matrix_inverse_twice(order):
    if os.path.exists(
        "../../poissonmatrices/boundary_matrix_inverse_{}.npy".format(order)
    ):
        P_inv = onp.load(
            "../../poissonmatrices/boundary_matrix_inverse_{}.npy".format(
                order
            )
        )
    else:
        P_inv = boundary_matrix_inverse_twice(order)
        onp.save(
            "../../poissonmatrices/boundary_matrix_inverse_{}.npy".format(
                order
            ),
            P_inv,
        )
    return P_inv


def load_legendre_boundary_inner_product(order):
    if os.path.exists(
        "../../poissonmatrices/legendre_boundary_inner_product_{}.npy".format(
            order
        )
    ):
        boundary_ip = onp.load(
            "../../poissonmatrices/legendre_boundary_inner_product_{}.npy".format(
                order
            )
        )
    else:
        boundary_ip = legendre_boundary_inner_product(order)
        onp.save(
            "../../poissonmatrices/legendre_boundary_inner_product_{}.npy".format(
                order
            ),
            boundary_ip,
        )
    return boundary_ip


def load_change_basis_boundary_to_volume(order):
    if os.path.exists(
        "../../poissonmatrices/change_basis_boundary_to_volume_CB_R_{}.npy".format(
            order
        )
    ):
        CB_R = onp.load(
            "../../poissonmatrices/change_basis_boundary_to_volume_CB_R_{}.npy".format(
                order
            )
        )
        CB_T = onp.load(
            "../../poissonmatrices/change_basis_boundary_to_volume_CB_T_{}.npy".format(
                order
            )
        )
        CB_L = onp.load(
            "../../poissonmatrices/change_basis_boundary_to_volume_CB_L_{}.npy".format(
                order
            )
        )
        CB_B = onp.load(
            "../../poissonmatrices/change_basis_boundary_to_volume_CB_B_{}.npy".format(
                order
            )
        )
    else:
        CB_R, CB_T, CB_L, CB_B = change_basis_boundary_to_volume(order)
        onp.save(
            "../../poissonmatrices/change_basis_boundary_to_volume_CB_R_{}.npy".format(
                order
            ),
            CB_R,
        )
        onp.save(
            "../../poissonmatrices/change_basis_boundary_to_volume_CB_T_{}.npy".format(
                order
            ),
            CB_T,
        )
        onp.save(
            "../../poissonmatrices/change_basis_boundary_to_volume_CB_L_{}.npy".format(
                order
            ),
            CB_L,
        )
        onp.save(
            "../../poissonmatrices/change_basis_boundary_to_volume_CB_B_{}.npy".format(
                order
            ),
            CB_B,
        )
    return CB_R, CB_T, CB_L, CB_B


def load_change_legendre_points_twice(order):
    if os.path.exists(
        "../../poissonmatrices/change_legendre_points_{}.npy".format(order)
    ):
        LP = onp.load(
            "../../poissonmatrices/change_legendre_points_{}.npy".format(
                order
            )
        )
    else:
        LP = change_legendre_points_twice(order)
        onp.save(
            "../../poissonmatrices/change_legendre_points_{}.npy".format(
                order
            ),
            LP,
        )
    return LP


def get_poisson_bracket(order, flux):
    V = load_poisson_volume(order)
    R = load_alpha_right_matrix_twice(order)
    T = load_alpha_top_matrix_twice(order)
    Rm = load_zeta_right_minus_matrix_twice(order)
    Rp = load_zeta_right_plus_matrix_twice(order)
    Tm = load_zeta_top_minus_matrix_twice(order)
    Tp = load_zeta_top_plus_matrix_twice(order)
    P_inv = load_boundary_matrix_inverse_twice(order)[: order + 1, :]
    boundary_ip = load_legendre_boundary_inner_product(order)
    CB_R, CB_T, CB_L, CB_B = load_change_basis_boundary_to_volume(order)
    # N stands for normalized
    CBN_R, CBN_T, CBN_L, CBN_B = (
        CB_R * boundary_ip[:, None],
        CB_T * boundary_ip[:, None],
        CB_L * boundary_ip[:, None],
        CB_B * boundary_ip[:, None],
    )

    LP = load_change_legendre_points_twice(order)

    def centered(zeta, H):
        alpha_R_points, alpha_T_points = H @ R.T, H @ T.T  # right, top
        zeta_R_points_minus = zeta @ Rm.T
        zeta_R_points_plus = np.roll(zeta, -1, axis=0) @ Rp.T
        zeta_T_points_minus = zeta @ Tm.T
        zeta_T_points_plus = np.roll(zeta, -1, axis=1) @ Tp.T
        zeta_R_points = (zeta_R_points_minus + zeta_R_points_plus) / 2
        zeta_T_points = (zeta_T_points_minus + zeta_T_points_plus) / 2
        interp_R_leg = (alpha_R_points * zeta_R_points) @ P_inv.T
        interp_T_leg = (alpha_T_points * zeta_T_points) @ P_inv.T
        return interp_R_leg, interp_T_leg

    def upwind(zeta, H):
        alpha_R_points, alpha_T_points = H @ R.T, H @ T.T  # right, top
        zeta_R_points_minus = zeta @ Rm.T  # (nx, ny, order+1)
        zeta_R_points_plus = np.roll(zeta, -1, axis=0) @ Rp.T
        zeta_T_points_minus = zeta @ Tm.T
        zeta_T_points_plus = np.roll(zeta, -1, axis=1) @ Tp.T
        vals_R = (alpha_R_points > 0) * alpha_R_points * zeta_R_points_minus + (
            alpha_R_points <= 0
        ) * alpha_R_points * zeta_R_points_plus
        vals_T = (alpha_T_points > 0) * alpha_T_points * zeta_T_points_minus + (
            alpha_T_points <= 0
        ) * alpha_T_points * zeta_T_points_plus
        interp_R_leg = vals_R @ P_inv.T
        interp_T_leg = vals_T @ P_inv.T
        return interp_R_leg, interp_T_leg

    def vanleer(zeta, H):
        assert zeta.shape[-1] == 1

        alpha_R_points = H @ R.T
        zeta_R_points_minus = zeta
        zeta_R_points_plus = np.roll(zeta, -1, axis=0)

        s_R_right = np.roll(zeta, -1, axis=0) - zeta
        s_R_left = zeta - np.roll(zeta, 1, axis=0)
        s_R_centered = (s_R_right + s_R_left) / 2
        s_R_minus = minmod_3(2 * s_R_left, s_R_centered, 2 * s_R_right)
        s_R_plus = np.roll(s_R_minus, -1, axis=0)

        vals_R = (alpha_R_points > 0) * alpha_R_points * (zeta_R_points_minus + s_R_minus / 2) + (
            alpha_R_points <= 0
        ) * alpha_R_points * (zeta_R_points_plus - s_R_plus / 2)


        alpha_T_points = H @ T.T
        zeta_T_points_minus = zeta
        zeta_T_points_plus = np.roll(zeta, -1, axis=1)

        s_T_right = np.roll(zeta, -1, axis=1) - zeta
        s_T_left = zeta - np.roll(zeta, 1, axis=1)
        s_T_centered = (s_T_right + s_T_left) / 2
        s_T_minus = minmod_3(2 * s_T_left, s_T_centered, 2 * s_T_right)
        s_T_plus = np.roll(s_T_minus, -1, axis=1)

        vals_T = (alpha_T_points > 0) * alpha_T_points * (zeta_T_points_minus + s_T_minus / 2) + (
            alpha_T_points <= 0
        ) * alpha_T_points * (zeta_T_points_plus - s_T_plus / 2)


        return vals_R, vals_T

    if flux == Flux.CENTERED:
        interp_func = centered
    if flux == Flux.UPWIND:
        interp_func = upwind
    if flux == Flux.VANLEER:
        interp_func = vanleer

    def poisson_bracket(zeta, H):
        interp_R, interp_T = interp_func(zeta, H)
        interp_L = np.roll(interp_R, 1, axis=0)
        interp_B = np.roll(interp_T, 1, axis=1)
        boundary_term = (
            (interp_R @ CBN_R)
            + (interp_T @ CBN_T)
            - (interp_L @ CBN_L)
            - (interp_B @ CBN_B)
        )
        volume_term = np.einsum("ikl,xyk,xyl->xyi", V, zeta, H)
        return volume_term - boundary_term

    return poisson_bracket


###############
# (phi - n)
###############


def get_diff(order, dx, dy):
    fe_ip = leg_FE_inner_product(order) * dx * dy
    l_ip = legendre_inner_product(order) * dx * dy

    def diff_term(phi, n):
        phi_term = phi @ fe_ip.T
        n_term = l_ip[None, None, :] * n
        return phi_term - n_term

    return diff_term


###############
# kappa d phi / dy
###############


def get_deriv_y(order, dx, dy):
    """
    Computes \int phi_i dH/dy dx dy
    """
    Vd = deriv_y_leg_FE_inner_product(order) * dx
    Bt = leg_FE_top_integrate(order) * dx
    Bb = leg_FE_bottom_integrate(order) * dx
    alt_tb = get_topbottom_alternate(order)
    full_M = -Vd.T + Bt.T - Bb.T

    def deriv_term(phi):
        return phi @ full_M

    return deriv_term
