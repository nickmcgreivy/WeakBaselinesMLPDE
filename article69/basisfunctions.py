import sys
import sympy
import numpy as np
from sympy.matrices import Matrix, zeros
from sympy import Rational, symbols, legendre, integrate, diff
from functools import lru_cache
from scipy.special import legendre as scipy_legendre


def num_elements(order):
    """
    The number of elements in a 2D serendipity basis representation
    """
    if order == 0:
        return 1
    if order == 1:
        return 4
    elif order == 2:
        return 8
    elif order == 3:
        return 12
    elif order == 4:
        return 17
    else:
        raise NotImplementedError


def num_elements_FE(order):
    if order == 0:
        order = 1
    return num_elements(order)


@lru_cache(maxsize=4)
def monomials(order):
    """
    Returns the monomials for a given order,
    i.e. the powers of x and y for serendipity basis functions
    """
    x = zeros((order + 1) ** 2, 2)
    i = 0
    for p_x in range(order + 1):
        for p_y in range(order + 1):
            if p_x <= 1 or p_y <= 1 or p_x + p_y <= order:
                x[i, 0] = p_x
                x[i, 1] = p_y
                i += 1
    return x[0:i, :]


@lru_cache(maxsize=4)
def node_locations(order):
    """
    Returns the locations of nodes in a serendipity FE basis for orders 1-4
    """
    z = -1
    o = 1
    h = 0
    t = -Rational(1, 3)
    tw = Rational(1, 3)
    q = -Rational(1, 2)
    tq = Rational(1, 2)
    if order == 1 or order == 0:
        return Matrix([[z, z], [o, z], [o, o], [z, o]])
    elif order == 2:
        return Matrix([[z, z], [h, z], [o, z], [o, h], [o, o], [h, o], [z, o], [z, h]])
    elif order == 3:
        return Matrix(
            [
                [z, z],
                [t, z],
                [tw, z],
                [o, z],
                [o, t],
                [o, tw],
                [o, o],
                [tw, o],
                [t, o],
                [z, o],
                [z, tw],
                [z, t],
            ]
        )
    elif order == 4:
        return Matrix(
            [
                [z, z],
                [q, z],
                [h, z],
                [tq, z],
                [o, z],
                [o, q],
                [o, h],
                [o, tq],
                [o, o],
                [h, h],
                [tq, o],
                [h, o],
                [q, o],
                [z, o],
                [z, tq],
                [z, h],
                [z, q],
            ]
        )
    else:
        raise NotImplementedError


@lru_cache(maxsize=4)
def legendre_poly(order):
    """
    Returns the legendre polynomials
    for the serendipity basis for a given order
    in 2 dimensions
    """
    v = monomials(order)
    x = symbols("x")
    y = symbols("y")
    NumElements = v.shape[0]
    return Matrix(
        [
            legendre(int(v[k, 0]), x) * legendre(int(v[k, 1]), y)
            for k in range(NumElements)
        ]
    )


@lru_cache(maxsize=4)
def legendre_npbasis(order):
    """
    Returns an (order+1, N_e, 2) matrix which corresponds
    to the np.polyval representation of a polynomial
    """
    N_e = num_elements(order)
    basis = np.zeros((order + 1, N_e, 2))
    v = monomials(order)
    for k in range(N_e):
        o_x = int(v[k, 0])
        o_y = int(v[k, 1])
        basis[-(o_x + 1) :, k, 0] = scipy_legendre(o_x)
        basis[-(o_y + 1) :, k, 1] = scipy_legendre(o_y)
    return basis


@lru_cache(maxsize=4)
def legendre_boundary_poly(order, x):
    """
    Takes a sympy symbol (either x or y)
    and returns a 1D legendre polynomial of max degree order
    """
    return Matrix([legendre(k, x) for k in range(order + 1)])


@lru_cache(maxsize=4)
def FE_poly(order):
    """
    Returns the symbolic polynomials in x any y
    corresponding to the N serendipity basis elements for
    a given order between 1 and 4
    """
    if order == 0:
        order = 1

    def _eval_monomials(v, x):
        """
        Given a list of monomials (vector of powers of x and y),
        evaluate the monomial at x=(x,y)
        """
        res = zeros(1, v.shape[0])
        for j in range(v.shape[0]):
            res[j] = x[0] ** v[j, 0] * x[1] ** v[j, 1]
        return res

    def FE_basis_weights(order):
        """
        Find the weights which multiply each serendipity polynomial
        to get a value of 1 at each node.

        Eventually these weights will be multiplied by a symbol
        to get a polynomial
        """
        v = monomials(order)
        x_j = node_locations(order)
        NumElements = v.shape[0]
        weights = zeros(NumElements, NumElements)
        V = zeros(NumElements, NumElements)
        for j in range(NumElements):
            V[j, :] = _eval_monomials(v, x_j[j, :])
        V_inv = V.T.inv()
        for i in range(NumElements):
            N = zeros(NumElements, 1)
            N[i] = 1.0
            weights[:, i] = V_inv * N
        return weights

    v = monomials(order)
    weights = FE_basis_weights(order)
    NumElements = v.shape[0]
    x = symbols("x")
    y = symbols("y")
    vals = Matrix([x ** v[k, 0] * y ** v[k, 1] for k in range(NumElements)])
    return weights * vals


######### The below functions are used for the HW implementation
######### and are not basis functions, though they use symbolic
######### computation


@lru_cache(maxsize=4)
def legendre_inner_product(order):
    legendre_basis = legendre_poly(order)
    N = num_elements(order)
    inner_prod = np.zeros(N)
    for k in range(N):
        expr = legendre_basis[k] * legendre_basis[k]
        inner_prod[k] = integrate(expr, ("x", -1, 1), ("y", -1, 1)) / 4
    return inner_prod


@lru_cache(maxsize=4)
def legendre_boundary_inner_product(order):
    legendre_boundary_basis = legendre_boundary_poly(order, symbols("x"))
    N = order + 1
    inner_prod = np.zeros(N)
    for k in range(N):
        expr = legendre_boundary_basis[k] * legendre_boundary_basis[k]
        inner_prod[k] = integrate(expr, ("x", -1, 1)) / 2
    return inner_prod


@lru_cache(maxsize=4)
def leg_FE_inner_product(order):
    legendre_basis = legendre_poly(order)
    FE_basis = FE_poly(order)
    inner_prod_matrix = np.zeros((legendre_basis.shape[0], FE_basis.shape[0]))
    for i in range(inner_prod_matrix.shape[0]):
        for j in range(inner_prod_matrix.shape[1]):
            expr = legendre_basis[i] * FE_basis[j]
            inner_prod_matrix[i, j] = integrate(expr, ("x", -1, 1), ("y", -1, 1)) / 4
    return inner_prod_matrix


@lru_cache(maxsize=4)
def deriv_y_leg_FE_inner_product(order):
    legendre_basis = legendre_poly(order)
    FE_basis = FE_poly(order)
    inner_prod_matrix = np.zeros((legendre_basis.shape[0], FE_basis.shape[0]))
    for i in range(inner_prod_matrix.shape[0]):
        for j in range(inner_prod_matrix.shape[1]):
            expr = (
                diff(legendre_basis[i], "y") * FE_basis[j] * 2
            )  # ignoring a factor divide-by-dy here
            inner_prod_matrix[i, j] = integrate(expr, ("x", -1, 1), ("y", -1, 1)) / 4
    return inner_prod_matrix


@lru_cache(maxsize=4)
def leg_FE_top_integrate(order):
    legendre_basis = legendre_poly(order)
    FE_basis = FE_poly(order)
    inner_prod_matrix = np.zeros((legendre_basis.shape[0], FE_basis.shape[0]))
    for i in range(inner_prod_matrix.shape[0]):
        for j in range(inner_prod_matrix.shape[1]):
            expr = legendre_basis[i].subs("y", 1) * FE_basis[j].subs("y", 1)
            inner_prod_matrix[i, j] = integrate(expr, ("x", -1, 1)) / 2
    return inner_prod_matrix


@lru_cache(maxsize=4)
def leg_FE_bottom_integrate(order):
    legendre_basis = legendre_poly(order)
    FE_basis = FE_poly(order)
    inner_prod_matrix = np.zeros((legendre_basis.shape[0], FE_basis.shape[0]))
    for i in range(inner_prod_matrix.shape[0]):
        for j in range(inner_prod_matrix.shape[1]):
            expr = legendre_basis[i].subs("y", -1) * FE_basis[j].subs("y", -1)
            inner_prod_matrix[i, j] = integrate(expr, ("x", -1, 1)) / 2
    return inner_prod_matrix


@lru_cache(maxsize=4)
def create_poisson_bracket_volume_matrix(order):
    """
    V_ikl = int dphi_i/dz^alpha phi_k Pi^{alpha beta} dpsi_l/dz^beta dx dy
    """
    legendre_basis = legendre_poly(order)
    FE_basis = FE_poly(order)
    V = np.zeros((legendre_basis.shape[0], legendre_basis.shape[0], FE_basis.shape[0]))
    for i in range(V.shape[0]):
        for k in range(V.shape[1]):
            for l in range(V.shape[2]):
                expr = diff(legendre_basis[i], "x") * legendre_basis[k] * diff(
                    FE_basis[l], "y"
                ) - diff(legendre_basis[i], "y") * legendre_basis[k] * diff(
                    FE_basis[l], "x"
                )
                V[i, k, l] = integrate(
                    expr, ("x", -1, 1), ("y", -1, 1)
                )  # there are two hidden factors of 4 in expr and integrate
    return V


@lru_cache(maxsize=4)
def create_poisson_bracket_boundary_matrix_centered(order):
    """
    B_ikl = oint phi_i phi_k Pi^{alpha beta} dpsi_l/dz^beta n^{alpha} ds
    """
    legendre_basis = legendre_poly(order)
    legendre_right = legendre_basis.subs("x", 1)
    legendre_left = legendre_basis.subs("x", -1)
    legendre_top = legendre_basis.subs("y", 1)
    legendre_bottom = legendre_basis.subs("y", -1)
    FE_basis = FE_poly(order)
    B = np.zeros(
        (legendre_basis.shape[0], legendre_basis.shape[0], FE_basis.shape[0], 4)
    )
    for i in range(B.shape[0]):
        for k in range(B.shape[1]):
            for l in range(B.shape[2]):
                exprR = (
                    legendre_right[i]
                    * legendre_right[k]
                    * diff(FE_basis[l], "y").subs("x", 1)
                )
                exprT = (
                    legendre_top[i]
                    * legendre_top[k]
                    * (-diff(FE_basis[l], "x").subs("y", 1))
                )
                exprL = (
                    legendre_left[i]
                    * legendre_left[k]
                    * diff(FE_basis[l], "y").subs("x", -1)
                    * (-1)
                )
                exprB = (
                    legendre_bottom[i]
                    * legendre_bottom[k]
                    * (-diff(FE_basis[l], "x").subs("y", -1))
                    * (-1)
                )
                B[i, k, l, 0] = integrate(exprR, ("y", -1, 1))
                B[i, k, l, 1] = integrate(exprT, ("x", -1, 1))
                B[i, k, l, 2] = integrate(exprL, ("y", -1, 1))
                B[i, k, l, 3] = integrate(exprB, ("x", -1, 1))
    return B


@lru_cache(maxsize=4)
def create_poisson_bracket_boundary_matrix_upwind(order):
    """
    B_ij = oint phi_i P_l ds
    """
    legendre_basis = legendre_poly(order)
    x = symbols("x")
    y = symbols("y")
    legendre_boundary_basis_x = legendre_boundary_poly(order, x)
    legendre_boundary_basis_y = legendre_boundary_poly(order, y)
    B = np.zeros((legendre_basis.shape[0], legendre_boundary_basis_x.shape[0], 4))
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            exprR = legendre_basis[i].subs("x", 1) * legendre_boundary_basis_y[j]
            exprT = legendre_basis[i].subs("y", 1) * legendre_boundary_basis_x[j]
            exprL = legendre_basis[i].subs("x", -1) * legendre_boundary_basis_y[j]
            exprB = legendre_basis[i].subs("y", -1) * legendre_boundary_basis_x[j]
            B[i, j, 0] = integrate(exprR, ("y", -1, 1))
            B[i, j, 1] = integrate(exprT, ("x", -1, 1))
            B[i, j, 2] = integrate(exprL, ("y", -1, 1))
            B[i, j, 3] = integrate(exprB, ("x", -1, 1))
    return B


@lru_cache(maxsize=4)
def get_leftright_alternate(order):
    leg_poly = legendre_poly(order)
    return np.asarray(leg_poly.subs("x", -1).subs("y", 1), dtype=int)[:, 0]


@lru_cache(maxsize=4)
def get_topbottom_alternate(order):
    leg_poly = legendre_poly(order)
    return np.asarray(leg_poly.subs("y", -1).subs("x", 1), dtype=int)[:, 0]


@lru_cache(maxsize=4)
def interpolation_points(order):
    if order == 0:
        return np.asarray([0.0])
    if order == 1:
        w2 = 1 / np.sqrt(3)
        w1 = -w2
        return np.asarray([w1, w2])
    elif order == 2:
        w1, w2, w3 = -np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)
        return np.asarray([w1, w2, w3])
    elif order == 3:
        w3, w4 = np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)), np.sqrt(
            3 / 7 + 2 / 7 * np.sqrt(6 / 5)
        )
        w1, w2 = -w4, -w3
        return np.asarray([w1, w2, w3, w4])
    elif order == 4:
        w3 = 0.0
        w4, w5 = 1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)), 1 / 3 * np.sqrt(
            5 + 2 * np.sqrt(10 / 7)
        )
        w1, w2 = -w5, -w4
        return np.asarray([w1, w2, w3, w4, w5])
    else:
        raise NotImplementedError


@lru_cache(maxsize=4)
def interpolation_points_twice(order):
    if order == 0:
        return np.asarray([0.0])
    if order == 1:
        w1, w2, w3 = -np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)
        return np.asarray([w1, w2, w3])
    elif order == 2:
        w3 = 0.0
        w4, w5 = 1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)), 1 / 3 * np.sqrt(
            5 + 2 * np.sqrt(10 / 7)
        )
        w1, w2 = -w5, -w4
        return np.asarray([w1, w2, w3, w4, w5])
    elif order == 3:
        w1 = -0.9491079123427585245262
        w2 = -0.7415311855993944398639
        w3 = -0.4058451513773971669066
        w4 = 0.0
        w5, w6, w7 = -w3, -w2, -w1
        return np.asarray([w1, w2, w3, w4, w5, w6, w7])
    elif order == 4:
        w1 = -0.9681602395076260898356
        w2 = -0.8360311073266357942994
        w3 = -0.6133714327005903973087
        w4 = -0.3242534234038089290385
        w5 = 0.0
        w6, w7, w8, w9 = -w4, -w3, -w2, -w1
        return np.asarray([w1, w2, w3, w4, w5, w6, w7, w8, w9])
    else:
        raise NotImplementedError


@lru_cache(maxsize=4)
def boundary_matrix(order):
    P = np.zeros((order + 1, order + 1))
    points = interpolation_points(order)
    x = symbols("x")
    legendre_boundary_basis = legendre_boundary_poly(order, x)
    for i, p in enumerate(points):
        P[i, :, None] = legendre_boundary_basis.subs("x", p)
    return P


@lru_cache(maxsize=4)
def boundary_matrix_twice(order):
    P = np.zeros((2 * order + 1, 2 * order + 1))
    points = interpolation_points_twice(order)
    x = symbols("x")
    legendre_boundary_basis = legendre_boundary_poly(2 * order, x)
    for i, point in enumerate(points):
        P[i, :, None] = legendre_boundary_basis.subs("x", point)
    return P


@lru_cache(maxsize=4)
def boundary_matrix_inverse(order):
    P = boundary_matrix(order)
    return np.linalg.inv(P)


@lru_cache(maxsize=4)
def boundary_matrix_inverse_twice(order):
    P = boundary_matrix_twice(order)
    return np.linalg.inv(P)


@lru_cache(maxsize=4)
def alpha_right_matrix(order):
    FE_basis = FE_poly(order)
    points = interpolation_points(order)
    R = np.zeros((order + 1, FE_basis.shape[0]))
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i, j] = diff(FE_basis[j], "y").subs("x", 1).subs("y", points[i])
    return R


@lru_cache(maxsize=4)
def alpha_right_matrix_twice(order):
    FE_basis = FE_poly(order)
    points = interpolation_points_twice(order)
    R = np.zeros((2 * order + 1, FE_basis.shape[0]))
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i, j] = diff(FE_basis[j], "y").subs("x", 1).subs("y", points[i]) * 2
    return R


@lru_cache(maxsize=4)
def alpha_top_matrix(order):
    FE_basis = FE_poly(order)
    points = interpolation_points(order)
    T = np.zeros((order + 1, FE_basis.shape[0]))
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            T[i, j] = -diff(FE_basis[j], "x").subs("y", 1).subs("x", points[i])
    return T


@lru_cache(maxsize=4)
def alpha_top_matrix_twice(order):
    FE_basis = FE_poly(order)
    points = interpolation_points_twice(order)
    T = np.zeros((2 * order + 1, FE_basis.shape[0]))
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            T[i, j] = -diff(FE_basis[j], "x").subs("y", 1).subs("x", points[i]) * 2
    return T


@lru_cache(maxsize=4)
def zeta_right_minus_matrix(order):
    leg_basis = legendre_poly(order)
    points = interpolation_points(order)
    Rm = np.zeros((order + 1, leg_basis.shape[0]))
    for i in range(Rm.shape[0]):
        for j in range(Rm.shape[1]):
            Rm[i, j] = leg_basis[j].subs("x", 1).subs("y", points[i])
    return Rm


@lru_cache(maxsize=4)
def zeta_right_plus_matrix(order):
    leg_basis = legendre_poly(order)
    points = interpolation_points(order)
    Rp = np.zeros((order + 1, leg_basis.shape[0]))
    for i in range(Rp.shape[0]):
        for j in range(Rp.shape[1]):
            Rp[i, j] = leg_basis[j].subs("x", -1).subs("y", points[i])
    return Rp


@lru_cache(maxsize=4)
def zeta_top_minus_matrix(order):
    leg_basis = legendre_poly(order)
    points = interpolation_points(order)
    Tm = np.zeros((order + 1, leg_basis.shape[0]))
    for i in range(Tm.shape[0]):
        for j in range(Tm.shape[1]):
            Tm[i, j] = leg_basis[j].subs("y", 1).subs("x", points[i])
    return Tm


@lru_cache(maxsize=4)
def zeta_top_plus_matrix(order):
    leg_basis = legendre_poly(order)
    points = interpolation_points(order)
    Tp = np.zeros((order + 1, leg_basis.shape[0]))
    for i in range(Tp.shape[0]):
        for j in range(Tp.shape[1]):
            Tp[i, j] = leg_basis[j].subs("y", -1).subs("x", points[i])
    return Tp


@lru_cache(maxsize=4)
def zeta_right_minus_matrix_twice(order):
    leg_basis = legendre_poly(order)
    points = interpolation_points_twice(order)
    Rm = np.zeros((2 * order + 1, leg_basis.shape[0]))
    for i in range(Rm.shape[0]):
        for j in range(Rm.shape[1]):
            Rm[i, j] = leg_basis[j].subs("x", 1).subs("y", points[i])
    return Rm


@lru_cache(maxsize=4)
def zeta_right_plus_matrix_twice(order):
    leg_basis = legendre_poly(order)
    points = interpolation_points_twice(order)
    Rp = np.zeros((2 * order + 1, leg_basis.shape[0]))
    for i in range(Rp.shape[0]):
        for j in range(Rp.shape[1]):
            Rp[i, j] = leg_basis[j].subs("x", -1).subs("y", points[i])
    return Rp


@lru_cache(maxsize=4)
def zeta_top_minus_matrix_twice(order):
    leg_basis = legendre_poly(order)
    points = interpolation_points_twice(order)
    Tm = np.zeros((2 * order + 1, leg_basis.shape[0]))
    for i in range(Tm.shape[0]):
        for j in range(Tm.shape[1]):
            Tm[i, j] = leg_basis[j].subs("y", 1).subs("x", points[i])
    return Tm


@lru_cache(maxsize=4)
def zeta_top_plus_matrix_twice(order):
    leg_basis = legendre_poly(order)
    points = interpolation_points_twice(order)
    Tp = np.zeros((2 * order + 1, leg_basis.shape[0]))
    for i in range(Tp.shape[0]):
        for j in range(Tp.shape[1]):
            Tp[i, j] = leg_basis[j].subs("y", -1).subs("x", points[i])
    return Tp


@lru_cache(maxsize=4)
def change_basis_boundary_to_volume(order):
    """
    Returns 4 arrays of size (order+1, num_elements(order)), in order: right, top, left, bottom

    The goal is to rewrite a boundary-legendre basis in volume-legendre basis.

    For example, if I have order=1 then my arrays are of size (2, 4). The right array...
    """
    leg_basis = legendre_poly(order)
    leg_boundary_basis_x = legendre_boundary_poly(order, symbols("x"))
    leg_boundary_basis_y = legendre_boundary_poly(order, symbols("y"))
    right = np.zeros((order + 1, num_elements(order)))
    top = np.zeros((order + 1, num_elements(order)))
    left = np.zeros((order + 1, num_elements(order)))
    bottom = np.zeros((order + 1, num_elements(order)))
    for i in range(order + 1):
        for j in range(num_elements(order)):
            exprR = leg_boundary_basis_y[i] * leg_basis[j].subs("x", 1)
            exprT = leg_boundary_basis_x[i] * leg_basis[j].subs("y", 1)
            exprL = leg_boundary_basis_y[i] * leg_basis[j].subs("x", -1)
            exprB = leg_boundary_basis_x[i] * leg_basis[j].subs("y", -1)
            right[i, j] = (
                integrate(exprR, ("y", -1, 1))
                / 2
                / legendre_boundary_inner_product(order)[i]
            )
            top[i, j] = (
                integrate(exprT, ("x", -1, 1))
                / 2
                / legendre_boundary_inner_product(order)[i]
            )
            left[i, j] = (
                integrate(exprL, ("y", -1, 1))
                / 2
                / legendre_boundary_inner_product(order)[i]
            )
            bottom[i, j] = (
                integrate(exprB, ("x", -1, 1))
                / 2
                / legendre_boundary_inner_product(order)[i]
            )
    return right, top, left, bottom


@lru_cache(maxsize=4)
def change_legendre_points_twice(order):
    leg_basis = legendre_poly(order)
    leg_boundary_basis = legendre_boundary_poly(order, symbols("x"))
    points = interpolation_points_twice(order)
    LP = np.zeros((order + 1, len(points)))
    for i in range(order + 1):
        for j in range(len(points)):
            LP[i, j] = leg_boundary_basis[i].subs("x", points[j])
    return LP
