import jax.numpy as np
from jax import vmap, jit
from legendredg import generate_legendre
from functools import partial

vmap_polyval = vmap(np.polyval, (0, None), -1)


def _quad_two_per_interval(f, a, b, n=5):
    mid = (a + b) / 2
    return _fixed_quad(f, a, mid, n) + _fixed_quad(f, mid, b, n)


def _fixed_quad(f, a, b, n=5):
    """
    Single quadrature of a given order.

    Inputs
    f: function which takes a vector of positions of length n
    and returns a (possibly) multivariate output of length (n, p)
    a: beginning of integration
    b: end of integration
    n: order of quadrature. max n is 8.
    """
    assert isinstance(n, int) and n <= 10 and n > 0
    w = {
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
        9: np.asarray(
            [
                0.0812743883615744119718922,
                0.1806481606948574040584720,
                0.2606106964029354623187429,
                0.3123470770400028400686304,
                0.3302393550012597631645251,
                0.3123470770400028400686304,
                0.2606106964029354623187429,
                0.1806481606948574040584720,
                0.0812743883615744119718922,
            ]
        ),
        10: np.asarray(
            [
                0.0666713443086881375935688,
                0.1494513491505805931457763,
                0.2190863625159820439955349,
                0.2692667193099963550912269,
                0.2955242247147528701738930,
                0.2955242247147528701738930,
                0.2692667193099963550912269,
                0.2190863625159820439955349,
                0.1494513491505805931457763,
                0.0666713443086881375935688,
            ]
        ),
    }[n]

    xi_i = {
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
        9: np.asarray(
            [
                -0.9681602395076260898355762,
                -0.8360311073266357942994298,
                -0.6133714327005903973087020,
                -0.3242534234038089290385380,
                0.0,
                0.3242534234038089290385380,
                0.6133714327005903973087020,
                0.8360311073266357942994298,
                0.9681602395076260898355762,
            ]
        ),
        10: np.asarray(
            [
                -0.9739065285171717200779640,
                -0.8650633666889845107320967,
                -0.6794095682990244062343274,
                -0.4333953941292471907992659,
                -0.1488743389816312108848260,
                0.1488743389816312108848260,
                0.4333953941292471907992659,
                0.6794095682990244062343274,
                0.8650633666889845107320967,
                0.9739065285171717200779640,
            ]
        ),
    }[n]

    x_i = (b + a) / 2 + (b - a) / 2 * xi_i
    wprime = w * (b - a) / 2
    return np.sum(wprime[:, None] * f(x_i), axis=0)


def evalf_1D_right(a):
    """
    Returns the representation of f at the right end of
    each of the nx gridpoints

    Inputs
    a: (nx, p) ndarray

    Outputs
    f: (nx,) ndarray
    """
    return np.sum(a, axis=1)


def evalf_1D_left(a, p):
    """
    Returns the representation of f at the left end of
    each of the nx gridpoints

    Inputs
    a: (nx, p) ndarray

    Outputs
    f: (nx,) ndarray
    """
    negonetok = (np.ones(p) * -1) ** np.arange(p)
    return np.sum(a * negonetok, axis=1)


def evalf_1D(x, a, dx, leg_poly):
    """
    Returns the value of DG representation of the
    solution at x.

    Inputs:
    x: 1D array of points
    a: DG representation, (nx, p) ndarray

    Ouputs:
    f: 1d array of points, equal to sum over p polynomials
    """
    j = np.floor(x / dx).astype(int)
    # print("j is {}".format(j))
    x_j = dx * (0.5 + j)
    xi = (x - x_j) / (0.5 * dx)
    # print("xi is {}".format(xi))
    poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
    return np.sum(poly_eval * a[j, :], axis=-1)


def integrate_abs_single(a, poly_eval):
    """
    a is (p) array
    returns: scalar, integral
    """
    vals = poly_eval * a
    return np.mean(np.abs(np.sum(vals, axis=1)))


def MAE_abs(a):
    p = a.shape[1]
    leg_poly = generate_legendre(p)
    NP = 2 * p
    xi = np.linspace(-1, 1, NP + 1)[:-1] + 1 / NP
    poly_eval = vmap_polyval(leg_poly, xi)
    return np.mean(vmap(integrate_abs_single, (0, None), 0)(a, poly_eval))


def integrate_abs_deriv_single(a, poly_eval):
    """
    a is (p) array
    returns: scalar, integral
    """
    deriv = poly_eval * a
    NP = deriv.shape[0]
    return np.sum(np.abs(np.sum(deriv, axis=1))) / NP


def integrate_abs_derivative(a):
    p = a.shape[1]
    if p == 1:
        return 0.0
    leg_poly = generate_legendre(p)
    vmap_polyder = vmap(np.polyder, 0, 0)
    NP = 10
    xi = np.linspace(-1, 1, NP + 1)[:-1] + 1 / NP
    poly_eval = vmap_polyval(vmap_polyder(leg_poly), xi)
    return np.sum(vmap(integrate_abs_deriv_single, (0, None), 0)(a, poly_eval))


def map_f_to_DG(f, t, p, nx, dx, leg_poly, quad_func=_fixed_quad, n=5):
    """
    Takes a function f of type lambda x, t: f(x,t) and
    generates the DG representation of the solution, an
    array of size (nx, p).

    Computes the inner product of f with p Legendre polynomials
    over nx regions, to produce an array of size (nx, p)

    Inputs
    f: lambda x, t: f(x, t), the value of f
    t: the current time

    Outputs
    a0: The DG representation of f(x, t) at t=t
    """
    twokplusone = 2 * np.arange(0, p) + 1
    return (
        twokplusone[None, :]
        / dx
        * inner_prod_with_legendre(f, t, p, nx, dx, leg_poly, quad_func=quad_func, n=n)
    )


def inner_prod_with_legendre(f, t, p, nx, dx, leg_poly, quad_func=_fixed_quad, n=5):
    """
    Takes a function f of type lambda x, t: f(x,t) and
    takes the inner product of the solution with p
    legendre polynomials over all nx grid cells,
    resulting in an array of size (nx, p).

    Inputs
    f: lambda x, t: f(x, t), the value of f
    t: the current time
    leg_poly: the legendre coefficients

    Outputs
    integral: The inner product representation of f(x, t) at t=t
    """

    _vmap_fixed_quad = vmap(
        lambda f, a, b: quad_func(f, a, b, n=n), (None, 0, 0), 0
    )  # is n = p+1 high enough order?
    twokplusone = np.arange(p) * 2 + 1
    j = np.arange(nx)
    a = dx * j
    b = dx * (j + 1)

    def xi(x):
        j = np.floor(x / dx)
        x_j = dx * (0.5 + j)
        return (x - x_j) / (0.5 * dx)

    to_int_func = lambda x: f(x, t)[:, None] * vmap_polyval(leg_poly, xi(x))

    return _vmap_fixed_quad(to_int_func, a, b)


@partial(
    jit,
    static_argnums=(
        1,
        2,
    ),
)
def convert_DG_representation(a, p_new, nx_new, dx_new, dx_high, leg_poly_high):
    """
    # Converts one DG representation to another. Starts by writing a function
    # which does the mapping for a single timestep, then vmaps for many timesteps.

    # Inputs
    # a: (nt, nx, p_old), high-resolution DG representation
    # p_new: The order of the new representation
    # upsampling: Spatial upsampling of new resolution

    # Outputs
    # a_new: (N_run, nt, nx//upsampling, p_new), low-resolution DG representation
    """
    nx_high, p_high = a.shape
    if p_new == p_high and nx_new == nx_high:
        return a
    leg_poly_new = generate_legendre(p_new)

    def convert_repr(a):
        """
        Same function except a is (nx, p_old) and a_new is (nx//upsampling, p_new)
        """

        def f_high(x, t):
            res = evalf_1D(x, a, dx_high, leg_poly_high)
            return res

        a_pre = map_f_to_DG(
            f_high,
            0.0,
            p_new,
            nx_new,
            dx_new,
            leg_poly_new,
            quad_func=_quad_two_per_interval,
            n=10,
        )
        return a_pre

    return convert_repr(a)