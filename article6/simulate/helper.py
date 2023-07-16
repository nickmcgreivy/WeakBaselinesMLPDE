import jax.numpy as jnp
from jax import vmap, jit
from functools import partial

vmap_polyval = vmap(jnp.polyval, (0, None), -1)


def _fixed_quad(f, a, b, n=5):
    assert isinstance(n, int) and n <= 8 and n > 0
    w = {
        1: jnp.asarray([2.0]),
        2: jnp.asarray([1.0, 1.0]),
        3: jnp.asarray(
            [
                0.5555555555555555555556,
                0.8888888888888888888889,
                0.555555555555555555556,
            ]
        ),
        4: jnp.asarray(
            [
                0.3478548451374538573731,
                0.6521451548625461426269,
                0.6521451548625461426269,
                0.3478548451374538573731,
            ]
        ),
        5: jnp.asarray(
            [
                0.2369268850561890875143,
                0.4786286704993664680413,
                0.5688888888888888888889,
                0.4786286704993664680413,
                0.2369268850561890875143,
            ]
        ),
        6: jnp.asarray(
            [
                0.1713244923791703450403,
                0.3607615730481386075698,
                0.4679139345726910473899,
                0.4679139345726910473899,
                0.3607615730481386075698,
                0.1713244923791703450403,
            ]
        ),
        7: jnp.asarray(
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
        8: jnp.asarray(
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

    xi_i = {
        1: jnp.asarray([0.0]),
        2: jnp.asarray([-0.5773502691896257645092, 0.5773502691896257645092]),
        3: jnp.asarray([-0.7745966692414833770359, 0.0, 0.7745966692414833770359]),
        4: jnp.asarray(
            [
                -0.861136311594052575224,
                -0.3399810435848562648027,
                0.3399810435848562648027,
                0.861136311594052575224,
            ]
        ),
        5: jnp.asarray(
            [
                -0.9061798459386639927976,
                -0.5384693101056830910363,
                0.0,
                0.5384693101056830910363,
                0.9061798459386639927976,
            ]
        ),
        6: jnp.asarray(
            [
                -0.9324695142031520278123,
                -0.661209386466264513661,
                -0.2386191860831969086305,
                0.238619186083196908631,
                0.661209386466264513661,
                0.9324695142031520278123,
            ]
        ),
        7: jnp.asarray(
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
        8: jnp.asarray(
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

    x_i = (b + a) / 2 + (b - a) / 2 * xi_i
    wprime = w * (b - a) / 2
    return jnp.sum(wprime * f(x_i))


def map_f_to_FV(f, t, nx, dx, quad_func=_fixed_quad, n=5):
    return integrate_f(f, t, nx, dx, quad_func=quad_func, n=n) / dx


def integrate_f(f, t, nx, dx, quad_func=_fixed_quad, n=5):
    """
    Takes a function f of type lambda x, t: f(x,t) and
    takes the integral over all nx grid cells,
    resulting in an array of size (nx).

    Inputs
    f: lambda x, t: f(x, t), the value of f
    t: the current time

    Outputs
    integral: The inner product representation of f(x, t) at t=t
    """

    fn = lambda x: f(x, t)

    _vmap_fixed_quad = vmap(lambda a, b: quad_func(fn, a, b, n=n), (0, 0), 0)
    j = jnp.arange(nx)
    a = dx * j
    b = dx * (j + 1)

    return _vmap_fixed_quad(a, b)


@partial(
    jit,
    static_argnums=(1,),
)
def convert_FV_representation(a, nx_new, Lx):
    """
    Converts one FV representation to another. Starts by writing a function
    which does the mapping for a single grid cell, then vmaps for many grid cells.
    """
    nx_old = a.shape[0]
    if nx_old >= nx_new and nx_old % nx_new == 0:
        return jnp.mean(a.reshape(-1, nx_old // nx_new), axis=-1)

    else:
        raise Exception


def f_burgers(u):
    return u**2 / 2
