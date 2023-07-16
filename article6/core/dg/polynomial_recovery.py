import jax.numpy as np
from sympy import legendre, diff, integrate, symbols
from functools import lru_cache

import numpy as onp
from scipy.special import eval_legendre
import matplotlib.pyplot as plt


def upper_B(m, k):
    x = symbols("x")
    expr = x ** k * (x + 0.5) ** m
    return integrate(expr, (x, -1, 0))


def lower_B(m, k):
    x = symbols("x")
    expr = x ** k * (x - 0.5) ** m
    return integrate(expr, (x, 0, 1))


def A(m, k):
    x = symbols("x")
    expr = legendre(k, x) * x ** m
    return integrate(expr, (x, -1, 1)) / (2 ** (m + 1))


@lru_cache(maxsize=10)
def get_B_matrix(p):
    res = onp.zeros((2 * p, 2 * p))
    for m in range(p):
        for k in range(2 * p):
            res[m, k] = upper_B(m, k)
    for m in range(p):
        for k in range(2 * p):
            res[m + p, k] = lower_B(m, k)
    return res


@lru_cache(maxsize=10)
def get_inverse_B(p):
    B = get_B_matrix(p)
    return onp.linalg.inv(B)


@lru_cache(maxsize=10)
def get_A_matrix(p):
    res = onp.zeros((2 * p, 2 * p))
    for m in range(p):
        for k in range(p):
            res[m, k] = A(m, k)
            res[m + p, k + p] = A(m, k)
    return res


def get_b_coefficients(a):
    """
    Inputs:
    a: (nx, p) array of coefficients

    Outputs:
    b: (nx, 2p) array of coefficients for the right boundary
    """
    p = a.shape[1]
    a_jplusone = np.roll(a, -1, axis=0)
    a_bar = np.concatenate((a, a_jplusone), axis=1)
    A = get_A_matrix(p)
    B_inv = get_inverse_B(p)
    rhs = np.einsum("ml,jl->jm", A, a_bar)
    b = np.einsum("km,jm->jk", B_inv, rhs)
    return b


def recovery_slope(a, p):
    """
    Inputs:
    a: (nx, p) array of coefficients

    Outputs:
    b: (nx,) array of slopes of recovery polynomial at right boundary
    """
    a_jplusone = np.roll(a, -1, axis=0)
    a_bar = np.concatenate((a, a_jplusone), axis=1)
    A = get_A_matrix(p)
    B_inv = get_inverse_B(p)[1, :]
    rhs = np.einsum("ml,jl->jm", A, a_bar)
    slope = np.einsum("m,jm->j", B_inv, rhs)
    return slope


# test recovery polynomial

"""
def eval_dg(a, x):
	res = onp.zeros(x.shape)
	for k in range(a.shape[0]):
		res += a[k] * eval_legendre(k, x)
	return res

def eval_recovery(b, x):
	res = onp.zeros(x.shape)
	for k in range(b.shape[0]):
		res += b[k] * (x**k)
	return res


p=3
a = onp.random.random((4,p))
b = get_b_coefficients(a)

N_plot = 200
x = np.linspace(-1, 1, N_plot)
x_plot_left = np.linspace(-1, 0, N_plot)
x_plot_right = np.linspace(0, 1, N_plot)
dg_left = eval_dg(a[0,:], x)
dg_right = eval_dg(a[1,:], x)

recovery = eval_recovery(b[0,:], x)
slope = recovery_slope(a, p)
print(slope[0])

plt.plot(x_plot_left, dg_left, color="green")
plt.plot(x_plot_right, dg_right, color="green")
plt.plot(x, recovery, color="blue")
plt.plot(x, b[0,0] + slope[0] * x, color="red")
plt.show()
"""
