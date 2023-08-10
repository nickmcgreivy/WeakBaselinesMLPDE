import jax.numpy as np
from basisfunctions import (
    legendre_npbasis,
    node_locations,
    legendre_inner_product,
)
from jax import vmap, jit
from poissonsolver import get_poisson_solve_fn_dg
from time import time

def _trapezoidal_integration(f, xi, xf, yi, yf, n=None):
	return (xf - xi) * (yf - yi) * (f(xi, yi) + f(xf, yi) + f(xi, yf) + f(xf, yf)) / 4


def _2d_fixed_quad(f, xi, xf, yi, yf, n=10):
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


def inner_prod_with_legendre(nx, ny, Lx, Ly, order, func, t, quad_func=_2d_fixed_quad, n=10):
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

def f_to_DG(nx, ny, Lx, Ly, order, func, t, quad_func=_2d_fixed_quad, n=10):
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



Lx = Ly = 2 * np.pi
order = 0
func = lambda x, y, t: np.sin(2 * np.pi * x / Lx) * np.sin(2 * np.pi * y / Lx)
t0 = 0.0
nxs = [32, 64, 128, 256]
N_test = 2

for nx in nxs:
	ny = nx

	dg_representation = f_to_DG(nx, ny, Lx, Ly, order, func, t0)
	solve = jit(get_poisson_solve_fn_dg(nx, ny, order, Lx, Ly))
	res = solve(dg_representation)

	t1 = time()
	for _ in range(N_test):
		res = solve(dg_representation).block_until_ready()
	t2 = time()
	ave_time = (t2 - t1) / N_test
	print("Time to solve Poisson equation using LU decomposition on {}x{} grid is {}".format(nx, ny, ave_time))



