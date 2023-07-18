import jax
import jax.numpy as jnp
from jax import vmap
import numpy as onp
from scipy.special import comb
from functools import lru_cache
from jax.lax import scan
from functools import partial
from jax import vmap, jit
from dataclasses import dataclass  # Needs Python 3.7 or higher
from jax.config import config
config.update("jax_enable_x64", True)
from matplotlib import cm


L = 1.0

@lru_cache(maxsize=10)
def generate_legendre(p):
	"""
	Returns a (p, p) array which represents
	p length-p polynomials. legendre_poly[k] gives
	an array which represents the kth Legendre
	polynomial. The polynomials are represented
	from highest degree of x (x^{p-1}) to lowest degree (x^0),
	as is standard in numpy poly1d.

	Inputs
	p: the number of Legendre polynomials

	Outputs
	poly: (p, p) array representing the Legendre polynomials
	"""
	assert p >= 1
	poly = onp.zeros((p, p))
	poly[0, -1] = 1.0
	twodpoly = onp.asarray([0.5, -0.5])
	for n in range(1, p):
		for k in range(n + 1):
			temp = onp.asarray([1.0])
			for j in range(k):
				temp = onp.polymul(temp, twodpoly)
			temp *= comb(n, k) * comb(n + k, k)
			poly[n] = onp.polyadd(poly[n], temp)

	return poly


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
	return jnp.sum(wprime[:, None] * f(x_i), axis=0)

vmap_polyval = vmap(jnp.polyval, (0, None), -1)

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
	j = jnp.floor(x / dx).astype(int)
	x_j = dx * (0.5 + j)
	xi = (x - x_j) / (0.5 * dx)
	poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
	return jnp.sum(poly_eval * a[j, :], axis=-1)



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
	twokplusone = jnp.arange(p) * 2 + 1
	j = jnp.arange(nx)
	a = dx * j
	b = dx * (j + 1)

	def xi(x):
		j = jnp.floor(x / dx)
		x_j = dx * (0.5 + j)
		return (x - x_j) / (0.5 * dx)

	to_int_func = lambda x: f(x, t)[:, None] * vmap_polyval(leg_poly, xi(x))

	return _vmap_fixed_quad(to_int_func, a, b)


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
	twokplusone = 2 * jnp.arange(0, p) + 1
	return (
		twokplusone[None, :]
		/ dx
		* inner_prod_with_legendre(f, t, p, nx, dx, leg_poly, quad_func=quad_func, n=n)
	)


def ssp_rk3(a_n, t_n, F, dt):
	a_1 = a_n + dt * F(a_n, t_n)
	a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * F(a_1, t_n))
	return 1 / 3 * a_n + 2 / 3 * (a_2 + dt * F(a_2, t_n)), t_n + dt



def _upwind_flux_DG_1D_advection(a, p):
	"""
	Computes the upwind flux F(f_{j+1/2}) where
	+ is given by right=True and - is given by right=False.
	For c > 0, F(f_{j+1/2}) = f_{j+1/2}^-
	where + = outside and - = inside.

	Inputs
	a: nx, p array
	right: Boolean, which side of flux is it

	Outputs
	F: (nx) array equal to f upwind.
	"""
	c = 1.0
	if c > 0:
		F = jnp.sum(a, axis=1)
	else:
		alt = (jnp.ones(p) * -1) ** jnp.arange(p)
		a = jnp.pad(a, ((0, 1), (0, 0)), "wrap")
		F = jnp.sum(alt[None, :] * a[1:, :], axis=1)
	return F

def minmod_3(z1, z2, z3):
    s = (
        0.5
        * (jnp.sign(z1) + jnp.sign(z2))
        * jnp.absolute(0.5 * ((jnp.sign(z1) + jnp.sign(z3))))
    )
    return s * jnp.minimum(jnp.absolute(z1), jnp.minimum(jnp.absolute(z2), jnp.absolute(z3)))

def _muscl_flux_DG_1D_advection(a):
    u = jnp.sum(a, axis=-1)
    du_j_minus = u - jnp.roll(u, 1)
    du_j_plus = jnp.roll(u, -1) - u
    du_j = minmod_3(du_j_minus, (du_j_plus + du_j_minus) / 4, du_j_plus)
    return u + du_j

def _flux_term_DG_1D_advection(a, t, p):
	negonetok = (jnp.ones(p) * -1) ** jnp.arange(p)
	if p == 1:
		flux_right = _muscl_flux_DG_1D_advection(a)
	else:
		flux_right = _upwind_flux_DG_1D_advection(a, p)
	flux_left = jnp.roll(flux_right, 1, axis=0)
	return negonetok[None, :] * flux_left[:, None] - flux_right[:, None]


def _volume_integral_DG_1D_advection(a, t):
	volume_sum = jnp.zeros(a.shape).at[:, 1::2].add(2 * jnp.cumsum(a[:, :-1:2], axis=1))
	return volume_sum.at[:, 2::2].add(2 * jnp.cumsum(a[:, 1:-1:2], axis=1))


def time_derivative_DG_1D_advection(a, t, p, dx):
	"""
	Compute da_j^m/dt given the matrix a_j^m which represents the solution,
	for a given flux. The time-derivative is given by a Galerkin minimization
	of the residual squared, with Legendre polynomial basis functions.
	For the 1D advection equation
			df/dt + c df/dx = 0
	with f_j = \sum a_j^m \phi_m, the time derivatives equal

	da_j^m/dt = ((2m+1)*c/deltax) [ (-1)^m F(f_{j-1/2}) - F(f_{j+1/2}) ]
							+ c(2m+1) (a_j^{m-1} + a_j^{m-3} + ...)

	Inputs
	a: (nx, p) array of coefficients
	t: time, scalar, not used here
	c: speed (scalar)
	flux: Enum, decides which flux will be used for the boundary

	Outputs
	da_j^m/dt: (nx, p) array of time derivatives
	"""
	twokplusone = 2 * jnp.arange(0, p) + 1
	flux_term = _flux_term_DG_1D_advection(a, t, p)
	volume_integral = _volume_integral_DG_1D_advection(a, t)
	c = 1.0
	return (c / dx) * twokplusone[None, :] * (flux_term + volume_integral)


def _scan(sol, x, rk_F):
	a, t = sol
	a_f, t_f = rk_F(a, t)
	return (a_f, t_f), None


def _scan_output(sol, x, rk_F):
	a, t = sol
	a_f, t_f = rk_F(a, t)
	return (a_f, t_f), a_f


def simulate_1D(a0, t0, p, nx, dx, dt, leg_poly, nt):
	dadt = lambda a, t: time_derivative_DG_1D_advection(a, t, p, dx)
	rk_F = lambda a, t: ssp_rk3(a, t, dadt, dt)

	scanf = jit(lambda sol, x: _scan_output(sol, x, rk_F))
	_, data = scan(scanf, (a0, t0), None, length=nt)
	return data




def plot_subfig(
	a, axs, color="blue", linewidth=0.5, linestyle="solid", label=None, lim=[-1, 1], ticks=None
):
	def evalf(x, a, j, dx, leg_poly):
		x_j = dx * (0.5 + j)
		xi = (x - x_j) / (0.5 * dx)
		vmap_polyval = vmap(jnp.polyval, (0, None), -1)
		poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
		return jnp.sum(poly_eval * a, axis=-1)

	nx = a.shape[0]
	p = a.shape[1]
	dx = L / nx
	xjs = jnp.arange(nx) * L / nx
	num_plot = 2 * p
	xs = xjs[None, :] + jnp.linspace(0.0, dx, num_plot)[:, None]
	vmap_eval = vmap(evalf, (1, 0, 0, None, None), 1)
	axs.plot(
		xs,
		vmap_eval(xs, a, jnp.arange(nx), dx, generate_legendre(p)),
		color=color,
		linewidth=linewidth,
		label=label,
		linestyle=linestyle,
	)
	axs.set_ylim(lim)
	axs.set_yticks(ticks)
	return


def plot_image(trajectory, fig, axs, lim=[-1, 1]):
	def evalf(x, a, j, dx, leg_poly):
		x_j = dx * (0.5 + j)
		xi = (x - x_j) / (0.5 * dx)
		vmap_polyval = vmap(jnp.polyval, (0, None), -1)
		poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
		return jnp.sum(poly_eval * a, axis=-1)

	nx = trajectory.shape[1]
	p = trajectory.shape[2]
	dx = L / nx
	xjs = jnp.arange(nx) * L / nx
	num_plot = 2 * p + 2
	dp = dx / (num_plot + 1)
	xs = xjs[None, :] + jnp.linspace(dp/2, dx - dp/2, num_plot)[:, None]
	vmap_eval = vmap(evalf, (1, 0, 0, None, None), 1)

	trajectory_highres = vmap(lambda a: jnp.abs(vmap_eval(xs, a, jnp.arange(nx), dx, generate_legendre(p))).T.reshape(-1))(trajectory)
	im = axs.imshow(jnp.abs(trajectory_highres), cmap='jet', vmin=lim[0], vmax=lim[1], origin='lower', extent=[0,1,0,1])
	
	plt.xlabel('x')
	plt.ylabel('t')

	plt.colorbar(im, ax=axs)




################
# Deep ONet hyperparams
################


def initial_condition(key):

	def f(x, t):
		return -0.95 + 0.45 * (0.75 * jnp.sin(2 * jnp.pi * (x-t) / L / 2)**2 - jnp.sin(2 * 2 * jnp.pi * (x-t) / L / 2)**2)#jnp.sin(2 * 2 * jnp.pi * (x - t) / L)# ((-jnp.sin(2 * jnp.pi * (x - t) / L / 2)**2 - jnp.sin(3 * 2 * jnp.pi * (x - t) / L / 2))**2) * 0.15

	return f



key = jax.random.PRNGKey(0)
f_init = initial_condition(key)
t0 = 0.0
T = 1.0

cfl_safety = 0.5
c = 1.0
p = 3
nx = 13
dx = L / nx
leg_poly = generate_legendre(p)
dt_i = cfl_safety * dx / (2 * p - 1) / c
nt = int(T / dt_i)
dt = T / nt


print("nt is {}".format(nt))

a0 = map_f_to_DG(f_init, t0, p, nx, dx, leg_poly, n=8)


##########
# compute runtime
##########


from time import time



f_sim = jit(lambda a0: simulate_1D(a0, t0, p, nx, dx, dt, leg_poly, nt))

_ = f_sim(a0).block_until_ready()

N_test = 100

total_time = 0.0

for n in range(N_test):
	t1 = time()
	_ = f_sim(a0).block_until_ready()
	t2 = time()
	total_time += t2 - t1

print("Average runtime is {}".format(total_time / N_test))








trajectory = f_sim(a0)
trajectory = jnp.concatenate((a0[None], trajectory), axis=0)


ts = jnp.linspace(0, T, nt + 1)

exact_trajectory = vmap(lambda t: map_f_to_DG(f_init, t, p, nx, dx, leg_poly))(ts)


diff = jnp.abs(trajectory - exact_trajectory)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,2)

plot_subfig(a0, axs[0],lim=[-1.3,-0.5], ticks=[-1.2, -1.0, -0.8, -0.6])

plot_image(diff, fig, axs[1], lim=[0.0,0.004])
axs[0].set_title("A similar IC")
axs[1].set_title("Error")
plt.show()







