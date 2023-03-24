###########################################
# Invariance of Poissons Equation to Resolution
###########################################


import sys
import sympy
from sympy.matrices import Matrix, zeros
from sympy import Rational, symbols, legendre, integrate, diff, integrate, nsimplify, lambdify
from scipy.linalg import lu_solve as np_lu_solve
from scipy.sparse import csr_matrix, csc_matrix, dok_matrix
from scipy.sparse.linalg import splu
from scipy import sparse
from scipy.special import legendre as scipy_legendre

import numpy as onp
import jax.numpy as jnp
import jax.random as random
import jax
from jax import lax, core, config, xla, jit, grad, vmap
from jax._src import api, abstract_arrays
from jax.interpreters import ad
from jax.scipy.linalg import lu_solve
from jaxlib import xla_client


from functools import lru_cache, partial
import os.path
import os
from jax.experimental import sparse as jsparse


from jax.scipy.linalg import lu_factor
import timeit
import time
from jax import config, jit



import matplotlib
import matplotlib.pyplot as plt
import jax
import h5py
import seaborn as sns



config.update("jax_enable_x64", True)



###########################################
# Basis functions
###########################################


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
	to the onp.polyval representation of a polynomial
	"""
	N_e = num_elements(order)
	basis = onp.zeros((order + 1, N_e, 2))
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
	inner_prod = onp.zeros(N)
	for k in range(N):
		expr = legendre_basis[k] * legendre_basis[k]
		inner_prod[k] = integrate(expr, ("x", -1, 1), ("y", -1, 1)) / 4
	return inner_prod


@lru_cache(maxsize=4)
def legendre_boundary_inner_product(order):
	legendre_boundary_basis = legendre_boundary_poly(order, symbols("x"))
	N = order + 1
	inner_prod = onp.zeros(N)
	for k in range(N):
		expr = legendre_boundary_basis[k] * legendre_boundary_basis[k]
		inner_prod[k] = integrate(expr, ("x", -1, 1)) / 2
	return inner_prod


@lru_cache(maxsize=4)
def leg_FE_inner_product(order):
	legendre_basis = legendre_poly(order)
	FE_basis = FE_poly(order)
	inner_prod_matrix = onp.zeros((legendre_basis.shape[0], FE_basis.shape[0]))
	for i in range(inner_prod_matrix.shape[0]):
		for j in range(inner_prod_matrix.shape[1]):
			expr = legendre_basis[i] * FE_basis[j]
			inner_prod_matrix[i, j] = integrate(expr, ("x", -1, 1), ("y", -1, 1)) / 4
	return inner_prod_matrix


@lru_cache(maxsize=4)
def deriv_y_leg_FE_inner_product(order):
	legendre_basis = legendre_poly(order)
	FE_basis = FE_poly(order)
	inner_prod_matrix = onp.zeros((legendre_basis.shape[0], FE_basis.shape[0]))
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
	inner_prod_matrix = onp.zeros((legendre_basis.shape[0], FE_basis.shape[0]))
	for i in range(inner_prod_matrix.shape[0]):
		for j in range(inner_prod_matrix.shape[1]):
			expr = legendre_basis[i].subs("y", 1) * FE_basis[j].subs("y", 1)
			inner_prod_matrix[i, j] = integrate(expr, ("x", -1, 1)) / 2
	return inner_prod_matrix


@lru_cache(maxsize=4)
def leg_FE_bottom_integrate(order):
	legendre_basis = legendre_poly(order)
	FE_basis = FE_poly(order)
	inner_prod_matrix = onp.zeros((legendre_basis.shape[0], FE_basis.shape[0]))
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
	V = onp.zeros((legendre_basis.shape[0], legendre_basis.shape[0], FE_basis.shape[0]))
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
	B = onp.zeros(
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
	B = onp.zeros((legendre_basis.shape[0], legendre_boundary_basis_x.shape[0], 4))
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
	return onp.asarray(leg_poly.subs("x", -1).subs("y", 1), dtype=int)[:, 0]


@lru_cache(maxsize=4)
def get_topbottom_alternate(order):
	leg_poly = legendre_poly(order)
	return onp.asarray(leg_poly.subs("y", -1).subs("x", 1), dtype=int)[:, 0]


@lru_cache(maxsize=4)
def interpolation_points(order):
	if order == 0:
		return onp.asarray([0.0])
	if order == 1:
		w2 = 1 / onp.sqrt(3)
		w1 = -w2
		return onp.asarray([w1, w2])
	elif order == 2:
		w1, w2, w3 = -onp.sqrt(3 / 5), 0.0, onp.sqrt(3 / 5)
		return onp.asarray([w1, w2, w3])
	elif order == 3:
		w3, w4 = onp.sqrt(3 / 7 - 2 / 7 * onp.sqrt(6 / 5)), onp.sqrt(
			3 / 7 + 2 / 7 * onp.sqrt(6 / 5)
		)
		w1, w2 = -w4, -w3
		return onp.asarray([w1, w2, w3, w4])
	elif order == 4:
		w3 = 0.0
		w4, w5 = 1 / 3 * onp.sqrt(5 - 2 * onp.sqrt(10 / 7)), 1 / 3 * onp.sqrt(
			5 + 2 * onp.sqrt(10 / 7)
		)
		w1, w2 = -w5, -w4
		return onp.asarray([w1, w2, w3, w4, w5])
	else:
		raise NotImplementedError


@lru_cache(maxsize=4)
def interpolation_points_twice(order):
	if order == 0:
		return onp.asarray([0.0])
	if order == 1:
		w1, w2, w3 = -onp.sqrt(3 / 5), 0.0, onp.sqrt(3 / 5)
		return onp.asarray([w1, w2, w3])
	elif order == 2:
		w3 = 0.0
		w4, w5 = 1 / 3 * onp.sqrt(5 - 2 * onp.sqrt(10 / 7)), 1 / 3 * onp.sqrt(
			5 + 2 * onp.sqrt(10 / 7)
		)
		w1, w2 = -w5, -w4
		return onp.asarray([w1, w2, w3, w4, w5])
	elif order == 3:
		w1 = -0.9491079123427585245262
		w2 = -0.7415311855993944398639
		w3 = -0.4058451513773971669066
		w4 = 0.0
		w5, w6, w7 = -w3, -w2, -w1
		return onp.asarray([w1, w2, w3, w4, w5, w6, w7])
	elif order == 4:
		w1 = -0.9681602395076260898356
		w2 = -0.8360311073266357942994
		w3 = -0.6133714327005903973087
		w4 = -0.3242534234038089290385
		w5 = 0.0
		w6, w7, w8, w9 = -w4, -w3, -w2, -w1
		return onp.asarray([w1, w2, w3, w4, w5, w6, w7, w8, w9])
	else:
		raise NotImplementedError


@lru_cache(maxsize=4)
def boundary_matrix(order):
	P = onp.zeros((order + 1, order + 1))
	points = interpolation_points(order)
	x = symbols("x")
	legendre_boundary_basis = legendre_boundary_poly(order, x)
	for i, p in enumerate(points):
		P[i, :, None] = legendre_boundary_basis.subs("x", p)
	return P


@lru_cache(maxsize=4)
def boundary_matrix_twice(order):
	P = onp.zeros((2 * order + 1, 2 * order + 1))
	points = interpolation_points_twice(order)
	x = symbols("x")
	legendre_boundary_basis = legendre_boundary_poly(2 * order, x)
	for i, point in enumerate(points):
		P[i, :, None] = legendre_boundary_basis.subs("x", point)
	return P


@lru_cache(maxsize=4)
def boundary_matrix_inverse(order):
	P = boundary_matrix(order)
	return onp.linalg.inv(P)


@lru_cache(maxsize=4)
def boundary_matrix_inverse_twice(order):
	P = boundary_matrix_twice(order)
	return onp.linalg.inv(P)


@lru_cache(maxsize=4)
def alpha_right_matrix(order):
	FE_basis = FE_poly(order)
	points = interpolation_points(order)
	R = onp.zeros((order + 1, FE_basis.shape[0]))
	for i in range(R.shape[0]):
		for j in range(R.shape[1]):
			R[i, j] = diff(FE_basis[j], "y").subs("x", 1).subs("y", points[i])
	return R


@lru_cache(maxsize=4)
def alpha_right_matrix_twice(order):
	FE_basis = FE_poly(order)
	points = interpolation_points_twice(order)
	R = onp.zeros((2 * order + 1, FE_basis.shape[0]))
	for i in range(R.shape[0]):
		for j in range(R.shape[1]):
			R[i, j] = diff(FE_basis[j], "y").subs("x", 1).subs("y", points[i]) * 2
	return R


@lru_cache(maxsize=4)
def alpha_top_matrix(order):
	FE_basis = FE_poly(order)
	points = interpolation_points(order)
	T = onp.zeros((order + 1, FE_basis.shape[0]))
	for i in range(T.shape[0]):
		for j in range(T.shape[1]):
			T[i, j] = -diff(FE_basis[j], "x").subs("y", 1).subs("x", points[i])
	return T


@lru_cache(maxsize=4)
def alpha_top_matrix_twice(order):
	FE_basis = FE_poly(order)
	points = interpolation_points_twice(order)
	T = onp.zeros((2 * order + 1, FE_basis.shape[0]))
	for i in range(T.shape[0]):
		for j in range(T.shape[1]):
			T[i, j] = -diff(FE_basis[j], "x").subs("y", 1).subs("x", points[i]) * 2
	return T


@lru_cache(maxsize=4)
def zeta_right_minus_matrix(order):
	leg_basis = legendre_poly(order)
	points = interpolation_points(order)
	Rm = onp.zeros((order + 1, leg_basis.shape[0]))
	for i in range(Rm.shape[0]):
		for j in range(Rm.shape[1]):
			Rm[i, j] = leg_basis[j].subs("x", 1).subs("y", points[i])
	return Rm


@lru_cache(maxsize=4)
def zeta_right_plus_matrix(order):
	leg_basis = legendre_poly(order)
	points = interpolation_points(order)
	Rp = onp.zeros((order + 1, leg_basis.shape[0]))
	for i in range(Rp.shape[0]):
		for j in range(Rp.shape[1]):
			Rp[i, j] = leg_basis[j].subs("x", -1).subs("y", points[i])
	return Rp


@lru_cache(maxsize=4)
def zeta_top_minus_matrix(order):
	leg_basis = legendre_poly(order)
	points = interpolation_points(order)
	Tm = onp.zeros((order + 1, leg_basis.shape[0]))
	for i in range(Tm.shape[0]):
		for j in range(Tm.shape[1]):
			Tm[i, j] = leg_basis[j].subs("y", 1).subs("x", points[i])
	return Tm


@lru_cache(maxsize=4)
def zeta_top_plus_matrix(order):
	leg_basis = legendre_poly(order)
	points = interpolation_points(order)
	Tp = onp.zeros((order + 1, leg_basis.shape[0]))
	for i in range(Tp.shape[0]):
		for j in range(Tp.shape[1]):
			Tp[i, j] = leg_basis[j].subs("y", -1).subs("x", points[i])
	return Tp


@lru_cache(maxsize=4)
def zeta_right_minus_matrix_twice(order):
	leg_basis = legendre_poly(order)
	points = interpolation_points_twice(order)
	Rm = onp.zeros((2 * order + 1, leg_basis.shape[0]))
	for i in range(Rm.shape[0]):
		for j in range(Rm.shape[1]):
			Rm[i, j] = leg_basis[j].subs("x", 1).subs("y", points[i])
	return Rm


@lru_cache(maxsize=4)
def zeta_right_plus_matrix_twice(order):
	leg_basis = legendre_poly(order)
	points = interpolation_points_twice(order)
	Rp = onp.zeros((2 * order + 1, leg_basis.shape[0]))
	for i in range(Rp.shape[0]):
		for j in range(Rp.shape[1]):
			Rp[i, j] = leg_basis[j].subs("x", -1).subs("y", points[i])
	return Rp


@lru_cache(maxsize=4)
def zeta_top_minus_matrix_twice(order):
	leg_basis = legendre_poly(order)
	points = interpolation_points_twice(order)
	Tm = onp.zeros((2 * order + 1, leg_basis.shape[0]))
	for i in range(Tm.shape[0]):
		for j in range(Tm.shape[1]):
			Tm[i, j] = leg_basis[j].subs("y", 1).subs("x", points[i])
	return Tm


@lru_cache(maxsize=4)
def zeta_top_plus_matrix_twice(order):
	leg_basis = legendre_poly(order)
	points = interpolation_points_twice(order)
	Tp = onp.zeros((2 * order + 1, leg_basis.shape[0]))
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
	right = onp.zeros((order + 1, num_elements(order)))
	top = onp.zeros((order + 1, num_elements(order)))
	left = onp.zeros((order + 1, num_elements(order)))
	bottom = onp.zeros((order + 1, num_elements(order)))
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
	LP = onp.zeros((order + 1, len(points)))
	for i in range(order + 1):
		for j in range(len(points)):
			LP[i, j] = leg_boundary_basis[i].subs("x", points[j])
	return LP


###########################################
# Sparse solve
###########################################



sparse_solve_p = core.Primitive("sparse_solve")



def sparse_solve_prim(b, sparse_data, sparse_indices, size, forward=True):
	return sparse_solve_p.bind(b, sparse_data, sparse_indices, size, forward)


def sparse_solve_impl(b, sparse_data, sparse_indices, size, forward=True):
	raise Exception("Sparse solve prim shouldn't be called except from within JIT")


def sparse_solve_abstract_eval(b, sparse_data, sparse_indices, size, forward=True):
	return abstract_arrays.ShapedArray(b.shape, b.dtype)



def sparse_solve_value_and_jvp(primals, tangents):
	(b, sparse_data, sparse_indices, size, forward) = primals
	(bt, _, _, _, _) = tangents
	primal_out = sparse_solve_prim(
		b, sparse_data, sparse_indices, size, forward=forward
	)
	output_tangent = sparse_solve_prim(
		bt, sparse_data, sparse_indices, size, forward=forward
	)
	return (primal_out, output_tangent)


def sparse_solve_transpose(ct, b, sparse_data, sparse_indices, size, forward=True):
	return (
		sparse_solve_prim(-ct, sparse_data, sparse_indices, size, forward=not forward),
		None,
		None,
		None,
		None,
	)



import custom_call_sparse_solve

for _name, _value in custom_call_sparse_solve.registrations().items():
	xla_client.register_cpu_custom_call_target(_name, _value)


def sparse_solve_xla_translation(c, bc, sparse_data, sparse_indices, size, forward):
	bc_shape = c.get_shape(bc)
	bc_dtype = bc_shape.element_type()
	bc_dims = bc_shape.dimensions()
	bc_shape = xla_client.Shape.array_shape(jnp.dtype(bc_dtype), bc_dims, (0,))
	out_shape = xla_client.Shape.array_shape(jnp.dtype(bc_dtype), bc_dims, (0,))
	data_shape = c.get_shape(sparse_data)
	data_dtype = data_shape.element_type()
	data_dims = data_shape.dimensions()
	data_shape = xla_client.Shape.array_shape(jnp.dtype(data_dtype), data_dims, (0,))
	indices_shape = c.get_shape(sparse_indices)
	indices_dtype = indices_shape.element_type()
	indices_dims = indices_shape.dimensions()
	indices_shape = xla_client.Shape.array_shape(
		jnp.dtype(indices_dtype), indices_dims, (0, 1)
	)

	assert bc_dtype == data_dtype

	if bc_dtype == jnp.float32:
		op_name = b"sparse_solve_f32"
	elif bc_dtype == jnp.float64:
		op_name = b"sparse_solve_f64"
	else:
		raise NotImplementedError(f"Unsupported dtype {bc_dtype}")
	return xla_client.ops.CustomCallWithLayout(
		c,
		op_name,
		operands=(
			bc,
			sparse_data,
			sparse_indices,
			size,
			xla_client.ops.ConstantLiteral(c, data_dims[0]),
			forward,
		),
		shape_with_layout=out_shape,
		operand_shapes_with_layout=(
			bc_shape,
			data_shape,
			indices_shape,
			xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
			xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
			xla_client.Shape.array_shape(jnp.dtype(bool), (), ()),
		),
	)


from jax.interpreters import batching


def sparse_solve_batch(vector_arg_values, batch_axes):
	bs, sparse_data, sparse_indices, size, forward = vector_arg_values
	args = sparse_data, sparse_indices, size, forward
	assert batch_axes[0] == 0
	assert batch_axes[1] == None
	assert batch_axes[2] == None
	assert batch_axes[3] == None
	assert batch_axes[4] == None
	res = jax.lax.map(lambda b: sparse_solve_prim(b, *args), bs)
	return res, batch_axes[0]


xla.backend_specific_translations["cpu"][sparse_solve_p] = sparse_solve_xla_translation
sparse_solve_p.def_impl(sparse_solve_impl)
sparse_solve_p.def_abstract_eval(sparse_solve_abstract_eval)
ad.primitive_jvps[sparse_solve_p] = sparse_solve_value_and_jvp
ad.primitive_transposes[sparse_solve_p] = sparse_solve_transpose
batching.primitive_batchers[sparse_solve_p] = sparse_solve_batch






###########################################
# Basis functions
###########################################


def get_bottom_indices(order):
	if order == 1 or order == 0:
		return jnp.asarray([0], dtype=int)
	if order == 2:
		return jnp.asarray([0, 1, 7], dtype=int)
	if order == 3:
		return jnp.asarray([0, 1, 2, 10, 11], dtype=int)
	if order == 4:
		return jnp.asarray([0, 1, 2, 3, 9, 14, 15, 16], dtype=int)
	raise Exception


def is_bottom_element(order, k):
	arr = get_bottom_indices(order)
	if order == 1 or order == 0:
		if k in arr:
			return True
	elif order == 2:
		if k in arr:
			return True
	elif order == 3:
		if k in arr:
			return True
	elif order == 4:
		if k in arr:
			return True
	else:
		raise Exception
	return False


def convert_to_bottom_indices(T, order):
	def convert_to_bottom_index(index):
		if order == 1 or order == 0:
			if index == 0:
				return 0
			else:
				raise Exception
		if order == 2:
			if index == 0 or index == 1:
				return index
			if index == 7:
				return 2
			else:
				raise Exception
		if order == 3:
			if index == 0 or index == 1 or index == 2:
				return index
			if index == 10 or index == 11:
				return index - 7
			else:
				raise Exception
		if order == 4:
			if index == 0 or index == 1 or index == 2 or index == 3:
				return index
			if index == 9:
				return 4
			if index == 14 or index == 15 or index == 16:
				return index - 9
			else:
				raise Exception

	T = onp.asarray(T, dtype=int)
	T_new = onp.zeros(T.shape)
	T_new[:, 0] = T[:, 0]
	T_new[:, 1] = T[:, 1]
	for i in range(T.shape[0]):
		T_new[i, 2] = convert_to_bottom_index(T[i, 2])
	return jnp.asarray(T_new, dtype=int)


def load_assembly_matrix(basedir, nx, ny, order):
	def create_assembly_matrix(nx, ny, order):
		"""
		Generates an assembly matrix which converts the
		local/element matrices to the global matrices
		"""
		table = {}
		nodes = node_locations(order)
		num_elem = nodes.shape[0]

		def lookup_table(ijk):
			i, j, k = ijk
			x, y = nodes[k, :]
			i_l = (i - 1) % nx
			i_r = (i + 1) % nx
			j_b = (j - 1) % ny
			j_t = (j + 1) % ny
			if (i, j, x, y) in table:
				return table[(i, j, x, y)]
			elif (i_l, j, x + 2, y) in table:
				return table[(i_l, j, x + 2, y)]
			elif (i_r, j, x - 2, y) in table:
				return table[(i_r, j, x - 2, y)]
			elif (i, j_t, x, y - 2) in table:
				return table[(i, j_t, x, y - 2)]
			elif (i, j_b, x, y + 2) in table:
				return table[(i, j_b, x, y + 2)]
			elif (i_l, j_t, x + 2, y - 2) in table:
				return table[(i_l, j_t, x + 2, y - 2)]
			elif (i_r, j_t, x - 2, y - 2) in table:
				return table[(i_r, j_t, x - 2, y - 2)]
			elif (i_l, j_b, x + 2, y + 2) in table:
				return table[(i_l, j_b, x + 2, y + 2)]
			elif (i_r, j_b, x - 2, y + 2) in table:
				return table[(i_r, j_b, x - 2, y + 2)]
			else:
				return None

		def assign_table(ijk, node_val):
			i, j, k = ijk
			x, y = nodes[k, :]
			table[(i, j, x, y)] = node_val
			return

		node_index = 0
		for j in range(ny):
			for i in range(nx):
				for k in range(num_elem):
					ijk = (i, j, k)
					node_val = lookup_table(ijk)
					if node_val is None:
						node_val = node_index
						node_index += 1
					assign_table(ijk, node_val)

		num_global_elements = max(table.values()) + 1
		M = onp.zeros((nx, ny, num_elem), dtype=int)
		T = -onp.ones((num_global_elements, 3), dtype=int)

		for i in range(nx):
			for j in range(ny):
				for k in range(num_elem):
					x, y = nodes[k, :]
					gamma = table[(i, j, x, y)]
					M[i, j, k] = gamma
					if T[gamma, 0] == -1 and is_bottom_element(order, k):
						T[gamma, 0] = i
						T[gamma, 1] = j
						T[gamma, 2] = k

		return num_global_elements, M, T

	if os.path.exists(
		"data/poissonmatrices/assembly_matrix_nx{}_ny{}_order{}.npy".format(
			basedir, nx, ny, order
		)
	):
		num_global_elements = onp.load(
			"data/poissonmatrices/num_global_elements_nx{}_ny{}_order{}.npy".format(
				basedir, nx, ny, order
			)
		)
		M = onp.load(
			"data/poissonmatrices/assembly_matrix_nx{}_ny{}_order{}.npy".format(
				basedir, nx, ny, order
			)
		)
		T = onp.load(
			"data/poissonmatrices/assembly_matrix_transpose_nx{}_ny{}_order{}.npy".format(
				basedir, nx, ny, order
			)
		)
	else:
		num_global_elements, M, T = create_assembly_matrix(nx, ny, order)
		onp.save(
			"data/poissonmatrices/num_global_elements_nx{}_ny{}_order{}.npy".format(
				basedir, nx, ny, order
			),
			num_global_elements,
		)
		onp.save(
			"data/poissonmatrices/assembly_matrix_nx{}_ny{}_order{}.npy".format(
				basedir, nx, ny, order
			),
			M,
		)
		onp.save(
			"data/poissonmatrices/assembly_matrix_transpose_nx{}_ny{}_order{}.npy".format(
				basedir, nx, ny, order
			),
			T,
		)
	return num_global_elements, M, T


def load_elementwise_volume(basedir, nx, ny, Lx, Ly, order):
	"""
	Returns the (num_elements x num_elements) matrix
	where the (i,j) component is the elementwise integral
	V_{ij} = int_Omega nabla psi_i nabla psi_j dx dy
	in "local" coordinates.

	Later we will map this matrix to "global" coordinates.
	"""

	def create_elementwise_volume(order):
		basis = FE_poly(order)
		num_elem = basis.shape[0]
		res1 = onp.zeros((num_elem, num_elem))
		res2 = onp.zeros((num_elem, num_elem))
		for i in range(num_elem):
			for j in range(num_elem):
				expr1 = diff(basis[i], "x") * diff(basis[j], "x")
				res1[i, j] = integrate(expr1, ("x", -1, 1), ("y", -1, 1))
				expr2 = diff(basis[i], "y") * diff(basis[j], "y")
				res2[i, j] = integrate(expr2, ("x", -1, 1), ("y", -1, 1))
		return res1, res2

	dx = Lx / nx
	dy = Ly / ny
	if os.path.exists(
		"data/poissonmatrices/elementwise_volume_{}_1.npy".format(basedir, order)
	):
		res1 = onp.load(
			"data/poissonmatrices/elementwise_volume_{}_1.npy".format(basedir, order)
		)
		res2 = onp.load(
			"data/poissonmatrices/elementwise_volume_{}_2.npy".format(basedir, order)
		)
	else:
		res1, res2 = create_elementwise_volume(order)
		onp.save(
			"data/poissonmatrices/elementwise_volume_{}_1".format(basedir, order),
			res1,
		)
		onp.save(
			"data/poissonmatrices/elementwise_volume_{}_2".format(basedir, order),
			res2,
		)
	V = res1 * (dy / dx) + res2 * (dx / dy)
	return V


def load_elementwise_source(basedir, nx, ny, Lx, Ly, order):
	def write_elementwise_source(order):
		FE_basis = FE_poly(order)
		legendre_basis = legendre_poly(order)
		res = onp.zeros((FE_basis.shape[0], legendre_basis.shape[0]))
		for i in range(FE_basis.shape[0]):
			for j in range(legendre_basis.shape[0]):
				expr = FE_basis[i] * legendre_basis[j]
				res[i, j] = integrate(expr, ("x", -1, 1), ("y", -1, 1))
		return res

	dx = Lx / nx
	dy = Ly / ny
	if os.path.exists(
		"data/poissonmatrices/elementwise_source_{}.npy".format(basedir, order)
	):
		res = onp.load(
			"data/poissonmatrices/elementwise_source_{}.npy".format(basedir, order)
		)
	else:
		res = write_elementwise_source(order)
		onp.save(
			"data/poissonmatrices/elementwise_source_{}.npy".format(basedir, order),
			res,
		)
	return res * dx * dy / 4


def load_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, num_global_elements):
	if os.path.exists(
		"data/poissonmatrices/volume_{}_{}_{}.npz".format(basedir, nx, ny, order)
	):
		sV = sparse.load_npz(
			"data/poissonmatrices/volume_{}_{}_{}.npz".format(basedir, nx, ny, order)
		)
	else:
		V = create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, num_global_elements)
		sV = sparse.csr_matrix(V)
		sparse.save_npz(
			"data/poissonmatrices/volume_{}_{}_{}.npz".format(
				basedir, nx, ny, order
			),
			sV,
		)
	return sV


def create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, num_global_elements):
	num_elem = num_elements(order)
	K_elementwise = load_elementwise_volume(basedir, nx, ny, Lx, Ly, order)

	sK = dok_matrix((num_global_elements, num_global_elements))

	for j in range(ny):
		for i in range(nx):
			sK[M[i, j, :][:, None], M[i, j, :][None, :]] += K_elementwise[:, :]
	return sK


def get_kernel(order):
	bottom_indices = get_bottom_indices(order)
	K = onp.zeros((2, 2, num_elements_FE(order), num_elements_FE(order)))
	if order == 1 or order == 0:
		K[0, 0, 0, 2] = 1.0
		K[1, 0, 0, 3] = 1.0
		K[0, 1, 0, 1] = 1.0
		K[1, 1, 0, 0] = 1.0
	elif order == 2:
		K[0, 0, 0, 4] = 1.0
		K[1, 0, 0, 6] = 1.0
		K[1, 0, 1, 5] = 1.0
		K[0, 1, 0, 2] = 1.0
		K[0, 1, 7, 3] = 1.0
		K[1, 1, 0, 0] = 1.0
		K[1, 1, 1, 1] = 1.0
		K[1, 1, 7, 7] = 1.0
	elif order == 3:
		K[0, 0, 0, 6] = 1.0
		K[1, 0, 0, 9] = 1.0
		K[1, 0, 1, 8] = 1.0
		K[1, 0, 2, 7] = 1.0
		K[0, 1, 0, 3] = 1.0
		K[0, 1, 11, 4] = 1.0
		K[0, 1, 10, 5] = 1.0
		K[1, 1, 0, 0] = 1.0
		K[1, 1, 1, 1] = 1.0
		K[1, 1, 2, 2] = 1.0
		K[1, 1, 10, 10] = 1.0
		K[1, 1, 11, 11] = 1.0
	elif order == 4:
		K[1, 1, 0, 0] = 1.0
		K[1, 1, 1, 1] = 1.0
		K[1, 1, 2, 2] = 1.0
		K[1, 1, 3, 3] = 1.0
		K[1, 1, 9, 9] = 1.0
		K[1, 1, 14, 14] = 1.0
		K[1, 1, 15, 15] = 1.0
		K[1, 1, 16, 16] = 1.0
		K[0, 0, 0, 8] = 1.0
		K[1, 0, 0, 13] = 1.0
		K[1, 0, 1, 12] = 1.0
		K[1, 0, 2, 11] = 1.0
		K[1, 0, 3, 10] = 1.0
		K[0, 1, 0, 4] = 1.0
		K[0, 1, 16, 5] = 1.0
		K[0, 1, 15, 6] = 1.0
		K[0, 1, 14, 7] = 1.0
	else:
		raise Exception
	return jnp.asarray(K)[:, :, bottom_indices, :]


def get_poisson_solver(nx, ny, Lx, Ly, order):
	basedir = None
	N_global_elements, M, T = load_assembly_matrix(basedir, nx, ny, order)
	T = convert_to_bottom_indices(T, order)
	S_elem = load_elementwise_source(basedir, nx, ny, Lx, Ly, order)

	K = get_kernel(order) @ S_elem

	sV = load_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	V_sp = jsparse.BCOO.from_scipy_sparse(sV)
	args = V_sp.data, V_sp.indices, N_global_elements
	kwargs = {"forward": True}
	custom_lu_solve = lambda b: sparse_solve_prim(b, *args, **kwargs)

	def solve(xi):
		xi = xi.at[:, :, 0].add(-jnp.mean(xi[:, :, 0]))
		xi = jnp.pad(xi, ((1, 0), (1, 0), (0, 0)), mode="wrap")
		F_ijb = jax.lax.conv_general_dilated(
			xi[None, ...],
			K,
			(1, 1),
			padding="VALID",
			dimension_numbers=("NHWC", "HWOI", "NHWC"),
		)[0]
		b = -F_ijb[T[:, 0], T[:, 1], T[:, 2]]

		res = custom_lu_solve(b)
		res = res - jnp.mean(res)
		output = res.at[M].get()
		return output

	jax.jit(solve)(jnp.zeros((nx, ny, num_elements(order))))

	return solve







def _trapezoidal_integration(f, xi, xf, yi, yf, n=None):
	return (xf - xi) * (yf - yi) * (f(xi, yi) + f(xf, yi) + f(xi, yf) + f(xf, yf)) / 4


def _2d_fixed_quad(f, xi, xf, yi, yf, n=3):
	"""
	Takes a 2D-valued function of two 2D inputs
	f(x,y) and four scalars xi, xf, yi, yf, and
	integrates f over the 2D domain to order n.
	"""

	w_1d = {
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

	xi_i_1d = {
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

	x_w, y_w =jnp.meshgrid(w_1d, w_1d)
	x_w = x_w.reshape(-1)
	y_w = y_w.reshape(-1)
	w_2d = x_w * y_w

	xi_x, xi_y =jnp.meshgrid(xi_i_1d, xi_i_1d)
	xi_x = xi_x.reshape(-1)
	xi_y = xi_y.reshape(-1)

	x_i = (xf + xi) / 2 + (xf - xi) / 2 * xi_x
	y_i = (yf + yi) / 2 + (yf - yi) / 2 * xi_y
	wprime = w_2d * (xf - xi) * (yf - yi) / 4
	return jnp.sum(wprime[None, :] * f(x_i, y_i), axis=1)


def evalf_2D(x, y, a, dx, dy, order):
	j = jnp.floor(x / dx).astype(int)
	k = jnp.floor(y / dy).astype(int)
	x_j = dx * (0.5 + j)
	y_k = dy * (0.5 + k)
	xi_x = (x - x_j) / (0.5 * dx)
	xi_y = (y - y_k) / (0.5 * dy)
	f_eval = _eval_legendre(order)
	legendre_val = jnp.transpose(f_eval(xi_x, xi_y), (1, 2, 0))
	return jnp.sum(a[j, k, :] * legendre_val, axis=-1)


def _evalf_2D_integrate(x, y, a, dx, dy, order):
	j = jnp.floor(x / dx).astype(int)
	k = jnp.floor(y / dy).astype(int)
	x_j = dx * (0.5 + j)
	y_k = dy * (0.5 + k)
	xi_x = (x - x_j) / (0.5 * dx)
	xi_y = (y - y_k) / (0.5 * dy)
	f_eval = _eval_legendre(order)
	return jnp.sum(a[j, k, :] * f_eval(xi_x, xi_y).T, axis=-1)


def _eval_legendre(order):
	polybasis = legendre_npbasis(order)  # (order+1, k, 2) matrix
	_vmap_polyval = vmap(jnp.polyval, (1, None), 0)

	def f(xi_x, xi_y):
		return _vmap_polyval(polybasis[:, :, 0], xi_x) * _vmap_polyval(
			polybasis[:, :, 1], xi_y
		)  # (k,) array

	return f


def inner_prod_with_legendre(nx, ny, Lx, Ly, order, func, t, quad_func=_2d_fixed_quad, n=5):
	dx = Lx / nx
	dy = Ly / ny

	i = jnp.arange(nx)
	x_i = dx * i
	x_f = dx * (i + 1)
	j = jnp.arange(ny)
	y_i = dy * j
	y_f = dy * (j + 1)

	def xi_x(x):
		k = jnp.floor(x / dx)
		x_k = dx * (0.5 + k)
		return (x - x_k) / (0.5 * dx)

	def xi_y(y):
		k = jnp.floor(y / dy)
		y_k = dy * (0.5 + k)
		return (y - y_k) / (0.5 * dy)

	quad_lambda = lambda f, xi, xf, yi, yf: quad_func(f, xi, xf, yi, yf, n=n)

	_vmap_integrate = vmap(
		vmap(quad_lambda, (None, 0, 0, None, None), 0), (None, None, None, 0, 0), 1
	)
	to_int_func = lambda x, y: func(x, y, t) * _eval_legendre(order)(xi_x(x), xi_y(y))
	return _vmap_integrate(to_int_func, x_i, x_f, y_i, y_f)

def f_to_DG(nx, ny, Lx, Ly, order, func, t, quad_func=_2d_fixed_quad, n=8):
	inner_prod = legendre_inner_product(order)
	dx = Lx / nx
	dy = Ly / ny

	return inner_prod_with_legendre(nx, ny, Lx, Ly, order, func, t, quad_func=quad_func, n=n) / (
		inner_prod[None, None, :] * dx * dy
	)


def f_to_source(nx, ny, Lx, Ly, order, func, t, quad_func=_2d_fixed_quad, n=5):
	repr_dg = f_to_DG(nx, ny, Lx, Ly, order, func, t, quad_func=quad_func, n=n)
	return repr_dg.at[:, :, 0].add(-jnp.mean(repr_dg[:, :, 0]))


def f_to_FE(nx, ny, Lx, Ly, order, func, t):
	dx = Lx / nx
	dy = Ly / ny
	i = jnp.arange(nx)
	x_i = dx * i + dx / 2
	j = jnp.arange(ny)
	y_i = dy * j + dx / 2
	nodes = jnp.asarray(node_locations(order), dtype=float)

	x_eval = (
		jnp.ones((nx, ny, nodes.shape[0])) * x_i[:, None, None]
		+ nodes[None, None, :, 0] * dx / 2
	)
	y_eval = (
		jnp.ones((nx, ny, nodes.shape[0])) * y_i[None, :, None]
		+ nodes[None, None, :, 1] * dy / 2
	)

	_vmap_evaluate = vmap(vmap(vmap(func, (0, 0, None)), (0, 0, None)), (0, 0, None))
	return _vmap_evaluate(x_eval, y_eval, t)







def get_initial_condition(key, Lx, Ly, initial_condition, num_init_modes=8, max_k = 4, min_k = 1):
	def gaussian(x, y, t):
		xc, yc = Lx / 2, Ly / 2
		return jnp.exp(
			-75 * ((x - xc) ** 2 / Lx ** 2 + (y - yc) ** 2 / Ly ** 2)
		)

	def diffusion_test(x, y, t):
		return jnp.sin(x * 2 * jnp.pi / Lx) * jnp.cos(y * 2 * jnp.pi / Ly)



	def cosine_hump(x, y, t):
		x0 = 0.25 * Lx
		y0 = 0.5 * Ly
		r0 = 0.2 * jnp.sqrt((Lx ** 2 + Ly ** 2) / 2)
		r = jnp.minimum(jnp.sqrt((x - x0) ** 2 + (y - y0) ** 2), r0) / r0
		return 0.25 * (1 + jnp.cos(jnp.pi * r))

	def two_cosine_humps(x, y, t):
		x0a = 0.25 * Lx
		x0b = 0.75 * Lx
		y0 = 0.5 * Ly
		r0 = 0.2 * jnp.sqrt((Lx ** 2 + Ly ** 2) / 2)
		ra = jnp.minimum(jnp.sqrt((x - x0a) ** 2 + (y - y0) ** 2), r0) / r0
		rb = jnp.minimum(jnp.sqrt((x - x0b) ** 2 + (y - y0) ** 2), r0) / r0
		return 0.25 * (1 + jnp.cos(jnp.pi * ra)) + 0.25 * (1 + jnp.cos(jnp.pi * rb))

	def double_shear(x, y, t):
		rho = 1 / 30
		delta = 0.05
		div = jnp.pi / 15
		return (
			delta * jnp.cos(2 * jnp.pi * x / Lx)
			+ (y > Ly / 2) * jnp.cosh((3 / 4 - y / Ly) / rho) ** (-2) / div
			- (y <= Ly / 2) * jnp.cosh((y / Ly - 1 / 4) / rho) ** (-2) / div
		)

	def vortex_waltz(x, y, t):
		x1 = 0.35 * Lx
		y1 = 0.5 * Ly
		x2 = 0.65 * Lx
		y2 = 0.5 * Ly
		denom_x = 0.8 * (Lx ** 2) / (100.0)
		denom_y = 0.8 * (Ly ** 2 / 100)
		gaussian_1 = jnp.exp(-((x - x1) ** 2 / denom_x + (y - y1) ** 2 / denom_y))
		gaussian_2 = jnp.exp(-((x - x2) ** 2 / denom_x + (y - y2) ** 2 / denom_y))
		return gaussian_1 + gaussian_2


	def sum_modes(x, y, amplitudes, ks_x, ks_y, phases_x, phases_y):
		return jnp.sum(
			amplitudes[None, :]
			* jnp.sin(
				ks_x[None, :] * 2 * jnp.pi / Lx * x[:, None] + phases_x[None, :]
			) * jnp.sin(
				ks_y[None, :] * 2 * jnp.pi / Ly * y[:, None] + phases_y[None, :]
			),
			axis=1,
		)

	if initial_condition == "zero" or initial_condition == "zeros":
		return zeros
	elif initial_condition == "cosine_hump":
		return cosine_hump
	elif (
		initial_condition == "two_humps"
		or initial_condition == "two_cosine_humps"
	):
		return two_cosine_humps
	elif initial_condition == "double_shear":
		return double_shear
	elif initial_condition == "vortex_waltz":
		return vortex_waltz
	elif initial_condition == "random":
		key1, key2, key3, key4, key5 = random.split(key, 5)
		phases_x = random.uniform(key1, (num_init_modes,)) * 2 * jnp.pi
		phases_y = random.uniform(key2, (num_init_modes,)) * 2 * jnp.pi
		ks_x = random.randint(
			key3, (num_init_modes,), min_k, max_k
		)
		ks_y = random.randint(
			key4, (num_init_modes,), min_k, max_k
		)
		amplitudes = random.uniform(key5, (num_init_modes,))
		return lambda x, y, t: sum_modes(x, y, amplitudes, ks_x, ks_y, phases_x, phases_y)
	elif initial_condition == 'diffusion_test':
		return diffusion_test
	else:
		raise NotImplementedError




def eval_FE(phi, Lx, Ly, order, x, y):
	nx = phi.shape[0]
	ny = phi.shape[1]
	basis = FE_poly(order)

	assert x >= 0.0 and x <= Lx
	assert y >= 0.0 and y <= Ly

	dx = Lx / nx
	dy = Ly / ny

	i = int(x // dx)
	j = int(y // dy)



	xi_i = x % dx
	xi_j = y % dy

	num_elem = phi.shape[-1]
	basis_x = onp.zeros(num_elem)
	for k in range(num_elem):
		basis_x[k] = basis[k].subs("x", xi_i).subs("y", xi_j)

	return onp.sum(basis_x * phi[i, j, :])







def plot_FE_basis(
	nx, ny, Lx, Ly, order, phi, title="", plotting_density=4
):
	"""
	Inputs:

	phi, (nx, ny, num_elem) matrix
	"""
	factor = order * plotting_density + 1
	num_elem = phi.shape[-1]
	basis = FE_poly(order)
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
			] = onp.sum(basis_x * phi[i, j, None, None, :], axis=-1)

	fig, axs = plt.subplots(figsize=(5 * onp.sqrt(Lx / Ly) + 1, onp.sqrt(Ly / Lx) * 5))
	x_plot = onp.linspace(0, Lx, Nx_plot + 1)
	y_plot = onp.linspace(0, Ly, Ny_plot + 1)
	pcm = axs.pcolormesh(
		x_plot,
		y_plot,
		output.T,
		shading="flat",
		cmap=sns.cm.icefire,  # vmin=0, vmax=1
	)
	axs.contour(
		(x_plot[:-1] + x_plot[1:]) / 2,
		(y_plot[:-1] + y_plot[1:]) / 2,
		output.T,
		colors="black",
	)
	axs.set_xlim([0, Lx])
	axs.set_ylim([0, Ly])
	axs.set_xticks([0, Lx])
	axs.set_yticks([0, Ly])
	axs.set_title(title)
	fig.colorbar(pcm, ax=axs, extend="max")




def plot_DG_basis(
	nx, ny, Lx, Ly, order, zeta, title="", plotting_density=4
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
	x_plot = onp.linspace(0, Lx, Nx_plot + 1)
	y_plot = onp.linspace(0, Ly, Ny_plot + 1)
	pcm = axs.pcolormesh(
		x_plot,
		y_plot,
		output.T,
		shading="flat",
		cmap=sns.cm.icefire,
	)
	axs.set_xlim([0, Lx])
	axs.set_ylim([0, Ly])
	axs.set_xticks([0, Lx])
	axs.set_yticks([0, Ly])
	axs.set_title(title)
	fig.colorbar(pcm, ax=axs, extend="max")






def l1_norm(a, b):
	return onp.abs(a-b)



Lx = Ly = 2 * jnp.pi
nx = ny = 64
nx_lows = [4, 8, 16]
order = 2
t = 0.0
key = jax.random.PRNGKey(3)
f_source = get_initial_condition(key, Lx, Ly, "two_humps")


# High-resolution
source = f_to_source(nx, ny, Lx, Ly, order, f_source, t, n=8)
f_solve = jit(get_poisson_solver(nx, ny, Lx, Ly, order))
phi = f_solve(source)


colors = ["red", "green", "blue"]

fig, ax = plt.subplots(figsize=(5,3))


for i, nx_low in enumerate(nx_lows):
	ny_low = nx_low

	source_low = f_to_source(nx_low, ny_low, Lx, Ly, order, f_source, t, n=8)
	f_solve_low = jit(get_poisson_solver(nx_low, ny_low, Lx, Ly, order))
	phi_low = f_solve_low(source_low)
	plot_DG_basis(nx_low, ny_low, Lx, Ly, order, source_low, title="Source for Poisson's equation", plotting_density=6)
	plot_FE_basis(nx_low, ny_low, Lx, Ly, order, phi_low, title="Approximation to Poisson's equation", plotting_density=6)


	mean_l1_errors = []

	nxs_test_res = [4, 8, 16, 32, 64]

	for nx_test in nxs_test_res:
		
		ny_test = nx_test
		dx_test = Lx / nx_test
		dy_test = Ly / ny_test

		x_eval = onp.arange(nx_test) * dx_test + dx_test/2
		y_eval = onp.arange(ny_test) * dy_test + dy_test/2

		eval_exact = lambda x, y: eval_FE(phi, Lx, Ly, order, x, y)
		eval_low = lambda x, y: eval_FE(phi_low, Lx, Ly, order, x, y)




		mean_l1_error = 0.0
		total_test_elements = nx_test * ny_test

		for x in x_eval:
			for y in y_eval:
				mean_l1_error += l1_norm(eval_exact(x, y), eval_low(x, y)) / total_test_elements

		print("Training resolution: {}x{} Testing resolution: {}x{} L1 error: {}".format(nx_low, ny_low, nx_test, ny_test, mean_l1_error))

		mean_l1_errors.append(mean_l1_error)

	ax.plot(nxs_test_res, mean_l1_errors, color=colors[i], marker='o', linewidth=2.0, label="Train resolution: {}x{}".format(nx_low, ny_low))
ax.set_xlabel("Testing resolution")
ax.set_ylabel("Mean L1 Error")
ax.set_ylim([0.001,0.05])
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='x', which='minor', bottom=False)
ax.set_xticks([4,8,16,32,64])
ax.set_xticklabels(["4", "8", "16", "32", "64"])

fig.legend()
fig.tight_layout()

plt.show()
	
