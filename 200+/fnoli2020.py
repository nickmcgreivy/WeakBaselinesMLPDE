import jax.numpy as np
import numpy as onp
from time import time
from jax import config, jit, vmap, hessian, lax, core, config, xla, grad
import argparse
from enum import Enum
import sys
import sympy
from sympy.matrices import Matrix, zeros
from sympy import Rational, symbols, legendre, integrate, diff, nsimplify, lambdify
from functools import lru_cache, partial
from scipy.special import legendre as scipy_legendre
import jax.numpy as np
from sympy import legendre, diff, integrate, symbols
from sympy.matrices import Matrix, zeros
from scipy.special import eval_legendre
import torch
import math
import os.path
import os
import jax
from jax.experimental import sparse as jsparse
from jax.experimental.sparse.linalg import spsolve
from scipy import sparse
from scipy.sparse import dok_matrix
from jax.scipy.linalg import lu_factor
import scipy
from jax.lax import scan
from jax._src import api, abstract_arrays
from jax.interpreters import ad
from jax.scipy.linalg import lu_solve
from scipy.linalg import lu_solve as np_lu_solve
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import splu
from jaxlib import xla_client
from jax.lib import xla_bridge
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
"""
import jax_cfd.base as cfd
from jax_cfd.base import boundaries
from jax_cfd.base import forcings
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
"""
PI = np.pi





use_64_bit = True
config.update("jax_enable_x64", use_64_bit)
if use_64_bit:
	floatdtype = np.float64
else:
	floatdtype = np.float32












sparse_solve_p = core.Primitive("sparse_solve")


#########
# Custom operation base
#########


def sparse_solve_prim(b, sparse_data, sparse_indices, size, forward=True):
	return sparse_solve_p.bind(b, sparse_data, sparse_indices, size, forward)


def sparse_solve_impl(b, sparse_data, sparse_indices, size, forward=True):
	raise Exception("Sparse solve prim shouldn't be called except from within JIT")


def sparse_solve_abstract_eval(b, sparse_data, sparse_indices, size, forward=True):
	return abstract_arrays.ShapedArray(b.shape, b.dtype)


#########
# grad
#########


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


#########
# JIT
#########

import custom_call_sparse_solve_ldlt as custom_call_sparse_solve

for _name, _value in custom_call_sparse_solve.registrations().items():
	xla_client.register_cpu_custom_call_target(_name, _value)


def sparse_solve_xla_translation(c, bc, sparse_data, sparse_indices, size, forward):
	bc_shape = c.get_shape(bc)
	bc_dtype = bc_shape.element_type()
	bc_dims = bc_shape.dimensions()
	bc_shape = xla_client.Shape.array_shape(np.dtype(bc_dtype), bc_dims, (0,))
	out_shape = xla_client.Shape.array_shape(np.dtype(bc_dtype), bc_dims, (0,))
	data_shape = c.get_shape(sparse_data)
	data_dtype = data_shape.element_type()
	data_dims = data_shape.dimensions()
	data_shape = xla_client.Shape.array_shape(np.dtype(data_dtype), data_dims, (0,))
	indices_shape = c.get_shape(sparse_indices)
	indices_dtype = indices_shape.element_type()
	indices_dims = indices_shape.dimensions()
	indices_shape = xla_client.Shape.array_shape(
		np.dtype(indices_dtype), indices_dims, (0, 1)
	)

	assert bc_dtype == data_dtype

	if bc_dtype == np.float32:
		op_name = b"sparse_solve_f32"
	elif bc_dtype == np.float64:
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
			xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
			xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
			xla_client.Shape.array_shape(np.dtype(bool), (), ()),
		),
	)


#########
# VMAP
########

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


#########
# BOILERPLATE CODE TO REGISTER PRIMITIVE
########

xla.backend_specific_translations["cpu"][sparse_solve_p] = sparse_solve_xla_translation
sparse_solve_p.def_impl(sparse_solve_impl)
sparse_solve_p.def_abstract_eval(sparse_solve_abstract_eval)
ad.primitive_jvps[sparse_solve_p] = sparse_solve_value_and_jvp
ad.primitive_transposes[sparse_solve_p] = sparse_solve_transpose
batching.primitive_batchers[sparse_solve_p] = sparse_solve_batch






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




class Flux(Enum):
	"""
	Flux is a subclass of Enum, which determines the flux that is used to compute
	the time-derivative of the equation.

	LEARNED is the data-driven discretization of the equation, called the "learned
	flux interpolation"
	"""

	UPWIND = "upwind"
	CENTERED = "centered"
	LEARNED = "learned"
	VANLEER = "vanleer"
	CONSERVATION = "conservation"

	def __str__(self):
		return self.value






### Step 1: Generate 2d polynomial bases at each face

@lru_cache(maxsize=5)
def monomials_double(order):
	x = zeros((2*order + 2) ** 2, 2)
	i = 0
	for p_xi in range(2 * order + 2):
		for p_eta in range(order + 1):
			if p_xi <= 3 or p_eta <= 1 or p_xi + 2 * p_eta <= 2 * order + 1:
				x[i, 0] = p_xi
				x[i, 1] = p_eta
				i += 1
	return x[0:i, :]

@lru_cache(maxsize=5)
def get_polynomial_elements_right(order):
	"""
	Outputs: length 2*(order + 1) basis of polynomials in x, y
	"""
	v = monomials_double(order)
	x = symbols("x")
	y = symbols("y")
	NumElements = v.shape[0]
	return Matrix(
		[
			x**int(v[k, 0]) * y**int(v[k, 1])
			for k in range(NumElements)
		]
	)

@lru_cache(maxsize=5)
def get_polynomial_elements_top(order):
	"""
	Outputs: length 2*(order + 1) basis of polynomials in x, y
	"""
	v = monomials_double(order)
	x = symbols("x")
	y = symbols("y")
	NumElements = v.shape[0]
	return Matrix(
		[
			x**int(v[k, 1]) * y**int(v[k, 0])
			for k in range(NumElements)
		]
	)


### Step 2: Generate R, M, Q matrices

@lru_cache(maxsize=5)
def get_M(order):
	K = num_elements(order)
	legendre_basis = legendre_poly(order)
	M = zeros(2 * K, 2 * K)

	for k in range(K):
		for l in range(K):
			expr = legendre_basis[k] * legendre_basis[l]
			val = integrate(expr, ("x", -1, 1), ("y", -1, 1)) / 4
			M[k, l] = val
			M[k + K, l + K] = val
	return M

@lru_cache(maxsize=5)
def get_R_right(order):
	K = num_elements(order)
	legendre_basis = legendre_poly(order)
	recovery_basis = get_polynomial_elements_right(order)
	R = zeros(2 * K, 2 * K)

	for k in range(K):
		for l in range(2 * K):
			expr = legendre_basis[k].subs("x", 2 * symbols("x") + 1) * recovery_basis[l]
			R[k,l] = integrate(expr, ("x", -1, 0), ("y", -1, 1)) / 2
	for k in range(K, 2 * K):
		for l in range(2 * K):
			expr = legendre_basis[k-K].subs("x", 2 * symbols("x") - 1) * recovery_basis[l]
			R[k,l] = integrate(expr, ("x", 0, 1), ("y", -1, 1)) / 2
	return R

@lru_cache(maxsize=5)
def get_R_top(order):
	K = num_elements(order)

	legendre_basis = legendre_poly(order)
	recovery_basis = get_polynomial_elements_top(order)

	R = zeros(2 * K, 2 * K)

	for k in range(K):
		for l in range(2 * K):
			expr = legendre_basis[k].subs("y", 2 * symbols("y") + 1) * recovery_basis[l]
			R[k,l] = integrate(expr, ("x", -1, 1), ("y", -1, 0)) / 2
	for k in range(K, 2 * K):
		for l in range(2 * K):
			expr = legendre_basis[k-K].subs("y", 2 * symbols("y") - 1) * recovery_basis[l]
			R[k,l] = integrate(expr, ("x", -1, 1), ("y", 0, 1)) / 2
	return R

@lru_cache(maxsize=5)
def get_Q_right(order):
	R = get_R_right(order)
	M = get_M(order)

	R_inv = R.inv()
	return R_inv @ M

@lru_cache(maxsize=5)
def get_Q_top(order):
	R = get_R_top(order)
	M = get_M(order)

	R_inv = R.inv()
	return R_inv @ M


### Step 3: Generate b coefficients 

def get_b_right(order):
	Q = np.asarray(get_Q_right(order), dtype=floatdtype)
	
	def b_right(zeta):
		a = np.concatenate((zeta, np.roll(zeta, -1, axis=0)), axis=-1)
		return a @ Q.T

	return b_right

def get_b_top(order):
	Q = np.asarray(get_Q_top(order), dtype=floatdtype)

	def b_top(zeta):
		a = np.concatenate((zeta, np.roll(zeta, -1, axis=1)), axis=-1)
		return a @ Q.T

	return b_top


### Step 4: Compute boundary term

@lru_cache(maxsize=5)
def get_A_right(order):
	K = num_elements(order)
	legendre_basis = legendre_poly(order)
	recovery_basis = get_polynomial_elements_right(order)

	# right
	A = zeros(K, 2 * K)
	for k in range(K):
		for l in range(2 * K):
			expr = legendre_basis[k].subs("x", 1) * diff(recovery_basis[l], "x").subs("x", 0) - 2 * diff(legendre_basis[k], "x").subs("x", 1) * recovery_basis[l].subs("x", 0)
			A[k,l] = integrate(expr, ("y", -1, 1)) / 2
	return A

@lru_cache(maxsize=5)
def get_A_left(order):
	K = num_elements(order)
	legendre_basis = legendre_poly(order)
	recovery_basis = get_polynomial_elements_right(order)

	# right
	A = zeros(K, 2 * K)
	for k in range(K):
		for l in range(2 * K):
			expr = legendre_basis[k].subs("x", -1) * diff(recovery_basis[l], "x").subs("x", 0) - 2 * diff(legendre_basis[k], "x").subs("x", -1) * recovery_basis[l].subs("x", 0)
			A[k,l] = integrate(expr, ("y", -1, 1)) / 2
	return A

@lru_cache(maxsize=5)
def get_A_top(order):
	K = num_elements(order)
	legendre_basis = legendre_poly(order)
	recovery_basis = get_polynomial_elements_top(order)

	# right
	A = zeros(K, 2 * K)
	for k in range(K):
		for l in range(2 * K):
			expr = legendre_basis[k].subs("y", 1) * diff(recovery_basis[l], "y").subs("y", 0) - 2 * diff(legendre_basis[k], "y").subs("y", 1) * recovery_basis[l].subs("y", 0)
			A[k,l] = integrate(expr, ("x", -1, 1)) / 2
	return A

@lru_cache(maxsize=5)
def get_A_bottom(order):
	K = num_elements(order)
	legendre_basis = legendre_poly(order)
	recovery_basis = get_polynomial_elements_top(order)

	# right
	A = zeros(K, 2 * K)
	for k in range(K):
		for l in range(2 * K):
			expr = legendre_basis[k].subs("y", -1) * diff(recovery_basis[l], "y").subs("y", 0) - 2 * diff(legendre_basis[k], "y").subs("y", -1) * recovery_basis[l].subs("y", 0)
			A[k,l] = integrate(expr, ("x", -1, 1)) / 2
	return A



def get_boundary_term(order, Lx, Ly):

	A_R = np.asarray(get_A_right(order), dtype=floatdtype) * (Ly/Lx)
	A_T = np.asarray(get_A_top(order), dtype=floatdtype) * (Lx/Ly)
	A_L = np.asarray(get_A_left(order), dtype=floatdtype) * (Ly/Lx)
	A_B = np.asarray(get_A_bottom(order), dtype=floatdtype) * (Lx/Ly)

	f_b_right = get_b_right(order)
	f_b_top = get_b_top(order)

	def f_boundary(zeta):
		b_R = f_b_right(zeta)
		b_T = f_b_top(zeta)
		b_L = f_b_right(np.roll(zeta, 1, axis=0))
		b_B = f_b_top(np.roll(zeta, 1, axis=1))

		right = b_R @ A_R.T
		top = b_T @ A_T.T
		left = b_L @ A_L.T
		bottom = b_B @ A_B.T

		return right + top - left - bottom

	return f_boundary




### Step 5: Compute volume term

@lru_cache(maxsize=5)
def get_V(order, Lx, Ly):
	K = num_elements(order)
	V = zeros(K, K)

	legendre_basis = legendre_poly(order)

	for k in range(K):
		for l in range(K):
			expr = legendre_basis[k] * (diff(diff(legendre_basis[l], "x"), "x") * (Ly/Lx) + diff(diff(legendre_basis[l], "y"), "y")) * (Lx/Ly)
			V[k,l] = integrate(integrate(expr, ("x", -1, 1)), ("y", -1, 1))
	return V

def get_volume_term(order, Lx, Ly):
	V = np.asarray(get_V(order, Lx, Ly), dtype=floatdtype)

	def f_volume(zeta):
		return zeta @ V

	return f_volume

### Step 6: Compute diffusion term


def get_diffusion_func(order, Lx, Ly, D):
	f_boundary_term = get_boundary_term(order, Lx, Ly)
	f_volume_term = get_volume_term(order, Lx, Ly)

	def f_diffusion(zeta):
		return D * (f_boundary_term(zeta) + f_volume_term(zeta))
	
	return f_diffusion






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
	nodes = np.asarray(node_locations(order), dtype=floatdtype)

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
	a, order_new, order_old, nx_new, ny_new, Lx, Ly, n = 8
):
	"""
	Inputs:
	a: (nx, ny, num_elements(order_old))

	Outputs:
	a_converted: (nx_new, ny_new, num_elements(order_new))
	"""
	nx_high, ny_high = a.shape[0:2]
	dx_high = Lx / nx_high
	dy_high = Ly / ny_high

	if (nx_new < nx_high and ny_new < ny_high and nx_high % nx_new == 0 and ny_high % nx_new == 0 and nx_high % 2 == 0 and ny_high % 2 == 0) or (nx_high / nx_new > 2 and ny_high / ny_new > 2 and nx_high % 2 == 0 and ny_high % 2 == 0):

		def convert_repr(a):
			def f_high(x, y, t):
				return _evalf_2D_integrate(x, y, a, dx_high, dy_high, order_old)

			return f_to_DG(nx_high // 2, ny_high // 2, Lx, Ly, order_new, f_high, 0.0, n=n)

		a_ds = convert_repr(a)
		return convert_DG_representation(a_ds, order_new, order_new, nx_new, ny_new, Lx, Ly, n=n)

	else:

		def convert_repr(a):
			def f_high(x, y, t):
				return _evalf_2D_integrate(x, y, a, dx_high, dy_high, order_old)

			return f_to_DG(nx_new, ny_new, Lx, Ly, order_new, f_high, 0.0, n=n)

		return convert_repr(a)


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


PI = np.pi

class GaussianRF(object):

	def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):

		self.dim = dim
		self.device = device

		if sigma is None:
			sigma = tau**(0.5*(2*alpha - self.dim))

		k_max = size//2

		if dim == 1:
			k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
						   torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

			self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
			self.sqrt_eig[0] = 0.0

		elif dim == 2:
			wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
									torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

			k_x = wavenumers.transpose(0,1)
			k_y = wavenumers

			self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
			self.sqrt_eig[0,0] = 0.0

		elif dim == 3:
			wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
									torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

			k_x = wavenumers.transpose(1,2)
			k_y = wavenumers
			k_z = wavenumers.transpose(0,2)

			self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
			self.sqrt_eig[0,0,0] = 0.0

		self.size = []
		for j in range(self.dim):
			self.size.append(size)

		self.size = tuple(self.size)

	def sample(self, N):

		coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
		coeff = self.sqrt_eig * coeff

		return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real



def get_initial_condition_FNO(s=256):
	GRF = GaussianRF(2, s, alpha=2.5, tau=7)
	return np.asarray(GRF.sample(1)[0][:,:,None], dtype=floatdtype)



def forward_euler(a_n, t_n, F, dt):
	return a_n + dt * F(a_n, t_n), t_n + dt


def ssp_rk2(a_n, t_n, F, dt):
	"""
	Takes a set of coefficients a_n, and outputs
	a set of coefficients a_{n+1} using a strong-stability
	preserving RK2 method.

	Uses the equations
	a_1 = a_n + dt * F(a_n, t_n)
	a_{n+1} = 1/2 a_n + 1/2 a_1 + 1/2 * dt * F(a_1, t_n + dt)

	Inputs
	a_n: value of vector at beginning of timestep
	t_n: time at beginning of timestep
	F: da/dt = F(a, t), vector function
	dt: timestep

	Outputs
	a_{n+1}: value of vector at end of timestep
	t_{n+1}: time at end of timestep

	"""
	a_1 = a_n + dt * F(a_n, t_n)
	return 0.5 * a_n + 0.5 * a_1 + 0.5 * dt * F(a_1, t_n + dt), t_n + dt


def ssp_rk3(a_n, t_n, F, dt):
	"""
	Takes a set of coefficients a_n, and outputs
	a set of coefficients a_{n+1} using a strong-stability
	preserving RK3 method.

	Uses the equations
	a_1 = a_n + dt * F(a_n, t_n)
	a_2 = 3/4 a_n + 1/4 * a_1 + 1/4 * dt * F(a_1, t_n + dt)
	a_{n+1} = 1/3 a_n + 2/3 a_2 + 2/3 * dt * F(a_2, t_n + dt/2)

	Inputs
	a_n: value of vector at beginning of timestep
	t_n: time at beginning of timestep
	F: da/dt = F(a, t), vector function
	dt: timestep

	Outputs
	a_{n+1}: value of vector at end of timestep
	t_{n+1}: time at end of timestep

	"""
	dadt1 = F(a_n, t_n)
	a_1 = a_n + dt * dadt1
	dadt2 = F(a_1, t_n + dt)
	a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * dadt2)
	dadt3 = F(a_2, t_n + dt / 2)
	return 1 / 3 * a_n + 2 / 3 * (a_2 + dt * dadt3), t_n + dt




def ssp_rk3_adaptive(a_n, F, dt, H, f_poisson_solve):
	a_1 = a_n + dt * F(a_n, H)
	H_1 = f_poisson_solve(a_1)
	a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * F(a_1, H_1))
	H_2 = f_poisson_solve(a_2)
	return 1 / 3 * a_n + 2 / 3 * (a_2 + dt * F(a_2, H_2))




FUNCTION_MAP = {
	"FE": forward_euler,
	"fe": forward_euler,
	"forward_euler": forward_euler,
	"rk2": ssp_rk2,
	"RK2": ssp_rk2,
	"ssp_rk2": ssp_rk2,
	"rk3": ssp_rk3,
	"RK3": ssp_rk3,
	"ssp_rk3": ssp_rk3,
	"ssp_rk3_adaptive": ssp_rk3_adaptive,

}









@lru_cache(maxsize=4)
def load_poisson_volume(order):
	if os.path.exists(
		"data/poissonmatrices/poisson_bracket_volume_{}.npy".format(order)
	):
		V = onp.load(
			"data/poissonmatrices/poisson_bracket_volume_{}.npy".format(
				order
			)
		)
	else:
		V = create_poisson_bracket_volume_matrix(order)
		onp.save(
			"data/poissonmatrices/poisson_bracket_volume_{}.npy".format(
				order
			),
			V,
		)
	return V


@lru_cache(maxsize=4)
def load_boundary_matrix_centered(order):
	if os.path.exists(
		"data/poissonmatrices/poisson_bracket_boundary_centered_{}.npy".format(
			order
		)
	):
		B = onp.load(
			"data/poissonmatrices/poisson_bracket_boundary_centered_{}.npy".format(
				order
			)
		)
	else:
		B = create_poisson_bracket_boundary_matrix_centered(order)
		onp.save(
			"data/poissonmatrices/poisson_bracket_boundary_centered_{}.npy".format(
				order
			),
			B,
		)
	return B


@lru_cache(maxsize=4)
def load_boundary_matrix_upwind(order):
	if os.path.exists(
		"data/poissonmatrices/poisson_bracket_boundary_upwind_{}.npy".format(
			order
		)
	):
		B = onp.load(
			"data/poissonmatrices/poisson_bracket_boundary_upwind_{}.npy".format(
				order
			)
		)
	else:
		B = create_poisson_bracket_boundary_matrix_upwind(order)
		onp.save(
			"data/poissonmatrices/poisson_bracket_boundary_upwind_{}.npy".format(
				order
			),
			B,
		)
	return B


def load_alpha_right_matrix_twice(order):
	if os.path.exists(
		"data/poissonmatrices/alpha_right_matrix_{}.npy".format(order)
	):
		R = onp.load(
			"data/poissonmatrices/alpha_right_matrix_{}.npy".format(order)
		)
	else:
		R = alpha_right_matrix_twice(order)
		onp.save(
			"data/poissonmatrices/alpha_right_matrix_{}.npy".format(order),
			R,
		)
	return R


def load_alpha_top_matrix_twice(order):
	if os.path.exists(
		"data/poissonmatrices/alpha_top_matrix_{}.npy".format(order)
	):
		T = onp.load(
			"data/poissonmatrices/alpha_top_matrix_{}.npy".format(order)
		)
	else:
		T = alpha_top_matrix_twice(order)
		onp.save(
			"data/poissonmatrices/alpha_top_matrix_{}.npy".format(order),
			T,
		)
	return T


def load_zeta_right_minus_matrix_twice(order):
	if os.path.exists(
		"data/poissonmatrices/zeta_right_minus_matrix_{}.npy".format(order)
	):
		Rm = onp.load(
			"data/poissonmatrices/zeta_right_minus_matrix_{}.npy".format(
				order
			)
		)
	else:
		Rm = zeta_right_minus_matrix_twice(order)
		onp.save(
			"data/poissonmatrices/zeta_right_minus_matrix_{}.npy".format(
				order
			),
			Rm,
		)
	return Rm


def load_zeta_right_plus_matrix_twice(order):
	if os.path.exists(
		"data/poissonmatrices/zeta_right_plus_matrix_{}.npy".format(order)
	):
		Rp = onp.load(
			"data/poissonmatrices/zeta_right_plus_matrix_{}.npy".format(
				order
			)
		)
	else:
		Rp = zeta_right_plus_matrix_twice(order)
		onp.save(
			"data/poissonmatrices/zeta_right_plus_matrix_{}.npy".format(
				order
			),
			Rp,
		)
	return Rp


def load_zeta_top_minus_matrix_twice(order):
	if os.path.exists(
		"data/poissonmatrices/zeta_top_minus_matrix_{}.npy".format(order)
	):
		Tm = onp.load(
			"data/poissonmatrices/zeta_top_minus_matrix_{}.npy".format(
				order
			)
		)
	else:
		Tm = zeta_top_minus_matrix_twice(order)
		onp.save(
			"data/poissonmatrices/zeta_top_minus_matrix_{}.npy".format(
				order
			),
			Tm,
		)
	return Tm


def load_zeta_top_plus_matrix_twice(order):
	if os.path.exists(
		"data/poissonmatrices/zeta_top_plus_matrix_{}.npy".format(order)
	):
		Tp = onp.load(
			"data/poissonmatrices/zeta_top_plus_matrix_{}.npy".format(order)
		)
	else:
		Tp = zeta_top_plus_matrix_twice(order)
		onp.save(
			"data/poissonmatrices/zeta_top_plus_matrix_{}.npy".format(
				order
			),
			Tp,
		)
	return Tp


def load_boundary_matrix_inverse_twice(order):
	if os.path.exists(
		"data/poissonmatrices/boundary_matrix_inverse_{}.npy".format(order)
	):
		P_inv = onp.load(
			"data/poissonmatrices/boundary_matrix_inverse_{}.npy".format(
				order
			)
		)
	else:
		P_inv = boundary_matrix_inverse_twice(order)
		onp.save(
			"data/poissonmatrices/boundary_matrix_inverse_{}.npy".format(
				order
			),
			P_inv,
		)
	return P_inv


def load_legendre_boundary_inner_product(order):
	if os.path.exists(
		"data/poissonmatrices/legendre_boundary_inner_product_{}.npy".format(
			order
		)
	):
		boundary_ip = onp.load(
			"data/poissonmatrices/legendre_boundary_inner_product_{}.npy".format(
				order
			)
		)
	else:
		boundary_ip = legendre_boundary_inner_product(order)
		onp.save(
			"data/poissonmatrices/legendre_boundary_inner_product_{}.npy".format(
				order
			),
			boundary_ip,
		)
	return boundary_ip


def load_change_basis_boundary_to_volume(order):
	if os.path.exists(
		"data/poissonmatrices/change_basis_boundary_to_volume_CB_R_{}.npy".format(
			order
		)
	):
		CB_R = onp.load(
			"data/poissonmatrices/change_basis_boundary_to_volume_CB_R_{}.npy".format(
				order
			)
		)
		CB_T = onp.load(
			"data/poissonmatrices/change_basis_boundary_to_volume_CB_T_{}.npy".format(
				order
			)
		)
		CB_L = onp.load(
			"data/poissonmatrices/change_basis_boundary_to_volume_CB_L_{}.npy".format(
				order
			)
		)
		CB_B = onp.load(
			"data/poissonmatrices/change_basis_boundary_to_volume_CB_B_{}.npy".format(
				order
			)
		)
	else:
		CB_R, CB_T, CB_L, CB_B = change_basis_boundary_to_volume(order)
		onp.save(
			"data/poissonmatrices/change_basis_boundary_to_volume_CB_R_{}.npy".format(
				order
			),
			CB_R,
		)
		onp.save(
			"data/poissonmatrices/change_basis_boundary_to_volume_CB_T_{}.npy".format(
				order
			),
			CB_T,
		)
		onp.save(
			"data/poissonmatrices/change_basis_boundary_to_volume_CB_L_{}.npy".format(
				order
			),
			CB_L,
		)
		onp.save(
			"data/poissonmatrices/change_basis_boundary_to_volume_CB_B_{}.npy".format(
				order
			),
			CB_B,
		)
	return CB_R, CB_T, CB_L, CB_B


def load_change_legendre_points_twice(order):
	if os.path.exists(
		"data/poissonmatrices/change_legendre_points_{}.npy".format(order)
	):
		LP = onp.load(
			"data/poissonmatrices/change_legendre_points_{}.npy".format(
				order
			)
		)
	else:
		LP = change_legendre_points_twice(order)
		onp.save(
			"data/poissonmatrices/change_legendre_points_{}.npy".format(
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


"""
Generates code for solving the 2D poisson equation

nabla^2 phi = - zeta

on a rectangular domain (x,y) in ([0, L_x], [0, L_y])

where phi is represented by a continous FE basis
and zeta is represented by a discontinous Legendre basis

The solution phi_j is given by the weak form

(B_{ij} - V_{ij}) phi_j = S_{ij} zeta_j

The continuous solution phi is given by

phi = sum_j phi_j psi_j(x,y)

where psi_j(x,y) are the FE basis functions.
The sum runs over every element in the simulation.
The discontinuous input zeta is given

zeta_k = sum_j zeta_kj alpha_j(x_k,y_k)

where alpha_j are the legendre basis functions.
"""

def get_bottom_indices(order):
	if order == 1 or order == 0:
		return np.asarray([0], dtype=int)
	if order == 2:
		return np.asarray([0, 1, 7], dtype=int)
	if order == 3:
		return np.asarray([0, 1, 2, 10, 11], dtype=int)
	if order == 4:
		return np.asarray([0, 1, 2, 3, 9, 14, 15, 16], dtype=int)
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
	return np.asarray(T_new, dtype=int)


def load_assembly_matrix(nx, ny, order):
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
			nx, ny, order
		)
	):
		num_global_elements = onp.load(
			"data/poissonmatrices/num_global_elements_nx{}_ny{}_order{}.npy".format(
				nx, ny, order
			)
		)
		M = onp.load(
			"data/poissonmatrices/assembly_matrix_nx{}_ny{}_order{}.npy".format(
				nx, ny, order
			)
		)
		T = onp.load(
			"data/poissonmatrices/assembly_matrix_transpose_nx{}_ny{}_order{}.npy".format(
				nx, ny, order
			)
		)
	else:
		num_global_elements, M, T = create_assembly_matrix(nx, ny, order)
		onp.save(
			"data/poissonmatrices/num_global_elements_nx{}_ny{}_order{}.npy".format(
				nx, ny, order
			),
			num_global_elements,
		)
		onp.save(
			"data/poissonmatrices/assembly_matrix_nx{}_ny{}_order{}.npy".format(
				nx, ny, order
			),
			M,
		)
		onp.save(
			"data/poissonmatrices/assembly_matrix_transpose_nx{}_ny{}_order{}.npy".format(
				nx, ny, order
			),
			T,
		)
	return num_global_elements, M, T


def load_elementwise_volume(nx, ny, Lx, Ly, order):
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
		"data/poissonmatrices/elementwise_volume_{}_1.npy".format(order)
	):
		res1 = onp.load(
			"data/poissonmatrices/elementwise_volume_{}_1.npy".format(order)
		)
		res2 = onp.load(
			"data/poissonmatrices/elementwise_volume_{}_2.npy".format(order)
		)
	else:
		res1, res2 = create_elementwise_volume(order)
		onp.save(
			"data/poissonmatrices/elementwise_volume_{}_1".format(order),
			res1,
		)
		onp.save(
			"data/poissonmatrices/elementwise_volume_{}_2".format(order),
			res2,
		)
	V = res1 * (dy / dx) + res2 * (dx / dy)
	return V


def load_elementwise_source(nx, ny, Lx, Ly, order):
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
		"data/poissonmatrices/elementwise_source_{}.npy".format(order)
	):
		res = onp.load(
			"data/poissonmatrices/elementwise_source_{}.npy".format(order)
		)
	else:
		res = write_elementwise_source(order)
		onp.save(
			"data/poissonmatrices/elementwise_source_{}.npy".format(order),
			res,
		)
	return res * dx * dy / 4


def load_volume_matrix(nx, ny, Lx, Ly, order, M, num_global_elements):
	if os.path.exists(
		"data/poissonmatrices/volume_{}_{}_{}.npz".format(nx, ny, order)
	):
		sV = sparse.load_npz(
			"data/poissonmatrices/volume_{}_{}_{}.npz".format(nx, ny, order)
		)
	else:
		V = create_volume_matrix(nx, ny, Lx, Ly, order, M, num_global_elements)
		sV = sparse.csr_matrix(V)
		sparse.save_npz(
			"data/poissonmatrices/volume_{}_{}_{}.npz".format(
				nx, ny, order
			),
			sV,
		)
	return sV


def create_volume_matrix(nx, ny, Lx, Ly, order, M, num_global_elements):
	num_elem = num_elements(order)
	K_elementwise = load_elementwise_volume(nx, ny, Lx, Ly, order)

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
	return np.asarray(K)[:, :, bottom_indices, :]


######
# Poisson solver
######


def get_poisson_solver(nx, ny, Lx, Ly, order):
	N_global_elements, M, T = load_assembly_matrix(nx, ny, order)
	T = convert_to_bottom_indices(T, order)
	S_elem = load_elementwise_source(nx, ny, Lx, Ly, order)

	K = get_kernel(order) @ S_elem

	sV = load_volume_matrix(nx, ny, Lx, Ly, order, M, N_global_elements)
	V_sp = jsparse.BCOO.from_scipy_sparse(sV)
	args = V_sp.data, V_sp.indices, N_global_elements
	kwargs = {"forward": True}
	custom_lu_solve = jit(lambda b: sparse_solve_prim(b, *args, **kwargs), device=jax.devices(backend='cpu')[0])


	platform = xla_bridge.get_backend().platform

	if platform == 'cpu':
		def matrix_solve(b):
			return custom_lu_solve(b)
	elif platform == 'gpu' or platform == 'tpu':
		
		def matrix_solve(b):
			return jax.pure_callback(custom_lu_solve, b, b)
		
	else:
		raise Exception


	def solve(xi):
		xi = xi.at[:, :, 0].add(-np.mean(xi[:, :, 0]))
		xi = np.pad(xi, ((1, 0), (1, 0), (0, 0)), mode="wrap")
		F_ijb = jax.lax.conv_general_dilated(
			xi[None, ...],
			K,
			(1, 1),
			padding="VALID",
			dimension_numbers=("NHWC", "HWOI", "NHWC"),
		)[0]
		b = -F_ijb[T[:, 0], T[:, 1], T[:, 2]]
		res = matrix_solve(b)
		res = res - np.mean(res)
		output = res.at[M].get()
		return output


	return solve


def _scan(sol, x, rk_F):
	"""
	Helper function for jax.lax.scan, which will evaluate f by stepping nt timesteps
	"""
	a, t = sol
	a_f, t_f = rk_F(a, t)
	return (a_f, t_f), None

def _scan_output(sol, x, rk_F):
	"""
	Helper function for jax.scan, same as _scan but outputs data
	"""
	a, t = sol
	a_f, t_f = rk_F(a, t)
	return (a_f, t_f), a_f

def _scan_loss(sol, a_exact, rk_F, f_loss):
	a, t = sol
	a_f, t_f = rk_F(a, t)
	return (a_f, t_f), f_loss(a_f, a_exact)

def simulate_2D(
	a0,
	t0,
	nx,
	ny,
	Lx,
	Ly,
	order,
	dt,
	nt,
	f_poisson_bracket,
	f_phi,
	a_data=None,
	output=False,
	f_diffusion=None,
	f_forcing=None,
	rk=ssp_rk3,
	square_root_loss=False,
	mean_loss=True,
):
	dx = Lx / nx
	dy = Ly / ny
	leg_ip = np.asarray(legendre_inner_product(order))
	denominator = leg_ip * dx * dy


	dadt = lambda a, t: time_derivative_2d_navier_stokes(a, t, f_poisson_bracket, f_phi, denominator, f_forcing=f_forcing, f_diffusion=f_diffusion)
	
	def f_rk(a, t):
		return rk(a, t, dadt, dt)

	def MSE(a, a_exact):
		return np.mean(np.sum((a - a_exact) ** 2 / leg_ip[None, None, :], axis=-1))

	def MSE_sqrt(a, a_exact):
		return np.sqrt(MSE(a, a_exact))

	if square_root_loss:
		loss = MSE_sqrt
	else:
		loss = MSE

	if a_data is not None:
		assert nt == a_data.shape[0]
		@jit
		def scanfloss(sol, a_exact):
			return _scan_loss(sol, a_exact, f_rk, loss)

		(a_f, t_f), loss = scan(scanfloss, (a0, t0), a_data)
		if mean_loss:
			return np.mean(loss)
		else:
			return loss
	else:
		if output:
			scanf = lambda sol, x: _scan_output(sol, x, f_rk)
			_, data = scan(scanf, (a0, t0), None, length=nt)
			return data
		else:
			scanf = lambda sol, x: _scan(sol, x, f_rk)
			(a_f, t_f), _ = scan(scanf, (a0, t0), None, length=nt)
			return (a_f, t_f)






def time_derivative_2d_navier_stokes(
	zeta, t, f_poisson_bracket, f_phi, denominator, f_forcing=None, f_diffusion = None,
):
	phi = f_phi(zeta, t)
	if f_forcing is not None:
		forcing_term = f_forcing(zeta)
	else:
		forcing_term = 0.0
	if f_diffusion is not None:
		diffusion_term = f_diffusion(zeta)
	else:
		diffusion_term = 0.0

	pb_term = f_poisson_bracket(zeta, phi)

	return (
		(pb_term + forcing_term + diffusion_term)
		/ denominator[None, None, :]
	)


def get_inner_fn(step_fn, dt_fn, t_inner, f_poisson_solve):

	def cond_fun(x):
		a, t = x
		return np.logical_and(t < t_inner, np.logical_not(np.isnan(a).any()))

	def body_fun(x):
		a, t = x
		H = f_poisson_solve(a)
		dt = dt_fn(H)
		dt = np.minimum(dt, t_inner - t)
		a_f = step_fn(a, dt, H)
		return (a_f, t + dt)

	@jax.jit
	def inner_fn(a):
		t = 0.0
		x = (a, t)
		a, _ = jax.lax.while_loop(cond_fun, body_fun, x)
		return a

	return inner_fn

def trajectory_fn(inner_fn, steps, carry=True, start_with_input=True):
	if carry:
		def step(carry_in, _):
			carry_out = inner_fn(carry_in)
			frame = carry_in if start_with_input else carry_out
			return carry_out, frame
	else:
		def step(carry_in, _):
			return inner_fn(carry_in), None
	@jax.jit
	def multistep(x_init):
		return jax.lax.scan(step, x_init, xs=None, length=steps)
	return multistep





def plot_DG_basis(
	axs, Lx, Ly, order, zeta, title="", plotting_density=4
):
	nx, ny = zeta.shape[0:2]
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
	
	x_plot = np.linspace(0, Lx, Nx_plot + 1)
	y_plot = np.linspace(0, Ly, Ny_plot + 1)
	pcm = axs.pcolormesh(
		x_plot,
		y_plot,
		output.T,
		shading="flat",
		cmap='jet',
		vmin=-2, 
		vmax=2
	)
	axs.set_xlim([0, Lx])
	axs.set_ylim([0, Ly])
	axs.set_xticks([0, Lx])
	axs.set_yticks([0, Ly])
	axs.set_title(title)
	return pcm











################
# PARAMETERS OF SIMULATION
################

Lx = 1.0
Ly = 1.0
order_exact = 2
order = order_exact
forcing_coefficient = 0.1
runge_kutta = "ssp_rk3"
t0 = 0.0
N_compute_runtime = 5
N_test = 5 # change to 5 or 10

nx_exact = 14
ny_exact = nx_exact
nxs_dg = [7]

t_runtime = 50.0
cfl_safeties = [10.0, 8.0]
cfl_safety_adaptive = 0.28 * (2 * order + 1)
cfl_safety_exact = 3.0
cfl_safety_scaled = [10.0, 10.0]
#cfl_safety_cfd = [40.0, 40.0, 35.0, 20.0, 10.0, 5.0]
Re = 1e3

"""
t_runtime = 30.0
cfl_safeties = 6.0
cfl_safety_adaptive = 0.36 * 5
cfl_safety_exact = 2.0
cfl_safety_scaled = [6.0, 6.0]
cfl_safety_cfd = [40.0, 40.0]
Re = 1e4
"""

viscosity = 1/Re
t_chunk = 1.0
outer_steps = int(t_runtime)


nx_ps_exact = ny_ps_exact = 128
max_velocity = 7.0
nxs_ps_baseline = [8, 16, 32, 64]#, 128, 256]
cfl_safety_cfd_exact = 5.0

key = jax.random.PRNGKey(42)



























def fno_forcing_cfd(grid, dx, dy, scale, offsets=None):
	if offsets == None:
		offsets = grid.cell_faces

	x = grid.mesh(offsets[0])[0]
	y = grid.mesh(offsets[1])[1]

	nx = x.shape[0]
	ny = x.shape[1]

	ff_x = lambda x, y, t: np.cos(2 * PI * (x + dx/2 + y))
	ff_y = lambda x, y, t: np.sin(2 * PI * (x + y + dy/2))
	x_term = inner_prod_with_legendre(nx, ny, Lx, Ly, 0, ff_x, 0.0, n = 8)[...,0] / (dx * dy)
	y_term = inner_prod_with_legendre(nx, ny, Lx, Ly, 0, ff_y, 0.0, n = 8)[...,0] / (dx * dy)


	u = scale / (2 * PI) * grids.GridArray( x_term, offsets[0], grid)
	v = scale / (2 * PI) * grids.GridArray( y_term, offsets[1], grid)

	if grid.ndim == 2:
		f = (u, v)
	else:
		raise NotImplementedError

	def forcing(v):
		del v
		return f
	return forcing


def get_forcing_FNO_ps(nx, ny):
	offsets = None #((0.5, 0.5), (0.5, 0.5))
	dx = Lx / nx
	dy = Ly / ny
	forcing_fn = lambda grid: fno_forcing_cfd(grid, dx, dy, scale = forcing_coefficient, offsets=offsets)
	return forcing_fn



def get_velocity_cfd(u_x, u_y):
	assert u_x.shape == u_y.shape
	bcs = boundaries.periodic_boundary_conditions(2)
  
	grid = grids.Grid(u_x.shape, domain=((0, Lx), (0, Ly)))
	u_x = grids.GridVariable(grids.GridArray(u_x, grid.cell_faces[0], grid=grid), bcs)
	u_y = grids.GridVariable(grids.GridArray(u_y, grid.cell_faces[1], grid=grid), bcs)
	return (u_x, u_y)

def get_trajectory_fn_ps(step_fn, outer_steps):
	rollout_fn = jax.jit(cfd.funcutils.trajectory(step_fn, outer_steps))
	
	def get_rollout(v0):
		_, trajectory = rollout_fn(v0)
		return trajectory
	
	return get_rollout

def concatenate_vorticity(v0, trajectory):
	return np.concatenate((v0[None], trajectory), axis=0)

def concatenate_velocity(u0, trajectory):
	return (np.concatenate((u0[0].data[None], trajectory[0].data),axis=0), np.concatenate((u0[1].data[None], trajectory[1].data),axis=0))


def downsample_ux(u_x, F):
  nx, ny = u_x.shape
  assert nx % F == 0
  return np.mean(u_x[F-1::F,:].reshape(nx // F, ny // F, F), axis=2)
	

def downsample_uy(u_y, F):
  nx, ny = u_y.shape
  assert ny % F == 0
  return np.mean(u_y[:, F-1::F].reshape(nx // F, F, ny // F), axis=1)


def get_dt_cfd(nx, ny, cfl_safety):
	return cfd.equations.stable_time_step(max_velocity, cfl_safety, viscosity, get_grid(nx,ny))

def get_grid(nx, ny):
	return grids.Grid((nx, ny), domain=((0, Lx), (0, Ly)))

def get_inner_steps_dt_cfd(nx, ny, cfl_safety, T):
	inner_steps = int(T // get_dt_cfd(nx, ny, cfl_safety)) + 1
	dt = T / inner_steps
	return inner_steps, dt

def shift_down_left(a):
	return (a + np.roll(a, 1, axis=1) + np.roll(a, 1, axis=0) + np.roll(np.roll(a, 1, axis=0), 1, axis=1)) / 4

def vorticity(u):
	return cfd.finite_differences.curl_2d(u).data


def downsample_u(u0, nx_new):
	ux, uy = u0
	nx_old, ny_old = ux.data.shape
	assert nx_old == ny_old
	factor = nx_old // nx_new
	ux_ds, uy_ds = (downsample_ux(ux.data, factor), downsample_uy(uy.data, factor))
	return get_velocity_cfd(ux_ds, uy_ds)



def downsample(v0_fno, nx, ny):
	
	DSx = int(v0_fno.shape[0] // nx)
	DSy = int(v0_fno.shape[1] // ny)

	return np.mean(np.mean(v0_fno.reshape(-1, ny, DSy), axis=-1).reshape(nx, DSx, ny), axis=1)



def get_ps_step_fn(nx, ny, T, cfl_safety):
	grid = get_grid(nx, ny)
	inner_steps, dt = get_inner_steps_dt_cfd(nx, ny, cfl_safety, T)
	step_fn = spectral.time_stepping.crank_nicolson_rk4(
		spectral.equations.NavierStokes2D(viscosity, grid, drag=0.0, smooth=True, 
			forcing_fn = get_forcing_FNO_ps(nx, ny)), dt)
	return jax.jit(cfd.funcutils.repeated(step_fn, inner_steps))


################
# HELPER FUNCTIONS
################


def compute_percent_error(a1, a2):
	return np.linalg.norm(((a1[:,:,0]-a2[:,:,0]))) / np.linalg.norm((a2[:,:,0]))

def compute_percent_error_ps(a1, a2):
	return np.linalg.norm(((a1-a2))) / np.linalg.norm((a2))


def concatenate_vorticity(v0, trajectory):
	return np.concatenate((v0[None], trajectory), axis=0)

def get_forcing_FNO(order, nx, ny):
	ff = lambda x, y, t: -forcing_coefficient * (np.sin( 2 * np.pi * (x + y) ) + np.cos( 2 * np.pi * (x + y) ))
	y_term = inner_prod_with_legendre(nx, ny, Lx, Ly, order, ff, 0.0, n = 8)
	return lambda zeta: y_term

def get_inner_steps_dt_DG(nx, ny, order, cfl_safety, T):
	dx = Lx / (nx)
	dy = Ly / (ny)
	dt_i = cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)
	inner_steps = int(T // dt_i) + 1
	dt = T / inner_steps
	return inner_steps, dt

def get_dg_step_fn(nx, ny, order, T, cfl_safety):
	flux = Flux.UPWIND
	
	f_poisson_bracket = get_poisson_bracket(order, flux)
	f_poisson_solve = get_poisson_solver(nx, ny, Lx, Ly, order)
	f_phi = lambda zeta, t: f_poisson_solve(zeta)
	f_diffusion = get_diffusion_func(order, Lx, Ly, viscosity)
	f_forcing_sim = get_forcing_FNO(order, nx, ny)
	
	inner_steps, dt = get_inner_steps_dt_DG(nx, ny, order, cfl_safety, T)

	@jit
	def simulate(a_i):
		a, _ = simulate_2D(a_i, t0, nx, ny, Lx, Ly, order, dt, inner_steps, 
						   f_poisson_bracket, f_phi, a_data=None, output=False, f_diffusion=f_diffusion,
							f_forcing=f_forcing_sim, rk=FUNCTION_MAP[runge_kutta])
		return a
	return simulate



def trajectory_fn(inner_fn, steps, start_with_input=True):
	def step(carry_in, _):
		carry_out = inner_fn(carry_in)
		frame = carry_in if start_with_input else carry_out
		return carry_out, frame

	@jax.jit
	def multistep(x_init):
		return jax.lax.scan(step, x_init, xs=None, length=steps)

	return multistep


def get_trajectory_fn(inner_fn, outer_steps, start_with_input=False):
	rollout_fn = trajectory_fn(inner_fn, outer_steps, start_with_input=start_with_input)

	@jax.jit
	def get_rollout(x_init):
		_, trajectory = rollout_fn(x_init)
		return trajectory

	return get_rollout


def get_trajectory_fn_adaptive(adaptive_step_fn, dt_fn, t_inner, outer_steps, f_poisson_solve):
	"""
	adaptive_step_fn should be lambda a, dt: rk(a, F, dt)
	"""
	inner_fn = get_inner_fn(adaptive_step_fn, dt_fn, t_inner, f_poisson_solve)
	traj_fn = get_traj_fn(inner_fn, outer_steps, start_with_input=False)
	return traj_fn

def get_dt_fn(nx, ny, order, cfl, t_runtime):

	dx = Lx / nx
	dy = Ly / ny

	R = load_alpha_right_matrix_twice(order) / dy
	T = load_alpha_top_matrix_twice(order) / dx

	def get_alpha(H):
		alpha_R = H @ R.T
		alpha_T = H @ T.T
		return alpha_R, alpha_T

	def get_max(H):
		alpha_R, alpha_T = get_alpha(H)
		max_R = np.amax(np.abs(alpha_R))
		max_T = np.amax(np.abs(alpha_T))
		return max_R, max_T

	def dt_fn(H):
		max_R, max_T = get_max(H)
		return cfl * (dx * dy) / (max_R * dy + max_T * dx) / (2 * order + 1)

	return dt_fn


def print_runtime():

	a0 = get_initial_condition_FNO()

	for i, nx in enumerate(nxs_dg):
		ny = nx

		a_i = convert_DG_representation(a0, order, 0, nx, ny, Lx, Ly, n=8)

		step_fn = get_dg_step_fn(nx, ny, order, t_runtime, cfl_safety = cfl_safeties[i])
		rollout_fn = get_trajectory_fn(step_fn, 1)


		a_final = rollout_fn(a_i)
		a_final.block_until_ready()

		if np.isnan(a_final).any():
			print("NaN in runtime trajectory")
			raise Exception

		times = onp.zeros(N_compute_runtime)
		for n in range(N_compute_runtime):
			t1 = time()
			a_final = rollout_fn(a_i)
			a_final.block_until_ready()
			t2 = time()

			if np.isnan(a_final).any():
				print("NaN in runtime trajectory")
				raise Exception

			times[n] = t2 - t1


		print("order = {}, t_runtime = {}, nx = {}".format(order, t_runtime, nx))
		print("runtimes: {}".format(times))


def print_runtime_adaptive():
	a0 = get_initial_condition_FNO()
	flux = Flux.UPWIND

	for nx in nxs_dg:
		ny = nx

		a_i = convert_DG_representation(a0, order, 0, nx, ny, Lx, Ly, n=8)

		f_poisson_bracket = get_poisson_bracket(order, flux)
		f_poisson_solve = get_poisson_solver(nx, ny, Lx, Ly, order)
		f_phi = lambda zeta, t: f_poisson_solve(zeta)
		f_diffusion = get_diffusion_func(order, Lx, Ly, viscosity)
		f_forcing_sim = get_forcing_FNO(order, nx, ny)
		dx = Lx / nx
		dy = Ly / ny
		leg_ip = np.asarray(legendre_inner_product(order))
		denominator = leg_ip * dx * dy
		F = lambda a, H: time_derivative_2d_navier_stokes(a, None, f_poisson_bracket, lambda zeta, t: H, denominator, f_forcing=f_forcing_sim, f_diffusion=f_diffusion)
		rk_fn = FUNCTION_MAP["ssp_rk3_adaptive"]
		adaptive_step_fn = lambda a, dt, H: rk_fn(a, F, dt, H, f_poisson_solve)

		dt_fn = get_dt_fn(nx, ny, order, cfl_safety_adaptive, t_runtime)

		rollout_fn_adaptive = get_trajectory_fn_adaptive(adaptive_step_fn, dt_fn, t_runtime, 1, f_poisson_solve)

		a0 = get_initial_condition_FNO()
		a_i = convert_DG_representation(a0, order, 0, nx, ny, Lx, Ly, n=8)

		a_final = rollout_fn_adaptive(a_i)
		a_final.block_until_ready()
		times = onp.zeros(N_compute_runtime)
		for n in range(N_compute_runtime):
			a0 = get_initial_condition_FNO()
			a_i = convert_DG_representation(a0, order, 0, nx, ny, Lx, Ly, n=8)

			t1 = time()
			a_final = rollout_fn_adaptive(a_i)
			a_final.block_until_ready()
			t2 = time()

			times[n] = t2 - t1

		print("order = {}, t_runtime = {}, nx = {}".format(order, t_runtime, nx))
		print("runtimes: {}".format(times))



def print_runtime_scaled():

	a0 = get_initial_condition_FNO()

	for nx in nxs_dg:
		ny = nx

		a_i = convert_DG_representation(a0, order, 0, nx, ny, Lx, Ly, n=8)

		@jax.jit
		def rollout_fn(a0):
			step_fn_one = get_dg_step_fn(nx, ny, order, t_runtime/len(cfl_safety_scaled), cfl_safety=cfl_safety_scaled[0])
			step_fn_two = get_dg_step_fn(nx, ny, order, t_runtime/len(cfl_safety_scaled), cfl_safety=cfl_safety_scaled[1])
			rollout_fn_one = get_trajectory_fn(step_fn_one, 1)
			rollout_fn_two = get_trajectory_fn(step_fn_two, 1)
			a_one = rollout_fn_one(a0)[-1]
			assert a_one.shape == a0.shape
			a_two = rollout_fn_two(a_one)
			return a_two

		a_final = rollout_fn(a_i)
		a_final.block_until_ready()
		times = onp.zeros(N_compute_runtime)
		for n in range(N_compute_runtime):
			t1 = time()
			a_final = rollout_fn(a_i)
			a_final.block_until_ready()
			t2 = time()
			times[n] = t2 - t1

		print("order = {}, t_runtime = {}, nx = {}".format(order, t_runtime, nx))
		print("runtimes: {}".format(times))



def print_errors():
	errors = onp.zeros((len(nxs_dg), outer_steps+1))

	for _ in range(N_test):
		a0 = get_initial_condition_FNO()

		a_i = convert_DG_representation(a0, order_exact, 0, nx_exact, ny_exact, Lx, Ly, n=8)
		exact_step_fn = get_dg_step_fn(nx_exact, ny_exact, order_exact, t_chunk, cfl_safety = cfl_safety_exact)
		exact_rollout_fn = get_trajectory_fn(exact_step_fn, outer_steps)
		exact_trajectory = exact_rollout_fn(a_i)
		exact_trajectory = concatenate_vorticity(a_i, exact_trajectory)

		if np.isnan(exact_trajectory).any():
			print("NaN in exact trajectory")
			raise Exception


		for n, nx in enumerate(nxs_dg):
			ny = nx

			a_i = convert_DG_representation(a0, order, 0, nx, ny, Lx, Ly, n=8)
			step_fn = get_dg_step_fn(nx, ny, order, t_chunk, cfl_safety = cfl_safeties[n])
			rollout_fn = get_trajectory_fn(step_fn, outer_steps)
			trajectory = rollout_fn(a_i)
			trajectory = concatenate_vorticity(a_i, trajectory)

			if np.isnan(trajectory).any():
				print("NaN in trajectory for nx={}")
				raise Exception

			for j in range(outer_steps+1):
				a_ex = convert_DG_representation(exact_trajectory[j], order, order_exact, nx, ny, Lx, Ly, n=8)
				errors[n, j] += compute_percent_error(trajectory[j], a_ex) / N_test


	for i, nx in enumerate(nxs_dg):
		print("nx = {}, errors = {}".format(nx, np.mean(errors[i])))



def print_errors_adaptive():
	flux = Flux.UPWIND
	errors = onp.zeros((len(nxs_dg), outer_steps+1))

	for _ in range(N_test):
		a0 = get_initial_condition_FNO()
		a_i = convert_DG_representation(a0, order_exact, 0, nx_exact, ny_exact, Lx, Ly, n=8)
		
		exact_step_fn = get_dg_step_fn(nx_exact, ny_exact, order_exact, t_chunk, cfl_safety = cfl_safety_exact)
		exact_rollout_fn = get_trajectory_fn(exact_step_fn, outer_steps)
		exact_trajectory = exact_rollout_fn(a_i)
		exact_trajectory = concatenate_vorticity(a_i, exact_trajectory)

		if np.isnan(exact_trajectory).any():
			print("NaN in exact trajectory")
			raise Exception


		for n, nx in enumerate(nxs_dg):
			ny = nx

			a_i = convert_DG_representation(a0, order, 0, nx, ny, Lx, Ly, n=8)

			f_poisson_bracket = get_poisson_bracket(order, flux)
			f_poisson_solve = get_poisson_solver(nx, ny, Lx, Ly, order)
			f_phi = lambda zeta, t: f_poisson_solve(zeta)
			f_diffusion = get_diffusion_func(order, Lx, Ly, viscosity)
			f_forcing_sim = get_forcing_FNO(order, nx, ny)
			dx = Lx / nx
			dy = Ly / ny
			leg_ip = np.asarray(legendre_inner_product(order))
			denominator = leg_ip * dx * dy
			F = lambda a, H: time_derivative_2d_navier_stokes(a, None, f_poisson_bracket, lambda zeta, t: H, denominator, f_forcing=f_forcing_sim, f_diffusion=f_diffusion)
			rk_fn = FUNCTION_MAP["ssp_rk3_adaptive"]
			adaptive_step_fn = lambda a, dt, H: rk_fn(a, F, dt, H, f_poisson_solve)

			dt_fn = get_dt_fn(nx, ny, order, cfl_safety_adaptive, t_runtime)

			rollout_fn_adaptive = get_trajectory_fn_adaptive(adaptive_step_fn, dt_fn, t_chunk, outer_steps, f_poisson_solve)

			trajectory = rollout_fn_adaptive(a_i)
			trajectory = concatenate_vorticity(a_i, trajectory)

			if np.isnan(trajectory).any():
				print("NaN in trajectory for nx={}".format(nx))
				raise Exception

			for j in range(outer_steps+1):
				a_ex = convert_DG_representation(exact_trajectory[j], order, order_exact, nx, ny, Lx, Ly, n=8)
				errors[n, j] += compute_percent_error(trajectory[j], a_ex) / N_test


	print("nxs: {}".format(nxs_dg))
	print("Mean errors: {}".format(np.mean(errors, axis=-1)))




def print_errors_scaled():
	errors = onp.zeros((len(nxs_dg), outer_steps+1))

	for _ in range(N_test):
		a0 = get_initial_condition_FNO()

		a_i = convert_DG_representation(a0, order_exact, 0, nx_exact, ny_exact, Lx, Ly, n=8)
		exact_step_fn = get_dg_step_fn(nx_exact, ny_exact, order_exact, t_chunk, cfl_safety = cfl_safety_exact)
		exact_rollout_fn = get_trajectory_fn(exact_step_fn, outer_steps)
		exact_trajectory = exact_rollout_fn(a_i)
		exact_trajectory = concatenate_vorticity(a_i, exact_trajectory)

		if np.isnan(exact_trajectory).any():
			print("NaN in exact trajectory")
			raise Exception


		for n, nx in enumerate(nxs_dg):
			ny = nx

			a_i = convert_DG_representation(a0, order, 0, nx, ny, Lx, Ly, n=8)
			step_fn_one = get_dg_step_fn(nx, ny, order, t_chunk/2, cfl_safety=cfl_safety_scaled[0])
			step_fn_two = get_dg_step_fn(nx, ny, order, t_chunk/2, cfl_safety=cfl_safety_scaled[1])

			@jax.jit
			def step_fn(a0):
				a_one = step_fn_one(a0)
				return step_fn_two(a_one)

			rollout_fn = get_trajectory_fn(step_fn, outer_steps)

			trajectory = rollout_fn(a_i)
			trajectory = concatenate_vorticity(a_i, trajectory)

			if np.isnan(trajectory).any():
				print("NaN in trajectory for nx={}")
				raise Exception

			for j in range(outer_steps+1):
				a_ex = convert_DG_representation(exact_trajectory[j], order, order_exact, nx, ny, Lx, Ly, n=8)
				errors[n, j] += compute_percent_error(trajectory[j], a_ex) / N_test
	print("nxs: {}".format(nxs_dg))
	print("Mean errors: {}".format(np.mean(errors, axis=-1)))



def print_runtime_ps():
	# compute runtime

	for i, nx in enumerate(nxs_ps_baseline):
		ny = nx

		cfl_safety = cfl_safety_cfd[i]
		step_fn = get_ps_step_fn(nx, ny, t_runtime, cfl_safety)
		rollout_fn = jit(get_trajectory_fn_ps(step_fn, 1))
		

		v0_fno = get_initial_condition_FNO()
		v0 = downsample(v0_fno[...,0], nx, ny)
		v_hat0 = np.fft.rfftn(v0)

		trajectory_hat = rollout_fn(v_hat0).block_until_ready()

		if np.isnan(trajectory_hat).any():
			print("NaN in exact trajectory")
			raise Exception

		times = onp.zeros(N_compute_runtime)
		for n in range(N_compute_runtime):

			v0_fno = get_initial_condition_FNO()
			v0 = downsample(v0_fno[...,0], nx, ny)
			v_hat0 = np.fft.rfftn(v0)

			t1 = time()
			trajectory_hat = rollout_fn(v_hat0).block_until_ready()
			t2 = time()
			times[n] = t2 - t1


			if np.isnan(trajectory_hat).any():
				print("NaN in exact trajectory")
				raise Exception

		print("ML-CFD PS baseline, nx = {}, times={}".format(nx, times))
		


def print_errors_ps():
	errors = onp.zeros((len(nxs_ps_baseline), outer_steps+1))
	

	step_fn_exact = get_ps_step_fn(nx_ps_exact, ny_ps_exact, t_chunk, cfl_safety_cfd_exact)
	rollout_fn_exact = jit(get_trajectory_fn_ps(step_fn_exact, outer_steps))

	for _ in range(N_test):

		v0_fno = get_initial_condition_FNO()[...,0]

		v0_ex = downsample(v0_fno, nx_ps_exact, ny_ps_exact)
		v_hat0_ex = np.fft.rfftn(v0_ex)
		trajectory_hat_ps_ex = rollout_fn_exact(v_hat0_ex)
		trajectory_hat_ps_ex = concatenate_vorticity(v_hat0_ex, trajectory_hat_ps_ex)
		trajectory_ps_ex = np.fft.irfftn(trajectory_hat_ps_ex, axes=(1,2))
		




		if np.isnan(trajectory_ps_ex).any():
			print("NaN in exact trajectory")
			raise Exception


		for i, nx in enumerate(nxs_ps_baseline):
			ny = nx
			cfl_safety = cfl_safety_cfd[i]
			step_fn = get_ps_step_fn(nx, ny, t_chunk, cfl_safety)
			rollout_fn = jit(get_trajectory_fn_ps(step_fn, outer_steps))

			v0 = downsample(v0_fno, nx, ny)
			v_hat0 = np.fft.rfftn(v0)
			trajectory_hat_ps = rollout_fn(v_hat0)
			if np.isnan(trajectory_hat_ps).any():
				print("NaN in trajectory nx = {}".format(nx))
				raise Exception
			trajectory_ps = np.fft.irfftn(trajectory_hat_ps, axes=(1,2))
			trajectory_ps = concatenate_vorticity(v0, trajectory_ps)

			for j in range(outer_steps+1):
				a_ex = downsample(trajectory_ps_ex[j], nx, ny)[..., None]
				errors[i, j] += compute_percent_error(trajectory_ps[j][...,None], a_ex) / N_test

	print("nxs: {}".format(nxs_ps_baseline))
	print("Mean errors: {}".format(np.mean(errors, axis=-1)))
	for i, nx in enumerate(nxs_ps_baseline):
		print(errors[i])







"""
print_errors_ps()
print_runtime_ps()
"""

"""
# plot
v0_fno = get_initial_condition_FNO()

nx_dg = 16
ny_dg = 16

a_i = convert_DG_representation(v0_fno, order, 0, nx_dg, ny_dg, Lx, Ly, n=8)
step_fn = get_dg_step_fn(nx_dg, ny_dg, order, 3.0, cfl_safety = cfl_safeties[0])
rollout_fn = get_trajectory_fn(step_fn, 3)
traj = rollout_fn(-a_i)
traj = concatenate_vorticity(a_i, -traj)

for i, nx in enumerate(nxs_ps_baseline):
	ny = nx

	cfl_safety = cfl_safety_cfd[i]
	v0 = downsample(v0_fno, nx, ny)
	v_hat0 = np.fft.rfftn(v0)
	step_fn = get_ps_step_fn(nx, ny, 3.0, cfl_safety)
	rollout_fn = get_trajectory_fn_ps(step_fn, 3)
	trajectory_hat = rollout_fn(v_hat0)
	trajectory_hat_ps = concatenate_vorticity(v_hat0, trajectory_hat)
	trajectory_ps = np.fft.irfftn(trajectory_hat_ps, axes=(1,2))



	fig, axs = plt.subplots(2,4,figsize=(11, 6))

	plot_DG_basis(axs[0,0], Lx, Ly, 0, trajectory_ps[0][:,:,None], title="PS Trajectory t=0")
	plot_DG_basis(axs[0,1], Lx, Ly, 0, trajectory_ps[1][:,:,None], title="PS Trajectory t=1")
	plot_DG_basis(axs[0,2], Lx, Ly, 0, trajectory_ps[2][:,:,None], title="PS Trajectory t=2")
	plot_DG_basis(axs[0,3], Lx, Ly, 0, trajectory_ps[3][:,:,None], title="PS Trajectory t=3")
	plot_DG_basis(axs[1,0], Lx, Ly, order, traj[0], title="DG Trajectory t=0")
	plot_DG_basis(axs[1,1], Lx, Ly, order, traj[1], title="DG Trajectory t=1")
	plot_DG_basis(axs[1,2], Lx, Ly, order, traj[2], title="DG Trajectory t=2")
	plot_DG_basis(axs[1,3], Lx, Ly, order, traj[3], title="DG Trajectory t=3")



	plt.show()


"""



device = xla_bridge.get_backend().platform
print(device)
print("nu is {}".format(viscosity))

#print_runtime_scaled()
#print_runtime_adaptive()
print_runtime()
#print_errors_scaled()
#print_errors_adaptive()
print_errors()






