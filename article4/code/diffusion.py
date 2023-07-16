import jax.numpy as np
from sympy import legendre, diff, integrate, symbols
from functools import lru_cache

from sympy.matrices import Matrix, zeros
from scipy.special import eval_legendre
from basisfunctions import num_elements, legendre_poly

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
	Q = np.asarray(get_Q_right(order), dtype=float)
	
	def b_right(zeta):
		a = np.concatenate((zeta, np.roll(zeta, -1, axis=0)), axis=-1)
		return a @ Q.T

	return b_right

def get_b_top(order):
	Q = np.asarray(get_Q_top(order), dtype=float)

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

	A_R = np.asarray(get_A_right(order), dtype=float) * (Ly/Lx)
	A_T = np.asarray(get_A_top(order), dtype=float) * (Lx/Ly)
	A_L = np.asarray(get_A_left(order), dtype=float) * (Ly/Lx)
	A_B = np.asarray(get_A_bottom(order), dtype=float) * (Lx/Ly)

	f_b_right = get_b_right(order)
	f_b_top = get_b_top(order)
	#f_b_left = get_b_right(order)
	#f_b_bottom = get_b_bottom(order)

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
	V = np.asarray(get_V(order, Lx, Ly), dtype=float)

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