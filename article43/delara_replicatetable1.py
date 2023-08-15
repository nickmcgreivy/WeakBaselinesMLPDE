import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import jit, vmap
from jax.lax import scan
from functools import partial
from time import time
# again, this only works on startup!
from jax import config
config.update("jax_enable_x64", True)

import sys
sys.path.append("simcode")
sys.path.append("analysiscode")
from helper import map_f_to_DG, get_dx, get_dt_nt, evalf_1D
from legendre import generate_legendre
from plot_data import plot_subfig, plot_trajectory
from rungekutta import ssp_rk3
from flux import Flux
from timederivative import time_derivative_DG_1D_burgers, pad_dirichlet

PI = np.pi

T = 100.0 # 1.0
L = 2.0
t0 = 0.0
nu = 0.1
u_max = 1.5
cfl = 0.2
safety_diff = 0.25
cfl_exact = 0.02
flux = Flux.GODUNOV
f_init = lambda x, t: jnp.ones(x.shape) # lambda x, t: jnp.sin(2 * PI * x / L)
bc = 'dirichlet' # 'periodic'
fL = lambda t: 1 + 0.5 * jnp.sin(10 * t)
fR = lambda t: 1.0


"""
p = 10
nx = 6
dx = get_dx(L, nx)
leg_poly = generate_legendre(p)
a0 = map_f_to_DG(f_init, t0, p, nx, dx, leg_poly, n=8)

@partial(jit, static_argnums=(3,))
def simulate(a0, t0, dt, nt):
	def _scan(sol, x, rk_F):
		a, t = sol
		a_f, t_f = rk_F(a, t)
		return (a_f, t_f), None

	dadt = lambda a, t: time_derivative_DG_1D_burgers(a, t, flux, dx, leg_poly, nu=nu, bc=bc, fL=fL, fR=fR)
	rk_F = lambda a, t: ssp_rk3(a, t, dadt, dt)
	scanf = jit(lambda sol, x: _scan(sol, x, rk_F))
	(a_f, t_f), _ = scan(scanf, (a0, t0), None, length=nt)
	return (a_f, t_f)

########################
# Simple test plotting
########################
dt, nt = get_dt_nt(dx, p, nu, u_max, cfl, T, safety_diff=safety_diff)
af, tf = simulate(a0, t0, dt, nt)



fig, axs = plt.subplots(1, 2, figsize=(7, 3), squeeze=True)
plot_subfig(a0, axs[0], L, color="blue", linewidth=0.5, linestyle="solid", label=None)
plot_subfig(af, axs[1], L, color="blue", linewidth=0.5, linestyle="solid", label=None)
plt.show()





########################
# Plot function in 2D
########################

T_plot = 1000
T_chunk = T / T_plot
dt, nt = get_dt_nt(dx, p, nu, u_max, cfl, T_chunk, safety_diff=safety_diff)
trajectory = np.zeros((T_plot+1, nx, p))
trajectory[0] = a0
a = a0
t = t0

for j in range(T_plot):
	a, t = simulate(a, t, dt, nt)
	trajectory[j+1] = a

plot_trajectory(trajectory, L, T)
plt.show()

"""

########################
# Plot error_max as function of nx, p
########################

@partial(jit, static_argnums=(3,))
def simulate(a0, t0, dt, nt, leg_poly):
	def _scan(sol, x, rk_F):
		a, t = sol
		a_f, t_f = rk_F(a, t)
		return (a_f, t_f), None

	dx = L / a0.shape[0]
	dadt = lambda a, t: time_derivative_DG_1D_burgers(a, t, flux, dx, leg_poly, nu=nu, bc=bc, fL=fL, fR=fR)
	rk_F = lambda a, t: ssp_rk3(a, t, dadt, dt)
	scanf = jit(lambda sol, x: _scan(sol, x, rk_F))
	(a_f, t_f), _ = scan(scanf, (a0, t0), None, length=nt)
	return (a_f, t_f)

####
# Exact solution
####

N_runtime = 5
p_exact = 3
nx_exact = 160
dx_exact = get_dx(L, nx_exact)
leg_poly_exact = generate_legendre(p_exact)
a0_exact = map_f_to_DG(f_init, t0, p_exact, nx_exact, dx_exact, leg_poly_exact, n=8)
dt, nt = get_dt_nt(dx_exact, p_exact, nu, u_max, cfl_exact, T, safety_diff=safety_diff)
af_exact, _ = simulate(a0_exact, t0, dt, nt, leg_poly_exact)

####
# Loop through plots
####
ps = [1, 2, 3, 4, 6, 8, 10] #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
nxs = [[40,80,320],[16, 48, 64],[6,12,24],[4,8,12,16],[2,3,4,5,8],[1,2,3,5],[1,3,4]] #[[40, 80, 160, 320], [16,32,48,64,96], [6,8,12,16,24,32], [4,6,8,10,12,16,24], [3,4,5,6,8,10], [2,3,4,5,6,8], [1,2,3,4,5,6], [1,2,3,4,5,6], [1,2,3,4,5], [1,2,3,4,5]]
runtimes = [[],[],[],[],[],[],[]]  # [[],[],[],[],[],[],[],[],[],[]]
max_errors = [[],[],[],[],[],[],[]] # [[],[],[],[],[],[],[],[],[],[]]
safeties = [1.0, 1.0, 1.0, 0.85, 0.45, 0.3, 0.2] #[1.0, 1.0, 1.0, 0.9, 0.75, 0.5, 0.4, 0.35, 0.3, 0.25]

for i, p in enumerate(ps):

	leg_poly = generate_legendre(p)
	safety_diff = safeties[i]

	for nx in nxs[i]:
		print("p={}, nx={}".format(p-1, nx))
		
		dx = get_dx(L, nx)
		a0 = map_f_to_DG(f_init, t0, p, nx, dx, leg_poly, n=8)

		####
		# test runtime
		####

		dt, nt = get_dt_nt(dx, p, nu, u_max, cfl, T, safety_diff=safety_diff)
		af, tf = simulate(a0, t0, dt, nt, leg_poly)
		af.block_until_ready()

		ti = time()
		for _ in range(N_runtime):
			af, tf, = simulate(a0, t0, dt, nt, leg_poly)
			af.block_until_ready()
		tf = time()

		runtimes[i].append((tf - ti) / N_runtime)

		####
		# Compute error
		####
		N_test_error = 1000
		x_test_error = jnp.linspace(0, L, N_test_error+1)[:-1]
		eval_f_test = vmap(evalf_1D, (0, None, None, None))

		solution_exact = eval_f_test(x_test_error, af_exact, dx_exact, leg_poly_exact)
		solution = eval_f_test(x_test_error, af, dx, leg_poly)
		
		#plt.plot(solution)
		#plt.plot(solution_exact)
		#plt.show()

		max_error = float(jnp.max(jnp.abs(solution_exact - solution)))
		max_errors[i].append(max_error)
		#print(max_error)

		
		#fig, axs = plt.subplots(1, 1, figsize=(7, 3), squeeze=True)
		#plot_subfig(af_exact, axs, L, color="blue")
		#plot_subfig(af, axs, L, color="red")
		
		#plt.show()
		


for i, p in enumerate(ps):
	print("p = {}".format(p-1))
	print("runtimes: {}".format(runtimes[i]))
	print("errors: {}".format(max_errors[i]))

for i, p in enumerate(ps):
	plt.loglog(runtimes[i], max_errors[i], marker='.', markersize=8, linewidth=0.5, label="p={}".format(p-1))
plt.legend()
plt.xlabel("Runtime")
plt.ylabel("Maximum error")
plt.show()



for i, p in enumerate(ps):
	nx_plot = [p * nx for nx in nxs[i]]
	plt.loglog(nx_plot, max_errors[i], marker='.', markersize=8, linewidth=0.5, label="p={}".format(p-1))
plt.legend()
plt.xlabel("Degrees of freedom")
plt.ylabel("Maximum error")
plt.show()
