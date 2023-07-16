import jax.numpy as jnp
import jax.experimental.host_callback as hcb
import jax
from time import time
import numpy as onp

fn_sin = lambda x: onp.sin(x)

@jax.jit
def f_cb(x):
	y = x**2 - 2 * x + 11
	return jax.pure_callback(fn_sin, y, y)
	#return hcb.call(fn_sin, y, result_shape=y)

@jax.jit
def f(x):
	y = x**2 - 2 * x + 11
	return jnp.sin(x)

N = 100
nx = 10

@jax.jit
def create():
	x = jnp.linspace(0, 100, nx)
	return f(x)

@jax.jit
def create_cb():
	x = jnp.linspace(0, 100,  nx)
	return f_cb(x)

_ = create().block_until_ready()
_ = create_cb().block_until_ready()
t0 = time()
for j in range(N):
	_ = create().block_until_ready()
t1 = time()
for j in range(N):
	_ = create_cb().block_until_ready()
t2 = time()
print("Average time to run normal function: {}".format((t1 - t0)/N))
print("Average time to run callback function: {}".format((t2 - t1)/N))
