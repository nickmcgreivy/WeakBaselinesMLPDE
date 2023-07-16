import torch
import math
import jax.numpy as np
import jax
from jax import config, grad, vmap
config.update("jax_enable_x64", True)

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
	return np.asarray(GRF.sample(1)[0][:,:,None])



def f_init_MLCFD(key):

	Lx = 2 * PI
	Ly = 2 * PI

	max_k = 5
	min_k = 1
	num_init_modes = 6
	amplitude_max = 4.0

	def sum_modes(x, y, amplitudes, ks_x, ks_y, phases_x, phases_y):
		return np.sum(
			amplitudes[None, :]
			* np.sin(
				ks_x[None, :] * 2 * PI / Lx * x[:, None] + phases_x[None, :]
			) * np.sin(
				ks_y[None, :] * 2 * PI / Ly * y[:, None] + phases_y[None, :]
			),
			axis=1,
		)

	key1, key2, key3, key4, key5 = jax.random.split(key, 5)
	phases_x = jax.random.uniform(key1, (num_init_modes,)) * 2 * PI
	phases_y = jax.random.uniform(key2, (num_init_modes,)) * 2 * PI
	ks_x = jax.random.randint(
		key3, (num_init_modes,), min_k, max_k
	)
	ks_y = jax.random.randint(
		key4, (num_init_modes,), min_k, max_k
	)
	amplitudes = jax.random.uniform(key5, (num_init_modes,)) * amplitude_max
	return lambda x, y, t: sum_modes(x, y, amplitudes, ks_x, ks_y, phases_x, phases_y)


def f_init_CNO(key):

	rho = 0.1

	key1, key2, = jax.random.split(key,2)

	num_modes = 10
	alpha_k = jax.random.uniform(key1, (num_modes,))
	beta_k = jax.random.uniform(key2, (num_modes,)) * 2 * PI
	k = np.arange(1,num_modes+1,num_modes)

	def sigma(x):
		""" assumes that x is a scalar """
		return np.sum(alpha_k * np.sin(2 * PI * k * x - beta_k))

	def chi_0(x, y, t):
		y_below = (y + sigma(x) < 0.5).astype(int)
		return - y_below * (1 - np.tanh(2 * PI * (y - 0.25) / rho)**2) + (1 - y_below) * (1 - np.tanh(2 * PI * (0.75 - y) / rho)**2)
	

	return vmap(chi_0, (0, 0, None), 0)

