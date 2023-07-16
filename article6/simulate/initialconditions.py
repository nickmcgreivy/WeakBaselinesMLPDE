import jax.numpy as jnp
import jax
from jax import random

from legendre import generate_legendre
from helper import map_f_to_FV

PI = jnp.pi


def get_a0(f_init, core_params, nx):
    return get_a(f_init, 0.0, core_params, nx)


def get_a(f_init, t, core_params, nx):
    dx = core_params.Lx / nx
    return map_f_to_FV(f_init, t, nx, dx, n=8)


def f_init_sum_of_amplitudes_burgers(
    Lx,
    key=jax.random.PRNGKey(0),
    min_num_modes=1,
    max_num_modes=6,
    min_k=0,
    max_k=3,
    amplitude_max=1.0,
):
    key1, key2, key3, key4 = random.split(key, 4)
    phases = random.uniform(key1, (max_num_modes,)) * 2 * PI
    ks = random.randint(key3, (max_num_modes,), min_k, max_k + 1)
    num_nonzero_modes = random.randint(key2, (1,), min_num_modes, max_num_modes + 1)[0]
    mask = jnp.arange(max_num_modes) < num_nonzero_modes
    amplitudes = jax.random.uniform(key4, (max_num_modes,)) * amplitude_max
    amplitudes = amplitudes * mask

    def sum_modes(x, t):
        return jnp.sum(
            amplitudes[None, :]
            * jnp.sin(ks[None, :] * 2 * PI / Lx * x[:, None] + phases[None, :]),
            axis=1,
        )

    return sum_modes


def forcing_func_sum_of_modes(
    Lx,
    min_num_modes=20,
    max_num_modes=20,
    min_k=3,
    max_k=6,
    amplitude_max=0.5,
    omega_max=0.4,
):
    def f_forcing(key):
        key1, key2, key3, key4, key5 = random.split(key, 5)
        phases = random.uniform(key1, (max_num_modes,)) * 2 * PI
        ks = random.randint(key3, (max_num_modes,), min_k, max_k + 1)
        num_nonzero_modes = random.randint(
            key2, (1,), min_num_modes, max_num_modes + 1
        )[0]
        mask = jnp.arange(max_num_modes) < num_nonzero_modes
        amplitudes = (
            2 * (jax.random.uniform(key4, (max_num_modes,)) - 0.5) * amplitude_max
        )
        amplitudes = amplitudes * mask

        omegas = (random.uniform(key5, (max_num_modes,)) - 0.5) * 2 * omega_max

        def sum_modes(x, t):
            return jnp.sum(
                amplitudes[None, :]
                * jnp.sin(
                    2 * PI * ks[None, :] / Lx * x[:, None]
                    + omegas[None, :] * t
                    + phases[None, :]
                ),
                axis=1,
            )

        return sum_modes

    return f_forcing


def get_initial_condition_fn(core_params, ic_string, **kwargs):
    Lx = core_params.Lx

    def f_init_zeros(x, t):
        return jnp.zeros(x.shape)

    def f_sawtooth(x, t):
        return 1 - 4 * jnp.abs(((x - t) % Lx) / Lx - 1 / 2)

    def f_sin(x, t):
        return -jnp.cos(2 * PI / Lx * (x - t))

    def f_gaussian(x, t):
        return jnp.exp(-32 * (((x - t) % Lx) - Lx / 2) ** 2 / (Lx**2))

    if ic_string == "zero" or ic_string == "zeros":
        return f_init_zeros
    elif ic_string == "sin_wave" or ic_string == "sin":
        return f_sin
    elif ic_string == "sum_sin":
        return f_init_sum_of_amplitudes_burgers(Lx, **kwargs)
    elif ic_string == "sawtooth":
        return f_sawtooth
    elif ic_string == "gaussian":
        return f_gaussian
    else:
        raise NotImplementedError
