import sys
import matplotlib.pyplot as plt
from itertools import cycle
import jax.numpy as np
import h5py
from jax import vmap
import jax
from legendre import generate_legendre
from helper import evalf_1D_right, evalf_1D_left, _fixed_quad, integrate_abs_derivative, evalf_1D
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from legendre import generate_legendre


def get_linestyle(k):
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    return linestyles[k % len(linestyles)]


def plot_subfig(
    a, subfig, L, color="blue", linewidth=0.5, linestyle="solid", label=None
):
    def evalf(x, a, j, dx, leg_poly):
        x_j = dx * (0.5 + j)
        xi = (x - x_j) / (0.5 * dx)
        vmap_polyval = vmap(np.polyval, (0, None), -1)
        poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
        return np.sum(poly_eval * a, axis=-1)

    nx = a.shape[0]
    p = a.shape[1]
    dx = L / nx
    xjs = np.arange(nx) * L / nx
    xs = xjs[None, :] + np.linspace(0.0, dx, 10)[:, None]
    vmap_eval = vmap(evalf, (1, 0, 0, None, None), 1)
    subfig.plot(
        xs,
        vmap_eval(xs, a, np.arange(nx), dx, generate_legendre(p)),
        color=color,
        linewidth=linewidth,
        label=label,
        linestyle=linestyle,
    )
    return

def plot_trajectory(trajectory, L, T):

    def evalf(x, a, j, dx, leg_poly):
        x_j = dx * (0.5 + j)
        xi = (x - x_j) / (0.5 * dx)
        vmap_polyval = vmap(np.polyval, (0, None), -1)
        poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
        return np.sum(poly_eval * a, axis=-1)

    nx = trajectory.shape[1]
    p = trajectory.shape[2]
    dx = L / nx
    n_plot = 200
    xjs = np.arange(n_plot+1) * L / n_plot

    im_f = vmap(vmap(lambda x, a: evalf_1D(x, a, dx, generate_legendre(p)), (0, None)), (None, 0))
    image = im_f(xjs, trajectory)
    plt.imshow(image, aspect='auto', origin='lower', interpolation='none', extent=[-L/2,L/2,0,T])
