import jax
import jax.numpy as jnp


def get_inner_fn(step_fn, dt_fn, t_inner):
    def cond_fun(x, t0):
        a, t = x
        return jnp.logical_and(t < t0 + t_inner, jnp.logical_not(jnp.isnan(a).any()))

    def body_fun(x, t0):
        a, t = x
        dt = jnp.minimum(dt_fn(a), t0 + t_inner - t)
        a_f, t_f = step_fn(a, t, dt)
        return (a_f, t_f)

    @jax.jit
    def inner_fn(x):
        a, t = x
        x = jax.lax.while_loop(lambda x: cond_fun(x, t), lambda x: body_fun(x, t), x)
        a, t = x
        return x

    return inner_fn


def trajectory_fn(inner_fn, steps, start_with_input=True):
    def step(carry_in, _):
        carry_out = inner_fn(carry_in)
        frame = carry_in if start_with_input else carry_out
        return carry_out, frame

    @jax.jit
    def multistep(x_init):
        return jax.lax.scan(step, x_init, xs=None, length=steps)

    return multistep


def get_trajectory_fn(inner_fn, outer_steps, start_with_input=True):
    rollout_fn = trajectory_fn(inner_fn, outer_steps, start_with_input=start_with_input)

    def get_rollout(x_init):
        _, trajectory = rollout_fn(x_init)
        return trajectory

    return get_rollout
