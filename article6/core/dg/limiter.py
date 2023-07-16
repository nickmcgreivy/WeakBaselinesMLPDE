import jax.numpy as np


def minmod(r):
    return np.maximum(0, np.minimum(1, r))


def minmod2(z1, z2):
    s = 0.5 * (np.sign(z1) + np.sign(z2))
    return s * np.minimum(np.absolute(z1), np.absolute(z2))


def minmod3(z1, z2, z3):
    s = (
        0.5
        * (np.sign(z1) + np.sign(z2))
        * np.absolute(0.5 * ((np.sign(z1) + np.sign(z3))))
    )
    return s * np.minimum(np.absolute(z1), np.minimum(np.absolute(z2), np.absolute(z3)))


def min_mod_limiter(a):
    """
    a is (nx, p)
    """
    nx, p = a.shape
    if p == 1:
        return a
    u_dp = np.roll(a[:, 0], -1) - a[:, 0]
    u_dm = a[:, 0] - np.roll(a[:, 0], 1)
    if p == 2:
        return a.at[:, 1].set(minmod3(a[:, 1], u_dp, u_dm))
    else:
        alt = (-1) ** np.arange(p - 1)
        u_t = np.sum(a[:, 1:], axis=1)
        u_tt = np.sum(a[:, 1:] * alt, axis=1)

        u_t_mod = minmod3(u_t, u_dp, u_dm)
        u_tt_mod = minmod3(u_tt, u_dp, u_dm)
        b = np.concatenate((u_t_mod[:, None], u_tt_mod[:, None]), axis=1)

        A = 0.5 * np.asarray([[1, -1], [1, 1]])
        a = a.at[:, 1:3].set(b @ A)
        if p > 3:
            a = a.at[:, 3:].set(0.0)
        return a


# Old Limiter
"""
def minmodlimit(a):
    still_limiting = np.ones(a.shape[0])
    a_new = np.zeros(a.shape)
    for k in range(a.shape[1] - 1, 0, -1):
        a1 = a[:, k - 1]
        a_temp = minmod_3(a[:, k], np.roll(a1, -1) - a1, a1 - np.roll(a1, 1))
        a_new = a_new.at[:, k].set(
            still_limiting * a_temp + (1 - still_limiting) * a[:, k]
        )
        still_limiting = still_limiting * (1 - np.equal(a[:, k], a_new[:, k]))
    a_new = a_new.at[:, 0].set(a[:, 0])
    return a_new
"""
