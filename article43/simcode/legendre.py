import numpy as np
from scipy.special import comb
from functools import lru_cache


@lru_cache(maxsize=10)
def generate_legendre(p):
    """
    Returns a (p, p) array which represents
    p length-p polynomials. legendre_poly[k] gives
    an array which represents the kth Legendre
    polynomial. The polynomials are represented
    from highest degree of x (x^{p-1}) to lowest degree (x^0),
    as is standard in numpy poly1d.

    Inputs
    p: the number of Legendre polynomials

    Outputs
    poly: (p, p) array representing the Legendre polynomials
    """
    assert p >= 1
    poly = np.zeros((p, p))
    poly[0, -1] = 1.0
    twodpoly = np.asarray([0.5, -0.5])
    for n in range(1, p):
        for k in range(n + 1):
            temp = np.asarray([1.0])
            for j in range(k):
                temp = np.polymul(temp, twodpoly)
            temp *= comb(n, k) * comb(n + k, k)
            poly[n] = np.polyadd(poly[n], temp)

    return poly
