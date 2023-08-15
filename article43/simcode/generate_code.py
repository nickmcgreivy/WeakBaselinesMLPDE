from sympy import legendre, diff, integrate, symbols
import numpy as np
from scipy.special import comb


def get_volume_integral(l, m, n):
    x = symbols("x")
    expr = legendre(l, x) * legendre(m, x) * diff(legendre(n, x), x)
    return integrate(expr, (x, -1, 1))


def get_volume_array(p):
    res = np.zeros((p, p, p))
    for l in range(p):
        for m in range(p):
            for n in range(p):
                res[l, m, n] = get_volume_integral(l, m, n)
    return res


def print_volume_code(p):
    print("volume_sum = np.zeros(a.shape)")
    arr = get_volume_array(p)
    for n in range(p):
        res = ""
        for l in range(p):
            for m in range(p):
                if arr[l, m, n] != 0.0:
                    value = arr[l, m, n] / 2
                    res += "{} * a[:, {}] * a[:, {}] + ".format(value, l, m)
        res += "0.0"
        print("volume_sum = volume_sum.at([:, {}]).add({})".format(n, res))
    print("return volume_sum")


def get_diffusion_volume_integral(m, n):
    x = symbols("x")
    expr = diff(legendre(m, x), x) * diff(legendre(n, x), x)
    return integrate(expr, (x, -1, 1)).evalf()


def get_diffusion_volume_array(p):
    res = np.zeros((p, p))
    for m in range(p):
        for n in range(p):
            res[m, n] = get_diffusion_volume_integral(m, n)
    return res


def print_diffusion_volume_code(p):
    print("volume_sum = np.zeros(a.shape)")
    arr = get_diffusion_volume_array(p)
    for n in range(p):
        res = ""
        for m in range(p):
            if arr[m, n] != 0.0:
                res += "{} * a[:, {}] + ".format(arr[m, n], m)

        res += "0.0"
        print("volume_sum = volume_sum.at[:, {}].add({})".format(n, res))



#print_volume_code(1)
#print_volume_code(2)
#print_volume_code(3)
#print_volume_code(4)
#print_volume_code(5)
#print_volume_code(6)
#print_volume_code(7)
#print_volume_code(8)
#print_volume_code(9)
print_volume_code(10)

#print_diffusion_volume_code(1)
#print_diffusion_volume_code(2)
#print_diffusion_volume_code(3)
#print_diffusion_volume_code(4)
#print_diffusion_volume_code(5)
#print_diffusion_volume_code(6)
#print_diffusion_volume_code(7)
#print_diffusion_volume_code(8)
#print_diffusion_volume_code(9)
print_diffusion_volume_code(10)

