import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from structure.Structure import Structure


def invdistfp(s: Structure, i: int, rc: float):
    invdists = np.zeros(s.nat)
    dinvdists = np.zeros((s.nat, s.nat, 3))
    for j in range(s.nat):
        if j != i:
            r = np.linalg.norm(s.ats[i, :] - s.ats[j, :])
            f, df = invdistfunc(r, rc)
            invdists[j] = f
            d = df * (s.ats[i, :] - s.ats[j, :]) / r
            dinvdists[j, j, :] -= d
            dinvdists[j, i, :] += d
    invdists = np.array(invdists)
    # return invdists, dinvdists
    order = np.argsort(invdists)
    return invdists[order], dinvdists[order, :, :]

def invdistfp_all(s: Structure, rc: float):
    f = []
    df = []
    for i in range(s.nat):
        fi, dfi = invdistfp(s, i, rc)
        f.append(fi)
        df.append(dfi)
    return f, df

def invdistfunc(r, rc):
    # return 1. / r
    if r < rc:
        return 1. * (r - rc)**2 / r, 1. - (rc / r) ** 2
    else:
        return 0., 0.

def main():
    pos = np.array(
        [[0., 0., 0.],
         [1., 0., 0.],
         [1., 1., 0.],
         [0., 1.1, 0.]])
    s = Structure([1] * 4, pos)
    dx = np.random.randn(*pos.shape) * 1.e-6
    f, df = invdistfp(s, 0, 4.)
    s.ats += dx
    ff, dff = invdistfp(s, 0, 4.)
    print(f)
    print(df)
    print(f - ff)
    print(np.einsum('ijk,jk->i', df, dx))


if __name__ == '__main__':
    main()






