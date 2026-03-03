from fortran import fingerprint as fp
from fortran import fingerprint_all as fpall
import numpy as np
from structure.Structure import *
# from OMFP import *

xmax = 18
gridLength = 100
x_grid = np.linspace(0, xmax, num = gridLength, endpoint=False)

def get_rcov(els):
    nat = len(els)
    rcov = np.zeros((nat))
    counter = 0
    for sym in els:
        rcov[counter] = fp.sym2rcov(elementSymbols[sym]) #/ 1.6 # todo: why does this improve performance for CH4
        #rcov[counter] = fp.sym2rcov(elementSymbols[(6 if sym==6 else 4)]) #/ 1.6 # todo: why does this improve performance for CH4
        #rcov[counter] = 0.1 if sym == 1 else 1.0
        counter += 1
    return rcov

def fomfp(ats, els, rcScaling=1.0, ns=2, npp=1, returnDerivatives=False):
    nat = ats.shape[0]
    rcov = get_rcov(els) * rcScaling
    rc = 10000000.0
    f, df = fp.fingerprint_and_derivatives(nat, ns, npp, rc ,np.zeros((3,3)), ats.transpose(), rcov, nat)
    df = np.transpose(df, axes=(0, 2, 1))
    if returnDerivatives:
        return f, df
    else:
        return f

def smooth_fomfp(ats, els, rcScaling=1.0, ns=2, npp=1, returnDerivatives=False):

    f, df = fomfp(ats, els, rcScaling, ns, npp, returnDerivatives=True)

    f_smooth = np.sum(np.sin(np.outer(f, x_grid)), axis=0)

    if returnDerivatives:

        return f_smooth, df
    else:
        return f_smooth

def fomfpall(ats, els, lattice=None, rc=6., rcScaling=1.0, ns=2, npp=1, maxNatSphere=None, returnDerivatives=False):
    nat = ats.shape[0]
    rcov = get_rcov(els) * rcScaling
    if maxNatSphere is None:
        maxNatSphere = nat
    #rc = 10000000.0
    if lattice is None:
        f, df = fpall.fingerprint_and_derivatives(maxNatSphere, ns, npp, rc / 2., np.zeros((3,3)), ats.transpose(), rcov, nat)
    else:
        f, df = fpall.fingerprint_and_derivatives(maxNatSphere, ns, npp, rc / 2., lattice.transpose(), ats.transpose(), rcov, nat)
    #df = df.transpose()
    #df = np.einsum('ijkl->likj', df)
    f = f.transpose()
    if returnDerivatives:
        df = np.transpose(df, (3, 0, 2, 1))
        return f, df
    else:
        return f


def smooth_fomfpall(ats, els, lattice=None, rc=6., rcScaling=1.0, ns=2, npp=1, maxNatSphere=None, returnDerivatives=False):
    f, df = fomfpall(ats, els, lattice, rc, rcScaling, ns, npp, maxNatSphere, returnDerivatives=True)

    f_smooth = np.zeros((ats.shape[0], gridLength))
    df_smooth = np.zeros((ats.shape[0], gridLength, ats.shape[0], 3))

    for i in range(ats.shape[0]):
        f_smooth[i, :] = np.sum(np.sin(np.outer(f[i, :], x_grid)), axis=0)
        if returnDerivatives:
            df_smooth_f = np.cos(np.outer(f[i, :], x_grid)) # df_smooth[i, j] = df_smooth_j / dlamda_i
            df_smooth_f_dlamda = np.einsum('ij,j->ij', df_smooth_f, x_grid)
            df_smooth[i,:,:,:] = np.einsum('ij,ikl->jkl', df_smooth_f_dlamda, df[i,:,:,:])

    # print(f.shape, df.shape, f_smooth.shape, df_smooth.shape)

    if returnDerivatives:
        return f_smooth, df_smooth
    else:
        return f_smooth

def main():
    import sys
    s = Structure.readXYZ('/home/jonas/sync/PhD/conical-intersection/omfp-transformer/src/omfp/00047.xyz')
    f, df = smooth_fomfpall(s.ats, s.elements,rc=8., rcScaling=1.0, ns=1, returnDerivatives=True)
    print(f.shape, df.shape)
    for i in range(5):
        print(x_grid[i], f[0, i])

    dx = np.random.randn(*s.ats.shape) * 1.e-5
    print('s', dx.shape, df.shape)
    sl = s.copy()
    sr = s.copy()
    fl, dfl = smooth_fomfpall(s.ats - dx / 2, s.elements,rc=8., rcScaling=1.0, ns=1, returnDerivatives=True)
    fr, dfr = smooth_fomfpall(s.ats + dx / 2, s.elements,rc=8., rcScaling=1.0, ns=1, returnDerivatives=True)
    rdf = fr - fl
    fdf = np.einsum('ij,klij->kl', dx, df)
    print(rdf)
    print(fdf)
    print()
    print(rdf - fdf)


if __name__ == '__main__':
    main()
