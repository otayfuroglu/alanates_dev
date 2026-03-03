import numpy as np
from numba import njit, jit
from structure.Structure import Structure
from functools import partial

@njit()
def sf2(iat, ats, els, rc, eta, Rs, sfEl):
    nat = ats.shape[0]
    sf = 0.
    dsf = np.zeros((nat, 3))
    for i in range(nat):
        if i == iat:
            continue
        if els[i] == sfEl:
            r = ats[iat,:] - ats[i,:]
            d = np.linalg.norm(r)
            fc, dfc = cutoff(d, rc)
            rdfc = dfc * r

            tmp1 = np.exp(-eta * (d-Rs)**2)
            dsf[i,:] = -tmp1 * rdfc + 2. * eta * (d-Rs) * r / d * tmp1 * fc
            dsf[iat,:] -= dsf[i,:]
            sf += tmp1 * fc
    return sf, dsf

@njit()
def sf3(iat, ats, els, rc, eta, lam, zeta, sfEls):
    nat = ats.shape[0]
    sf = 0.
    dsf = np.zeros((nat, 3))
    for j in range(nat):
        if j==iat:
            continue
        dij = ats[iat, :] - ats[j, :]
        rij = np.linalg.norm(dij)
        fcij, dfcij = cutoff(rij, rc)
        rdfcij = dfcij * dij
        for k in range(j+1, nat):
            if k == iat:
                continue
            if (els[k] == sfEls[0] and els[j] == sfEls[1]) or (els[k] == sfEls[1] and els[j] == sfEls[0]):
                dik = ats[iat,:] - ats[k,:]
                djk = ats[j,:] - ats[k,:]
                rik = np.linalg.norm(dik)
                rjk = np.linalg.norm(djk)
                fcik, dfcik = cutoff(rik, rc)
                rdfcik = dfcik * dik
                fcjk, dfcjk = cutoff(rjk, rc)
                rdfcjk = djk * dfcjk
                costheta = (rjk**2 - rij**2 - rik**2)/(-2. * rij * rik)
                dcosthetaj = -dik / (rij * rik) + dij * np.sum(dij*dik) / (rij**3 * rik)
                dcosthetak = -dij / (rij * rik) + dik * np.sum(dij*dik) / (rik**3 * rij)
                fcpart = fcij * fcik * fcjk
                dfcpartj =  fcik * (rdfcjk * fcij - rdfcij * fcjk)
                dfcpartk = -fcij * (rdfcjk * fcik + rdfcik * fcjk)
                prefactor = 2.**(1 - zeta)
                cospart = (1. + lam * costheta)**zeta
                dcospartj = zeta * (1. + lam * costheta)**(zeta - 1) * lam * dcosthetaj
                dcospartk = zeta * (1. + lam * costheta)**(zeta - 1) * lam * dcosthetak
                exppart = np.exp(-eta * (rij**2 + rik**2 + rjk**2))
                dexppartj = exppart * eta * (2. * (dij - djk))
                dexppartk = exppart * eta * (2. * (dik + djk))
                dSFj = prefactor * (cospart * exppart * dfcpartj + cospart * dexppartj * fcpart + dcospartj * exppart * fcpart)
                dSFk = prefactor * (cospart * exppart * dfcpartk + cospart * dexppartk * fcpart + dcospartk * exppart * fcpart)
                dsf[j,:] += dSFj
                dsf[k,:] += dSFk
                dsf[iat,:] -= dSFj + dSFk
                sf += prefactor * cospart * exppart * fcpart
    return sf, dsf


@njit()
def cutoff(r: float, rc: float) -> (float, float):
    if r < rc:
        f = np.tanh(1. - r / rc)**3
        df = -3. / rc / r * (np.tanh(1. - r / rc)**2 - np.tanh(1. - r / rc)**4)
    else:
        f = 0.
        df = 0.
    return f, df


class SymFunctions:
    def __init__(self, sfparams):
        self.elements = sfparams.keys()
        self.sfs = {}
        for el in self.elements:
            self.sfs[el] = []
            for sfp in sfparams[el]:
                print(sfp)
                if sfp['type'] == 2:
                    self.sfs[el].append(
                        partial(sf2, rc=sfp['rc'], eta=sfp['eta'], Rs=sfp['Rs'], sfEl=sfp['sfEl'])
                        # lambda will not work because of 'late binding'
                        #lambda iat, ats, els: sf2(iat, ats, els, sfp['rc'], sfp['eta'], sfp['Rs'], sfp['sfEl'])
                    )
                elif sfp['type'] == 3:
                    self.sfs[el].append(
                        partial(sf3, rc=sfp['rc'], eta=sfp['eta'], lam=sfp['lam'], zeta=sfp['zeta'], sfEls=sfp['sfEls'])
                        # lambda will not work because of 'late binding'
                        #lambda iat, ats, els: sf3(iat, ats, els, sfp['rc'], sfp['eta'], sfp['lam'], sfp['zeta'], sfp['sfEls'])
                    )
                else:
                    raise Exception('Unknown SF type')
    def computeSfs(self, ats: Structure):
        allsfslist = []
        alldsflist = []
        for i in range(ats.nat):
            sflist = []
            dsflist = []
            for sf in self.sfs[ats.elements[i]]:
                s, ds = sf(i, ats.ats, ats.elements)
                sflist.append(s)
                dsflist.append(ds)
            allsfslist.append(np.array(sflist))
            alldsflist.append(np.array(dsflist))
        return allsfslist, alldsflist


