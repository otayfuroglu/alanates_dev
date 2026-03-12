import sys
sys.path.append('/arf/home/otayfuroglu/alanates_dev/measureSimilarty/src/')
import argparse
import re

from omfp.fJustOMFP import fomfpall
from omfp.fingerprintDistance import FPDistCalc
#  from fortran import fingerprint as fp
#  from fortran import fingerprint_all as fpall

#  from structure.Structure import *
from structure.Structure import Structure, ang2bohr

from ase import Atoms
from ase.io import read, write
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
import tqdm
from collections import defaultdict

from scipy.cluster.hierarchy import linkage, fcluster

import random





def read_cif(inf):
    s = read(inf)
    return Structure(s.get_atomic_numbers(), s.get_positions() * ang2bohr, lattice=s.get_cell(True).array * ang2bohr)


def atoms2structure(atoms):
    return Structure(atoms.get_atomic_numbers(), atoms.get_positions() * ang2bohr, lattice=atoms.get_cell(True).array * ang2bohr)


def convert_cif():
    #  path = '/home/jonas/sync/PhD/alanates/minima/'
    fs = [f for f in os.listdir(path) if 'cif' in f]
    print(fs)
    for f in fs:
        idx = f.index('poslow') + 6
        id = int(f[idx:idx+2])
        phase = 'isolated' if 'iso' in f else 'polymeric'
        s = read_cif(path + f)
        s.save(path + 'ascii/{}-{:03d}.ascii'.format(phase, id))


def merge_structures():
    #  dir = '/home/jonas/tmp/alanates/out/'
    refs = [Structure.readAscii(dir + f) for f in os.listdir(dir) if 'ascii' in f]
    print(len(refs))
    #  Structure.saveDataFull(refs, '/home/jonas/tmp/alanates/out.data')


def dist_test():
    refpos_a = np.array([-1, -1.5])
    refpos_b = np.array([1, 1.5, 2])
    xs = np.linspace(-3, 3, 100)
    ds = []
    for x in xs:
        d = 0.
        # for p in refpos_a:
        #     d = d + (p-x)**2
        # for p in refpos_b:
        #     d = d - (p-x)**2
        da = (x - refpos_a)**2
        db = (x - refpos_b)**2
        d = np.product(da)**(1. / da.size) #- np.product(db)
        ds.append(d)
    plt.plot(xs, ds)
    plt.show()


def check_hungarian():
    from scipy.optimize import linear_sum_assignment
    #  f = open('/home/jonas/sync/PhD/heckdarting/build/out.txt', 'r')
    C = []
    for line in f:
        C.append([float(x) for x in line.split()])
    C = np.array(C)
    plt.imshow(C)
    plt.show()
    r_ind, c_ind = linear_sum_assignment(C)
    fp_dist = C[r_ind, c_ind].sum()
    print(fp_dist)


def lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair):
    symbols = [atom_type_symbol_pair[key] for key in lammps_atoms.get_atomic_numbers()]
    return Atoms(symbols=symbols, positions=lammps_atoms.positions, cell=lammps_atoms.cell)


def comp(a, b):
    n = np.lcm(a.nat, b.nat)
    aa = a.repeat([0, 0, n // a.nat - 1])
    bb = b.repeat([0, 0, n // b.nat - 1])
    return dist_calc.compare(aa, bb) / n


def comp_all(i):
    atoms = atoms_list[i]
    s = atoms2structure(atoms)

    ds = []
    for (f, r) in refs:
        if f > i:
            continue
        d = comp(r, s)
        ds.append((d, f))
    return i, [d for d, f in ds]


def make_pairwise_dist_plot():
    ddd = []
    for fa, ra in refs:
        ds = []
        for fb, rb in refs:
            d = comp(ra, rb)
            ds.append(d)
        ddd.append(ds)
        print(fa, ' '.join([str(d) for d in ds]))

    plt.imshow(ddd)
    plt.colorbar()
    plt.show()


def expand_structure(s):
    return s.repeat([0, 0, 24 // s.nat - 1])



parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-extxyz_path", type=str, required=True, help="..")
parser.add_argument("-refs_path", type=str, required=True, help="..")
parser.add_argument("-nproc", type=int, required=True, help="..")
args = parser.parse_args()
refs_path = args.refs_path
extxyz_path = args.extxyz_path

#  ref_dir = "/kernph/tayfur0000/works/alanates/vasp_opt/LiAlH4_vasp_opt_geoms/"
#  refs = [(f, read_cif(ref_dir + f)) for f in os.listdir(ref_dir) if '.cif' in f]
#  refs = sorted(refs, key=lambda x: x[0])
# refs from extxyz

refs = read(refs_path, index=":")
n_refs = len(refs)
refs = [(i, atoms2structure(atoms)) for i, atoms in enumerate(refs)]

fp_calc = lambda s: fomfpall(s.ats, s.elements, lattice=s.lattice, ns=1, npp=1, maxNatSphere=100)
dist_calc = FPDistCalc(fp_calc)

#  refs = [(f, expand_structure(s)) for f, s in refs]
#  ref_fps = [(f, fp_calc(s)) for f, s in refs]

atoms_list = read(extxyz_path, index=":")
n_atoms_list = len(atoms_list)
iterable = list(range(n_atoms_list))
random.shuffle(iterable)

results = []
with Pool(args.nproc) as pool:
    for result in tqdm.tqdm(pool.imap_unordered(func=comp_all, iterable=iterable), total=n_atoms_list):
        results.append(result)

results = sorted(results, key=lambda x: x[0])

# triangle matrix to squared
matrix = np.zeros((n_atoms_list, n_refs))
for i, ds in results:
    for (f, d) in enumerate(ds):
        matrix[i, f] = d

if n_refs > 1:
    matrix = matrix + matrix.T - np.diag(matrix.diagonal())
np.save(f"all_matrix", matrix)
#  fig_name = os.path.basename(args.extxyz_path).split('.')[0] + '_pairwise_dist.png'
#  plt.imshow(matrix, aspect='auto')
#  plt.xlabel('Frame id')
#  plt.ylabel('Frame id')
#  plt.colorbar()
#  plt.savefig(fig_name)
