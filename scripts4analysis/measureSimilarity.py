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
        if f >= i:
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
args = parser.parse_args()
refs_path = args.refs_path
extxyz_path = args.extxyz_path

#  ref_dir = "/kernph/tayfur0000/works/alanates/vasp_opt/LiAlH4_vasp_opt_geoms/"
#  refs = [(f, read_cif(ref_dir + f)) for f in os.listdir(ref_dir) if '.cif' in f]
#  refs = sorted(refs, key=lambda x: x[0])
# refs from extxyz

refs = read(refs_path, index=":")
refs = [(i, atoms2structure(atoms)) for i, atoms in enumerate(refs)]

fp_calc = lambda s: fomfpall(s.ats, s.elements, lattice=s.lattice, ns=1, npp=1, maxNatSphere=100)
dist_calc = FPDistCalc(fp_calc)

#  refs = [(f, expand_structure(s)) for f, s in refs]
#  ref_fps = [(f, fp_calc(s)) for f, s in refs]

atoms_list = read(extxyz_path, index=":")
n_atoms_list = len(atoms_list)

results = []
with Pool(112) as pool:
    for result in tqdm.tqdm(pool.imap_unordered(func=comp_all, iterable=range(n_atoms_list)), total=n_atoms_list):
        results.append(result)

results = sorted(results, key=lambda x: x[0])

# triangle matrix to squared
matrix = np.zeros((n_atoms_list, n_atoms_list))
for i, ds in results:
        for (f, d) in enumerate(ds):
            if f >= i:
                continue
            matrix[i, f] = d
            matrix[f, i] = d
matrix = matrix + matrix.T - np.diag(matrix.diagonal())




#  matrix = np.zeros((n_atoms_list, n_atoms_list))
#  for i in tqdm.tqdm(range(n_atoms_list)):
#
#          atoms = atoms_list[i]
#          s = atoms2structure(atoms)
#          ds = []
#          for (f, r) in refs:
#              if f >= i:
#                  continue
#              d = comp(r, s)
#              ds.append((d, f))
#              matrix[i, f] = d
#              matrix[f, i] = d
#      #  result = comp_all(i)
#      #  results.append(result)
#  #  results = sorted(results, key=lambda x: x[0])
#  matrix = matrix + matrix.T - np.diag(matrix.diagonal())


#  matrix = np.array([fl_ddd[1] for fl_ddd in results])
#  print(matrix)

np.save("all_matrix", matrix)

fig_name = os.path.basename(args.extxyz_path).split('.')[0] + '_pairwise_dist.png'
plt.imshow(matrix, aspect='auto')
plt.xlabel('Frame id')
plt.ylabel('Frame id')
plt.colorbar()
plt.savefig(fig_name)
#  matrix = np.load("all_matrix.npy")

#  linked = linkage(matrix,'complete')
#  label_list = range(len(atoms_list))
#  cluster_conf = defaultdict(list)
#  thresh = 0.1
#  for key, idx in zip(fcluster(linked, thresh, criterion='distance'), label_list):
#      cluster_conf[key].append(idx)

#  print(len(cluster_conf.keys()))


#  quit()
#  v_max = matrix.max()
#  v_min = matrix.min()

#  idx_sum_fp = np.array( [(i, sum(arr)) for i, arr in results])
#
#  mask = idx_sum_fp[:,1] > 18.0
#  l1 = idx_sum_fp[idx_sum_fp[:,1] > 18.0]
#  mask1 = idx_sum_fp[idx_sum_fp[:,1] < 18.0]
#  l2 = mask1[mask1[:, 1] > 11.0]
#  l3 = idx_sum_fp[idx_sum_fp[:,1] <  11.0]
#
#  for i in l2[:, 0]:
#      atoms = read(args.extxyz_path, index=int(i))
#      write(f"l2_{os.path.basename(args.extxyz_path).split('.')[0]}.extxyz", atoms, append=True)


#  for i, l in enumerate([l1, l2, l3]):
#      ddd = [fl_ddd[1] for fl_ddd in results if fl_ddd[0] in l]
#
#      plt.imshow(np.array(ddd).T, vmin=v_min, vmax=v_max,  aspect='auto')
#      plt.xlabel('MD Frame id')
#      plt.ylabel('ref id')
#
#      if i == 0:
#          plt.colorbar()
#      plt.savefig("%s_%s" %(i, fig_name))
#
#  quit()
#  #
#  x = idx_sum_fp[:,0]
#  y = idx_sum_fp[:,1]
#  plt.scatter(x, y)
#  plt.xlabel('Frame id')
#  plt.ylabel('Sum of FP distance')
#  #  #  for x, y in zip(x,y):
#  #  #      if y > 15:
#  #  #          plt.annotate(x, (x, y))
#  plt.savefig(fig_name)
#  #
#  quit()
#

#  ddd = [fl_ddd[1] for fl_ddd in results]

#  fig_name = os.path.basename(args.extxyz_path).split('.')[0] + '_pairwise_dist.png'
#  plt.imshow(np.array(ddd).T, aspect='auto')
#  plt.xlabel('Frame id')
#  plt.ylabel('ref id')
#  plt.colorbar()
#  plt.savefig(fig_name)
#  plt.show()
