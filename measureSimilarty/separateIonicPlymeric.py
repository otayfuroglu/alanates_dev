
import numpy as np

import os
from coordNum import coordnum
from ase import Atoms
from ase.io import read, write
from ase.geometry import cell_to_cellpar, cellpar_to_cell

from tqdm import tqdm

import argparse
import re


def atoms2Ascii(atoms):

    # number of atoms
    atoms_prep_list = [[f"{len(atoms):7.0f}"]]

    # cell parameters
    cell = cellpar_to_cell(cell_to_cellpar(atoms.cell))
    dxx = cell[0, 0]
    dyx, dyy = cell[1, 0:2]
    dzx, dzy, dzz = cell[2, 0:3]

    cell_template = '{:15.12f} {:15.12f} {:15.12f}'
    atoms_prep_list += [cell_template.format(dxx, dyx, dyy)]
    atoms_prep_list += [cell_template.format(dzx, dzy, dzz)]

    # positons and symbols
    atom_template = '{:15.12f} {:15.12f} {:15.12f} {:2s}'
    atoms_prep_list += [[atom_template.format(position[0], position[1], position[2], symbol)]
            for position, symbol, in zip( (atoms.positions).tolist(), atoms.symbols)]

    with open(f"tmp.ascii", "w") as fl:
        for line in atoms_prep_list:
            for item in line:
                fl.write(str(item))
            fl.write("\n")



def checkH2Bond(atoms):
    H_index = [atom.index for atom in atoms if atom.symbol == "H"]
    HH_bonds = [atoms.get_distance(i, j) for i in H_index for j in  H_index]
    HH_bonds = list(filter(lambda x: x != 0, HH_bonds))
    return min(HH_bonds)


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-trj_path", type=str, required=True, help="..")
parser.add_argument("-interval", type=int, required=False, default=1, help="..")
#  parser.add_argument("-nproc", type=int, required=True, help="..")
args = parser.parse_args()
trj_path = args.trj_path

idxs = slice(0, -1, args.interval)
atoms_list = read(trj_path, index=idxs)

keyword = re.findall(r'\d+', trj_path)
fl_out = f"{keyword[0]}_atoms.extxyz"

for i, atoms in enumerate(tqdm(atoms_list)):
    # to remove unphysical structures
    if checkH2Bond(atoms) < 0.9:
        continue

    atoms2Ascii(atoms)
    nat = len(atoms)
    #  print(nat)

    coordnum("tmp.ascii", "tmp.extxyz", nat)
    atoms_with_coord = read("tmp.extxyz")
    coordnums = atoms_with_coord.arrays["coordn"]
    av_coordnum = coordnums[coordnums > 0.0].mean()
    os.remove("tmp.ascii")
    os.remove("tmp.extxyz")

    #  print(av_coordnum)
    if np.floor(av_coordnum + 0.5) == 6.0:
        write(f"polymeric_{fl_out}", atoms, append=True)
        write(f"polymeric_all_{fl_out}", atoms, append=True)
    elif 4.7 < av_coordnum:
        write(f"like_polymeric_{fl_out}", atoms, append=True)
        write(f"polymeric_all_{fl_out}", atoms, append=True)
    elif np.floor(av_coordnum + 0.5) == 4.0:
        write(f"isolated_{fl_out}", atoms, append=True)
        write(f"isolated_all_{fl_out}", atoms, append=True)
    elif av_coordnum <=  4.7:
        write(f"like_isolated_{fl_out}", atoms, append=True)
        write(f"isolated_all_{fl_out}", atoms, append=True)
    #  break
