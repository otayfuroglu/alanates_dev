
import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList

atoms_list = read("./minE_strucute.extxyz", index=":")
cutoff_radius = 4.95

ele_list = ["Li", "Al"]

for i, atoms in enumerate(atoms_list):
    #  if i != 8:
    #      continue
    print(i, "="*10)
    for ele1 in ele_list:
        central_atom_index = [atom.index for atom in atoms if atom.symbol == ele1][0]
        nl = NeighborList([cutoff_radius / 2] * len(atoms), skin=0.0, self_interaction=False, bothways=True)
        nl.update(atoms)

        central_atom = atoms[central_atom_index]
        neighbors, offsets = nl.get_neighbors(central_atom_index)
        #  print(offsets)


        for ele2 in ele_list:
            counter = 0
            for neighbor_index, offset in zip(neighbors, offsets):
                if atoms[neighbor_index].symbol == ele2:
                     counter += 1
            print(f"{ele1}-{ele2} --> {counter}")
    #  print(neighbor)
