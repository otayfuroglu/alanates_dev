#
from ase import Atoms
from ase.io import read
from ase.neighborlist import NeighborList
import matplotlib.patches as mpatches

import numpy as np
from matplotlib import pyplot as plt


nat_cry_type = {"rock_salt": [6, 12, 12], "ZincBlende": [4, 12, 12], "Fluorite": [4, 12, 6],
                "Wurtzite": [4, 12, 12], "NickelArsenide": [6, 12, 6], "CaesiumChlride": [8, 6, 6],
                "CadmiumIodide": [6, 12, 6]}
color_dict = {"rock_salt": "r", "ZincBlende": "b", "Fluorite": "g",
                "Wurtzite": "c", "NickelArsenide": "m", "CaesiumChlride": "teal",
                "CadmiumIodide": "orange"}

#  color_dict = {"rock_salt": "r", "ZincBlende": "b"} # "g", "c", "m"

def getStructureType(atoms: Atoms) -> str:

    ele_list = ["Li", "Al"]
    cutoff_radius = 4.95

    l = []

    counter1 = 0
    for ele1 in ele_list:
        central_atom_index = [atom.index for atom in atoms if atom.symbol == ele1][0]
        nl = NeighborList([cutoff_radius / 2] * len(atoms), skin=0.0, self_interaction=False, bothways=True)
        nl.update(atoms)
        central_atom = atoms[central_atom_index]
        neighbors, offsets = nl.get_neighbors(central_atom_index)
        #  print(offsets)
        for ele2 in ele_list:
            counter2 = 0
            for neighbor_index, offset in zip(neighbors, offsets):
                if atoms[neighbor_index].symbol == ele2:
                     counter2 += 1

            if ele1 != ele2:
                counter1 += counter2
            else:
                l += [counter2]
    l.insert(0, counter1 / len(ele_list))
    l = np.array(l)

    nat_cry_type_mad = {}
    for key, values in nat_cry_type.items():
        #  mad = np.abs(l - np.array(values)).mean()
        rmsd = np.sqrt(np.abs(l**2 - np.array(values)**2).mean())
        nat_cry_type_mad[key] = rmsd

    for key, value in nat_cry_type_mad.items():
        if value < 1.0:
            return key
            #  print(key)
    #  label = min(nat_cry_type_mad, key=nat_cry_type_mad.get)
    #  return label

#  atoms = read("./21_identical_structure.extxyz")
#  getStructureType(atoms)
#  quit()

energies = []
struc_types = []

i = -1
for atoms in read("unique_structures.extxyz", index=":"):
#  for atoms in read("./unique_structures.extxyz", index=":"):

    i += 1
    struc_type = getStructureType(atoms)
    if struc_type is None:
        continue

    if struc_type not in struc_types:
        struc_types += [struc_type]

    try:
        energy = atoms.get_potential_energy()
    except:
        continue
    plt.axhline (y=energy , xmin=0.01, xmax=0.3,
                 color=color_dict[struc_type], alpha=1, linewidth=2)
    plt.text(0.35, energy, f"Frame_{i}", fontsize=10)

    energies += [energy]

# Horizontal line 3

#  plt.axhline (y = 2, color = 'm')

# Horizontal line 4
# We also set range of the line

#  plt.axhline (y = 4, xmin =0.6, xmax =0.8, color='c')
struc_type_legends =[]
for struc_type in struc_types:
    struc_type_legends += [mpatches.Patch(color=color_dict[struc_type], label=struc_type)]

plt.ylim([min(energies)-0.05, max(energies)+0.05])
#  plt.ylim([-48.01, -47.9])
plt.legend(handles=struc_type_legends)
#  plt.show()
plt.savefig("energyOrderStructureType.png")
