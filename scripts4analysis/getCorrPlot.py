#
#
from ase import Atoms
from ase.io import read, write

import numpy as np
from matplotlib import pyplot as plt


def getLabelEnergies(atoms_list):
    return [(a.info["label"], a.get_potential_energy() / len(a)) for a in atoms_list]


atoms_list = read("vasp/vasp_opt_selected_100_isolated.extxyz", index=":")# +  read("vasp/vasp_opt_selected_100_polymeric.extxyz", index=":")

vasp_l_es = getLabelEnergies(atoms_list)

atoms_list = read("nequip/opt_nequip_selected_100_isolated.extxyz", index=":")# +  read("n2p2/opt_n2p2_selected_100_polymeric.extxyz", index=":")

mode_l_es = getLabelEnergies(atoms_list)

vasp_energies = []
model_energies = []
for vasp_l_e in vasp_l_es:
    for mode_l_e in mode_l_es:
        if vasp_l_e[0] == mode_l_e[0]:
            #  print(vasp_l_e[1], mode_l_e[1])
            vasp_energies += [vasp_l_e[1]]
            model_energies += [mode_l_e[1]]

vasp_energies = np.array(vasp_energies)
model_energies = np.array(model_energies)

rmse = np.sqrt(sum((vasp_energies - model_energies)**2))
print(rmse)


plt.scatter(vasp_energies, model_energies)
plt.axline(xy1=(-4.02, -4.02), slope=1, color='red', linestyle="-")
plt.xlabel("VASP (eV / atom)")
plt.ylabel("nequip (eV / atom)")
#  plt.ylabel("nequip (eV)")
#  plt.legen()
plt.savefig("energyCorr.png")
