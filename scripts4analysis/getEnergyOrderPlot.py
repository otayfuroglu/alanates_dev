#
#
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import NeighborList
import matplotlib.patches as mpatches

import numpy as np
from matplotlib import pyplot as plt


keyword = "isolated"
#  keyword = "polymeric"

#  atoms_list = read(f"vasp_opt_{keyword}_all.extxyz", index=":")
#  atoms_e_list = [(atoms, atoms.get_potential_energy()) for atoms in atoms_list]
#
#  atoms_e_list = sorted(atoms_e_list, key=lambda x: x[1])
#  for atoms, e in atoms_e_list:
#      write(f"sorted_opt_vasp_{keyword}_all.extxyz", atoms, append=True)
#
#  quit()


all_enegies = []

def getNames(atoms_list):
    names = []
    for atoms in atoms_list:
        names += [atoms.info["label"]]
    return names


def getEplot(atoms_list, shift, color, label):
    global all_enegies
    energies = []
    for atoms in atoms_list:
        energy = atoms.get_potential_energy()
        energies += [energy / len(atoms)]
    min_e = min(energies)
    for energy in energies:
        energy -= min_e
        #  energy = atoms.get_potential_energy()
        #  if any([f"{energy:.3f}" == f"{en_in:.3f}" for en_in in energies]):
            #  print("Equal Energy")
            #  continue
        plt.axhline (y=energy, xmin=0.01+shift, xmax=0.3+shift, linestyle="-", color=color, alpha=1, linewidth=0.5)
        #  plt.text(energy, 0.4, f"Frame_{i}", fontsize=12)
        all_enegies += [energy]
        #  if energy <= -67.29751586:
            #  write("min_energy_structres.extxyz", atoms, append=True)
    #  print(sorted(energies_v1)[0])
    #  return energies
    return mpatches.Patch(color=color, label=f"{label} # {len(energies)}")


struc_type_legends = []

#  atoms_list = read(f"../vasp/vasp_opt_{keyword}_all.extxyz", index=":")
#  atoms_list = read(f"../vasp/vasp_opt_{keyword}_all.extxyz", index=":")
atoms_list = read(f"vasp_opt_initial_structures_with_label_{keyword}.extxyz", index=":")
#  label = "n2p2 En. Vasp opt. Isolated"
#  label = "vasp opt. isolated"
#  label = "minima hopping"
label = f"{keyword}_VASP_PBE"
struc_type_legends += [getEplot(atoms_list, shift=0.0, color="b", label=label)]
#  atoms_list = read(f"sorted_opt_vasp_{keyword}_all.extxyz", index=slice(0,1))
#  label = f"Vasp Opt MineE {keyword}"
#  struc_type_legends += [getEplot(atoms_list, shift=0.6, color="c", label=label)]

#  atoms_list = read(f"/Users/omert/Desktop/alanates/qe/opt/qe_opt_{keyword}.extxyz", index=":")
atoms_list = read(f"vasp_opt_PBE_initial_structures_{keyword}.extxyz", index=":")
#  names_qe = getNames(atoms_list)
#  label = "rotated"
#  label = "Vasp Opt. Initial Polymeric"
#  label = f"n2p2 opt. {keyword}"
#  label = "nequip opt. isolated"
label = f"{keyword}_VASP_SCAN"
struc_type_legends+= [getEplot(atoms_list, shift=0.3, color="r", label=label)]


#  avg_diff = np.array(energies_v1).mean() - np.array(energies_v2).mean()
#  print(avg_diff)

plt.ylim([min(all_enegies)-0.01, max(all_enegies)+0.01])
plt.ylabel("Relative Energy / atom (eV)")
plt.legend(handles=struc_type_legends)
#  plt.show()
plt.savefig(f"vasp_scan_pbe_energyOrder_{keyword}.png")
