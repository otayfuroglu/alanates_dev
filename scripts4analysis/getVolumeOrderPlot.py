#
#
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import NeighborList
import matplotlib.patches as mpatches

import numpy as np
from matplotlib import pyplot as plt


#  atoms_list = read("vasp_opt_polymeric_all.extxyz", index=":")
#  atoms_e_list = [(atoms, atoms.get_volume()) for atoms in atoms_list]
#
#  atoms_e_list = sorted(atoms_e_list, key=lambda x: x[1])
#  for atoms, e in atoms_e_list:
#      write("sorted_opt_vasp_polymeric_all.extxyz", atoms, append=True)
#
#  quit()


all_volumes = []

def getEplot(atoms_list, shift, color, label):
    global all_volumes
    volumes = []
    for atoms in atoms_list:
        #  volume = atoms.get_volume() / len(atoms)
        volume = atoms.get_volume()
        #  if any([f"{volume:.3f}" == f"{en_in:.3f}" for en_in in volumes]):
            #  print("Equal Energy")
            #  continue
        plt.axhline (y=volume , xmin=0.01+shift, xmax=0.3+shift, linestyle="-", color=color, alpha=1, linewidth=0.5)
        #  plt.text(volume, 0.4, f"Frame_{i}", fontsize=12)
        volumes += [volume]
        all_volumes += [volume]
        #  if volume <= -67.29751586:
            #  write("min_volume_structres.extxyz", atoms, append=True)
    #  print(sorted(volumes)[0])
    #  return volumes
    return mpatches.Patch(color=color, label=f"{label} # {len(volumes)}")


struc_type_legends = []
#  atoms_list = read("../vasp/vasp_opt_isolated_all.extxyz", index=":")
atoms_list = read("vasp_opt_isolated_mh.extxyz", index=":")
#  label = "n2p2 En. Vasp opt. Isolated"
#  label = "Isolated"
label = "minima hoping"
struc_type_legends += [getEplot(atoms_list, shift=0.0, color="b", label=label)]

atoms_list = read("tight_vasp_opt_isolated_rotated_v1.extxyz", index=":")
label = "rotated"
#  label = "Vasp Opt. Initial Polymeric"
struc_type_legends+= [getEplot(atoms_list, shift=0.3, color="r", label=label)]

#  atoms_list = read("sorted_opt_vasp_polymeric_all.extxyz", index=slice(0,1))
#  label = "Vasp Opt MineE polymeric"
#  struc_type_legends += [getEplot(atoms_list, shift=0.6, color="c", label=label)]

#  avg_diff = np.array(volumes).mean() - np.array(volumes).mean()
#  print(avg_diff)

plt.ylim([min(all_volumes)-0.5, max(all_volumes)+0.5])
plt.legend(handles=struc_type_legends)
#  plt.show()
plt.savefig("volumeOrder.png")
