import numpy as np
from ase.io import read, write

from get_angles import get_angles

from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib import pyplot as plt
import numpy as np



atoms_list = read("polymeric.extxyz", index=":")

ordered_angles_list = []
for atoms in atoms_list:
    try:
        ordered_angles_list += [get_angles(atoms)]
    except:
        pass

ddd = []
for vector1 in ordered_angles_list:
    rmsds = []
    avg = []
    for vector2 in ordered_angles_list:
        # RMSD mertic to compare each other
        rmsds += [np.sqrt(np.sum((np.array(vector1) - np.array(vector2))**2) / len(vector1))]
        #  avg += [np.sum((np.array(vector1)) - np.array(vector2).sum()) / len(vector2)]

    ddd += [rmsds]
    #  ddd += [avg]
        #  break

linked = linkage(np.array(ddd),'complete')
label_list = range(len(atoms_list))
cluster_conf = defaultdict(list)
thresh = 6
for key, idx in zip(fcluster(linked, thresh, criterion='distance'), label_list):
    cluster_conf[key].append(idx)

for k, items in enumerate(cluster_conf.values()):
    #  print("="*10)

    # to write a structe among from identical ones
    atoms = atoms_list[items[0]]
    write("unique_structures.extxyz", atoms, append=True)

    # to write identical structure a file
    for i in items:
        atoms = atoms_list[i]
        #  print(atoms.get_potential_energy())
        write("%s_identical_structure.extxyz" %k, atoms, append=True)

    #  if k == 10:
    #      break


# to get picture of comparison
#  plt.imshow(np.array(ddd).T, aspect='auto')
#  plt.xlabel('Frame id')
#  plt.ylabel('ref id')
#  plt.colorbar()
#  plt.savefig("anglesFP.png")


