import numpy as np
from ase.io import read, write

from get_angles import get_angles

from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib import pyplot as plt
import numpy as np



ref_atoms_list = read("unique_structures.extxyz", index=":")
atoms_list = read("../../opt_alanates_mh/unique_isolated.extxyz", index=":")

ref_ordered_angles_list = []
for atoms in ref_atoms_list:
    try:
        ref_ordered_angles_list += [get_angles(atoms)]
    except:
        pass

ordered_angles_list = []
for atoms in atoms_list:
    try:
        ordered_angles_list += [get_angles(atoms)]
    except:
        pass

ddd = []
for vector1 in ref_ordered_angles_list:
    rmsds = []
    avg = []
    for vector2 in ordered_angles_list:
        # RMSD mertic to compare each other
        #  rmsds += [np.sqrt(np.sum((np.array(vector1) - np.array(vector2))**2) / len(vector1))]
        avg += [abs(np.sum(np.array(vector1)) / len(vector1) - np.array(vector2).sum()/len(vector2))]

    #  ddd += [rmsds]
    ddd += [avg]
        #  break


# to get picture of comparison
plt.imshow(np.array(ddd).T, aspect='auto')
plt.xlabel('MD Frames')
plt.ylabel('Minima Hopping Frames')
plt.colorbar()
plt.savefig("compREFanglesFP.png")


