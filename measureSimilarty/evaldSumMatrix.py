#
from dscribe.descriptors import EwaldSumMatrix, SineMatrix, ACSF
from ase.io import read, write
from ase import Atoms

import numpy as np
import matplotlib.pyplot as plt

import argparse
import os


import matplotlib.pyplot as plt

import matplotlib.colors as colors




def lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair):
    symbols = [atom_type_symbol_pair[key] for key in lammps_atoms.get_atomic_numbers()]
    return Atoms(symbols=symbols, positions=lammps_atoms.positions, cell=lammps_atoms.cell)


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-trj_path", type=str, required=True, help="..")
parser.add_argument("-interval", type=int, required=False, default=1, help="..")

args = parser.parse_args()
lammps_trj_path = args.trj_path

idxes = slice(0, -1, args.interval)
atom_type_symbol_pair = {1:"Al", 2:"Li", 3:"H"}
lammps_trj = read(lammps_trj_path, format="lammps-dump-text", index=idxes, parallel=True)

ems = EwaldSumMatrix(
    n_atoms_max=48,
    permutation="none",
    flatten=True
)

# Setting up the sine matrix descriptor
sm = SineMatrix(
    n_atoms_max=48,
    permutation="sorted_l2",
    sparse=False,
    flatten=True
)

# Setting up the ACSF descriptor
acsf = ACSF(
    species=["H", "Li", "Al"],
    rcut=6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)

fps = []
for i, lammps_atoms in enumerate(lammps_trj):
    atoms = lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair)
    #  write(f"selectedGeoms/frame_{i}.vasp", atoms)
    fps += [ems.create(atoms, accuracy=1e-5).sum() / len(atoms)]
    #  fps += [sm.create(atoms).sum() / len(atoms)]
    #  fps += [acsf.create(atoms).sum() / len(atoms)]

#  print(fps)


fps = np.array(fps)

OPT_GEOM_DIR = "/Users/omert/Desktop/alanates/workingOnStructures/LiAlH4_vasp_opt_geoms"
#  ref_fl_names = [fl for fl in os.listdir(OPT_GEOM_DIR) if "isolated" in fl]
ref_fl_names = sorted([fl for fl in os.listdir(OPT_GEOM_DIR) if ".cif" in fl])

similarity_matrix = np.zeros((len(fps), len(ref_fl_names)))

cmap = colors.LinearSegmentedColormap.from_list('my_colormap', ['r', 'g', 'b'])

for i, ref_fl_name in enumerate(ref_fl_names):
    ref_atoms = read(f"{OPT_GEOM_DIR}/{ref_fl_name}")
    ref = ems.create(ref_atoms, accuracy=1e-5).sum() / len(ref_atoms)
    diff = np.abs(fps - np.array(ref))
    similarity_matrix[:, i] = diff
    plt.imshow(similarity_matrix.transpose(), interpolation='nearest', cmap=cmap)
    plt.text(21, 1 * i, ref_fl_name)
#
#  print(similarity_matrix)
#
# Create a colormap that goes from blue to red

# Plot the data using the colormap
#  plt.colorbar()
#  plt.show()
plt.savefig("similarity.png")
# Set the tick marks and labels
#  ax.set_xticks(np.arange(similarity_matrix.shape[1]) + 0.5, minor=False)
#  ax.set_yticks(np.arange(similarity_matrix.shape[0]) + 0.5, minor=False)
#  ax.set_xticklabels(['Vector 2'], minor=False)
#  ax.set_yticklabels(['Vector 1'], minor=False)

# Show the plot
