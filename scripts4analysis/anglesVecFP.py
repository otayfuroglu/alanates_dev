import numpy as np
from ase.io import read
from ase.build import molecule
from ase.neighborlist import NeighborList

from matplotlib import pyplot as plt
import numpy as np

def angle_between_lines(vector1, vector2):
    # Calculate the dot product between the two vectors
    dot_product = np.dot(vector1, vector2)

    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the cosine of the angle between the vectors
    cosine_theta = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians
    angle_radians = np.arccos(cosine_theta)

    # Convert radians to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def getAllVectors():
    vectors = []
    for central_atom_index in central_atom_indexs:
        central_atom = atoms[central_atom_index]
        neighbors, offsets = nl.get_neighbors(central_atom_index)

        for neighbor_index, offset in zip(neighbors, offsets):
            neighbor = atoms[neighbor_index]
            spatial_vector = neighbor.position - (central_atom.position + np.dot(offset, atoms.get_cell()))
            vectors += [spatial_vector]

    angles = []
    for _ in range(len(vectors)):
        vector1 = vectors.pop()
        for vector2 in vectors:
            angle = angle_between_lines(vector1, vector2)
            print(f"The angle is {angle} degrees.")
            angles += [angle]
        #  break



# Create a molecule (or any atomic structure)
#  atoms = read("./isloated_rotated.extxyz", index=-1)

# Specify the index of the central atom
#  central_atom_indexs = [atom.index for atom in atoms if atom.symbol == "Al" ]  # You should set this to the index of your central atom

cutoff_radius = 1.2  # Adjust this value to set the neighbor cutoff
#  nl = NeighborList([cutoff_radius / 2] * len(atoms), self_interaction=False, bothways=True)
#  nl.update(atoms)

#  nodeVectors = getNodeVectors()
#  sumNodeVector = []
#  for key, values in nodeVectors.items():
#      #  print(np.sum(values, axis=0))
#      sumNodeVector += [np.sum(values, axis=1)]


#  print(angle_between_lines(sumNodeVector[0], sumNodeVector[1]))
#  getAllVectors()
#  print(sumNodeVector)


def getAngels(atoms_list: list) -> list:
    def getNodeVectors():
        central_atom_index_vectors = {}
        for central_atom_index in central_atom_indexs:
            central_atom = atoms[central_atom_index]
            neighbors, offsets = nl.get_neighbors(central_atom_index)
            vectors = []
            for neighbor_index, offset in zip(neighbors, offsets):
                neighbor = atoms[neighbor_index]
                spatial_vector = neighbor.position - (central_atom.position + np.dot(offset, atoms.get_cell()))
                vectors += [spatial_vector]
            central_atom_index_vectors[central_atom_index] = vectors
        return central_atom_index_vectors

    results = []
    for i, atoms in enumerate(atoms_list):
        if i == 0:
            central_atom_indexs = [atom.index for atom in atoms if atom.symbol == "Al" ]
            nl = NeighborList([cutoff_radius / 2] * len(atoms), self_interaction=False, bothways=True)
        nl.update(atoms)
        nodeVectors = getNodeVectors()
        sumNodeVector = []
        for key, values in nodeVectors.items():
            sumNodeVector += [np.sum(values, axis=1)]
        try:
            results += [angle_between_lines(sumNodeVector[0], sumNodeVector[1])]
        except:
            pass
    return results



atoms_list = read("./isloated_rotated.extxyz", index=":")
angles = getAngels(atoms_list)

ddd = []
for angle1 in angles:
    diffs = []
    for angle2 in angles:
        diffs += [abs(angle1 - angle2)]

    ddd.append(diffs)
#  quit()

#  print(len(ddd))

#  ddd = [fl_ddd[1] for fl_ddd in results]

plt.imshow(np.array(ddd).T, aspect='auto')
plt.xlabel('Frame id')
plt.ylabel('ref id')
plt.colorbar()
plt.savefig("anglesFP.png")
