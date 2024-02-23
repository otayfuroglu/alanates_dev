import numpy as np
from ase.io import read, write
from ase.geometry.analysis import Analysis
from sklearn.metrics import pairwise_distances
import minimahopping.mh.periodictable as periodictable

import tqdm


def get_rcovs(elements):
    rcovs = []
    for element in elements:
        rcovs.append(periodictable.getRcov_n(element))
    rcovs = np.array(rcovs)
    return rcovs


def get_minimal_pairwise_distances(atoms):
    # Initializations
    nat = len(atoms)
    lattice = atoms.get_cell()
    positions = atoms.get_positions()
    minimal_distances = np.ones((nat,nat)) * np.inf
    new_positions = positions.copy()

    for ix in range(-1,2,1):
        for iy in range(-1,2,1):
            for iz in range(-1,2,1):
                positions_w = positions - ix*lattice[0,:]  - iy*lattice[1,:] - iz*lattice[2,:]
                distances = pairwise_distances(positions, positions_w)
                minimal_distances = np.minimum(minimal_distances, distances)
                for i in range(nat):
                    for j in range(nat):
                        dist = 0
                        for k in range(len(positions[i])):
                            diff = positions[i][k] - (positions[j][k] - ix * lattice[0, k] - iy * lattice[1, k] - iz * lattice[2, k])
                            dist += diff * diff

                        if minimal_distances[i][j] > dist:
                            minimal_distances[i][j] = dist


    assert np.inf not in minimal_distances, "error in minimal distances. Found infinite distance"
    atoms.set_positions(new_positions)
    return minimal_distances


def get_molecules(atoms, distances, rcovs, factor_cov, verbose = True):
    nat = len(atoms)
    number_of_molecules = 0
    molecule_index = np.zeros((nat), dtype = int)
    belongs_to = np.zeros(nat, dtype=bool)
    molecules = []

    for iat in range(nat):
        if not belongs_to[iat]:
            belongs_to[iat] = True
            molecule_atoms = []
            molecule_atoms.append(iat)
            number_of_molecules += 1
            molecule_index[iat] = number_of_molecules
            molecule_size = 1
            for kat in range(nat):
                if belongs_to[kat] and molecule_index[kat] == number_of_molecules:
                    for jat in range(nat):
                        cutoff_distance = factor_cov*(rcovs[kat]+rcovs[jat])
                        if not belongs_to[jat] and distances[kat,jat] < cutoff_distance:
                            molecule_size = molecule_size + 1
                            belongs_to[jat] = True
                            molecule_index[jat] = number_of_molecules
                            molecule_atoms.append(jat)
            if verbose:
                molecules.append(molecule_atoms)
    #  print(number_of_molecules)
    return belongs_to, molecules

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)




def get_angles(atoms):

    ana = Analysis(atoms)
    AlHBonds = ana.get_bonds('Al', 'H', unique=True)[0]


    bond_lst = []

    for bond in AlHBonds:
        for number in bond:
            bond_lst.append(number)
    bond_lst = list(set(bond_lst))
    atoms = atoms[bond_lst]
    #write("toto.xyz", atoms)


    distances = get_minimal_pairwise_distances(atoms)

    # Get the covalent radii of the elements
    elements = atoms.get_atomic_numbers()
    rcovs = get_rcovs(elements)

    # Initializations
    belongs_to, molecules = get_molecules(atoms, distances, rcovs, factor_cov=0.7)


    lattice = atoms.get_cell()
    all_positions = atoms.get_positions().copy()
    for molecule_idx in molecules:
        molecule = atoms[molecule_idx]
        molecule_positions = molecule.get_positions()
        optimal_molecule_positions = molecule.get_positions().copy()
        ankor_position = molecule_positions[0]
        for i,pos in enumerate(molecule_positions):
            min_distance = np.sum((ankor_position- pos)**2)
            index_list=[0, 0, 0]
            for ix in range(-1,2,1):
                for iy in range(-1,2,1):
                    for iz in range(-1,2,1):
                        pos_new = pos - ix*lattice[0,:]  - iy*lattice[1,:] - iz*lattice[2,:]
                        distance = np.sum((ankor_position- pos_new)**2)
                        if distance < min_distance:
                            min_distance = distance
                            index_list=[ix, iy, iz]
            optimal_position = pos - index_list[0]*lattice[0,:]  - index_list[1]*lattice[1,:] - index_list[2]*lattice[2,:]
            optimal_molecule_positions[i,:] = optimal_position

        molecule.set_positions(optimal_molecule_positions)
        all_positions[molecule_idx,:] = optimal_molecule_positions
        atoms.set_positions(all_positions)



    molecule1 = atoms[molecules[0]]
    molecule2 = atoms[molecules[1]]
    mols = [molecule1, molecule2]

    vectors = []

    for mol in mols:
        mol_vectors = []
        h_atoms = [mol.index for mol in mol if mol.symbol == 'H']
        al_atoms = [mol.index for mol in mol if mol.symbol == 'Al']

        positions = mol.get_positions()

        for hindex in h_atoms:
            vector = positions[al_atoms[0],:] - positions[hindex,:]        
            mol_vectors.append(vector)
        vectors.append(mol_vectors)

    molecule1_vectors = vectors[0] 
    molecule2_vectors = vectors[1] 

    angles = []
    for vec1 in molecule1_vectors:
        for vec2 in molecule2_vectors:
            angle = angle_between(vec1, vec2) * 180./np.pi
            angles.append(angle)


    return sorted(angles)


if __name__ == "__main__":
    atoms = read("input.extxyz")
    angles = get_angles(atoms)
    print(angles)


