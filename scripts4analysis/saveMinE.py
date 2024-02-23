#
from ase.io import read, write

#  atoms_list = read("./unique_structures.extxyz", index=":")
atoms_list = read("./unique_isolated.extxyz", index=":")

energies = []

i = 0
for atoms in atoms_list:
    energy = atoms.get_potential_energy()
    energies += [(i, energy)]
    i += 1


soted_energies = sorted(energies, key=lambda x: x[1])
print(soted_energies)
write("minE_strucute.extxyz", atoms_list[soted_energies[0][0]])
write("maxE_strucute.extxyz", atoms_list[soted_energies[-1][0]])
#  print(soted_energies)
