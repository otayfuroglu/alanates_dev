#
from ase.io import read, write


mh_energies = [atoms.get_potential_energy() for atoms in read("vasp_opt_isolated_mh.extxyz", index=":")]
for atoms in read("vasp_opt_isolated_rotated_all.extxyz", index=":"):
    energy = atoms.get_potential_energy()
    if any([f"{energy:.3f}" == f"{en_in:.3f}" for en_in in mh_energies]):
    #  if f"{atoms.get_potential_energy():.3f}" == f"{atoms2.get_potential_energy():.3f}":
        write("identical_structures.extxyz", atoms, append=True)

