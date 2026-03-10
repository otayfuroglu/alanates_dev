
from ase.io import read, write
import argparse

# Robust energy getter
def get_energy(atoms):
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    if "Energy" in atoms.info:
        return float(atoms.info["Energy"])
    if atoms.calc is not None:
        return float(atoms.get_potential_energy())
    raise ValueError("Energy not found in atoms.info or calculator results.")


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-extxyz_path", type=str, required=True, help="..")
args = parser.parse_args()

# ---------- INPUT ----------
extxyz_file = args.extxyz_path   # your ASE extxyz file
output_file = "ground_state.extxyz"
# ---------------------------

# Read all structures
atoms_list = read(extxyz_file, index=":")

# Find ground-state structure
energies = [get_energy(atoms) for atoms in atoms_list]
gs_index = min(range(len(energies)), key=lambda i: energies[i])
gs_atoms = atoms_list[gs_index]

# Save it
write(output_file, gs_atoms)

print(f"Ground-state index: {gs_index}")
print(f"Ground-state energy: {energies[gs_index]:.8f} eV")
print(f"Saved to: {output_file}")

