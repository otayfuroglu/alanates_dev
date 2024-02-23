from ase.io import read, write
from matplotlib import pyplot as plt
import argparse
import os

plt.rcParams["figure.figsize"] = (15,5)
parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-extxyz_path", type=str, required=False, default=None, help="..")
args = parser.parse_args()

extxyz_path = args.extxyz_path
atoms_list = read(extxyz_path, index=":")
energies = []
for atoms in atoms_list:
    energy = atoms.get_potential_energy()
    if any([f"{energy:.2f}" == f"{en_in:.2f}" for en_in in energies]):
        continue
    #  energies += [energy / len(atoms)]
    energies += [energy]

    write(f"uncorrelated_{extxyz_path.split('/')[-1]}", atoms, append=True)
    #  vol = atoms.get_volume()
    #  if 240 < vol < 300:
        #  continue
    #  if 141.7 < vol < 143.35 :
        #  continue
    #  vols += [vol]
    #  write("for_traindata2.extxyz", atoms, append=True)
    #  write("for_tight_edffg.extxyz", atoms, append=True)

print(len(energies))
plt.plot(range(len(energies)), energies, ".", label=f"{extxyz_path.split('.')[0]}")
plt.legend()
#  plt.xlabel("Configurations")
#  plt.ylabel("Volume (A^3)")
plt.ylabel("Energy (eV)")
plt.show()

quit()


#  for atoms in read("./vasp_opt_for_tight_edffg.extxyz", index=":"):
#  for atoms in read("isolated.extxyz", index=":"):
#  for atoms in read("vasp_opt_isolated_rotated_all.extxyz", index=":"):
#  extxyz_path_ls = [path for path in os.listdir(".") if "extxy" in path]

for i, extxyz_path in enumerate(extxyz_path_ls):
    #  print(extxyz_path)
    vols = []
    energies = []
    try:
        atoms_list = read(extxyz_path, index=":")
    except:
        continue
    for atoms in atoms_list:
        write("all_minimahoppin_n2p2_structures.extxyz", atoms, append=True)
        energies += [atoms.get_potential_energy() / len(atoms)]
        #  vol = atoms.get_volume()
        #  if 240 < vol < 300:
            #  continue
        #  if 141.7 < vol < 143.35 :
            #  continue
        #  vols += [vol]
        #  write("for_traindata2.extxyz", atoms, append=True)
        #  write("for_tight_edffg.extxyz", atoms, append=True)
    plt.plot(range(len(energies)), sorted(energies), ".", label=f"{extxyz_path.split('.')[0]}")
    plt.legend()
    #  plt.xlabel("Configurations")
    #  plt.ylabel("Volume (A^3)")
    plt.ylabel("Energy (eV)")
    #  plt.savefig(f"energies_{extxyz_path.split('.')[0]}.png")
    #  plt.savefig(f"energies_{extxyz_path.split('.')[0]}.png")
    #  plt.savefig(f"energies.png")
    #  plt.show()
    #  break


#  vols2 = []
#  energies2 = []
#  for atoms in read("tight_vasp_opt_isolated_rotated_v1.extxyz", index=":"):
#      energies2 += [atoms.get_potential_energy() / len(atoms)]
#      vol = atoms.get_volume()
#      #  if 240 < vol < 300:
#          #  continue
#      #  if 141.7 < vol < 143.35 :
#          #  continue
#      vols2 += [vol]

#  plt.plot(range(len(energies)), energies)
#  plt.plot(range(len(vols)), energies, label="minima hopping")
#  plt.plot(range(len(vols2)), sorted(vols2), label="rotated")
#  plt.legend()
#  plt.xlabel("Configurations")
#  plt.ylabel("Volume (A^3)")
#  plt.ylabel("Energy (eV)")
plt.show()
#  plt.savefig("energies.png")
#  plt.savefig("energies_sorted.png")
#  plt.savefig("vols.png")
