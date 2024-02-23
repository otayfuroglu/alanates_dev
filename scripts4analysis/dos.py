import numpy as np
from ase.io import read, write


one = read("isolated_96_atoms_all.extxyz", index=":")
two = read("polymeric_96_atoms_all.extxyz", index=":")

all_s = one + two

print(len(one))
print(len(two))
print(len(all_s))

min_energy = min([x.get_potential_energy() for x in all_s])

energies_one = [x.get_potential_energy()-min_energy for x in one]
energies_two = [x.get_potential_energy()-min_energy for x in two]



na = 800
dos_one = np.zeros((na))
dos_two = np.zeros((na))
de = .03

for en in energies_one:
    for ip in range(na):
        e = ip*0.01
        dos_one[ip] = dos_one[ip] + np.exp(-.5*((e-en)/de)**2)


for en in energies_two:
    for ip in range(na):
        e = ip*0.01
        dos_two[ip] = dos_two[ip] + np.exp(-.5*((e-en)/de)**2)

f=open("dos_96.dat", "w")
for i in range(na):
    e = i*0.01
    f.write(str(e)+'   '+str(dos_one[i])+'    '+str(dos_two[i])+'\n')
f.close()

