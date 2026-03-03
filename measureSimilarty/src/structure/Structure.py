import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
from typing import Sequence
import yaml
import shlex
from io import StringIO
import xml

ang2bohr = 1.8897259886
ev2ha    = 0.03674932539796232
ev2ry    = 0.073498688455102

elementSymbolToNumber = {'H': 1,'He': 2,'Li': 3,'Be': 4,'B': 5,'C': 6,'N': 7,'O': 8,'F': 9,'Ne': 10,
                         'Na': 11,'Mg': 12,'Al': 13,'Si': 14,'P': 15,'S': 16,'Cl': 17,'Ar': 18,'K': 19,'Ca': 20,
                         'Sc': 21,'Ti': 22,' V': 23,'Cr': 24,'Mn': 25,'Fe': 26,'Co': 27,'Ni': 28,'Cu': 29,'Zn': 30,
                         'Ga': 31,'Ge': 32,'As': 33,'Se': 34,'Br': 35,'Kr': 36,'Rb': 37,'Sr': 38,'Y': 39,'Zr': 40,
                         'Nb': 41,'Mo': 42,'Tc': 43,'Ru': 44,'Rh': 45,'Pd': 46,'Ag': 47,'Cd': 48,'In': 49,'Sn': 50,
                         'Sb': 51,'Te': 52,'I': 53,'Xe': 54,'Cs': 55,'Ba': 56,'La': 57,'Ce': 58,'Pr': 59,'Nd': 60,
                         'Pm': 61,'Sm': 62,'Eu': 63,'Gd': 64,'Tb': 65,'Dy': 66,'Ho': 67,'Er': 68,'Tm': 69,'Yb': 70,
                         'Lu': 71,'Hf': 72,'Ta': 73,'W': 74,'Re': 75,'Os': 76,'Ir': 77,'Pt': 78,'Au': 79,'Hg': 80,
                         'Tl': 81,'Pb': 82,'Bi': 83,'Po': 84,'At': 85,'Rn': 86,'Fr': 87,'Ra': 88,'Ac': 89,'Th': 90,
                         'Pa': 91,'U': 92,'Np': 93,'Pu': 94,'Am': 95,'Cm': 96,'Bk': 97,'Cf': 98,'Es': 99,'Fm': 100,
                         'Md': 101,'No': 102,'Lr': 103,'Rf': 104,'Db': 105,'Sg': 106,'Bh': 107,'Hs': 108,'Mt': 109,'Ds': 110,
                         'Rg': 111,'Cn': 112,'Nh': 113,'Fl': 114,'Mc': 115,'Lv': 116,'Ts': 117,'Og': 118}

elementSymbols = [' ', ' H', 'He', 'Li', 'Be', ' B', ' C', ' N', ' O', ' F', 'Ne', 'Na', 'Mg', 'Al', 'Si', ' P', ' S',
       'Cl', 'Ar', ' K', 'Ca', 'Sc', 'Ti', ' V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
       'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', ' Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
       'In', 'Sn', 'Sb', 'Te', ' I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
       'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', ' W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
       'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', ' U', 'Np', 'Pu', 'Am', 'Cm',
       'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
       'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

atomicMasses = [0.,  1.008,    4.002,     6.94,    9.012,    10.81,   12.011,   14.007,   15.999,
                    18.998,  20.1797,   22.989,   24.305,   26.981,   28.085,   30.973,    32.06,
                     35.45,   39.948,  39.0983,   40.078,   44.955,   47.867,  50.9415,  51.9961,
                    54.938,   55.845,   58.933,  58.6934,   63.546,    65.38,   69.723,   72.630,
                    74.921,   78.971,   79.904,   83.798,  85.4678,    87.62,   88.905,   91.224,
                    92.906,    95.95,      97.,   101.07,  102.905,   106.42, 107.8682,  112.414,
                   114.818,  118.710,  121.760,   127.60,  126.904,  131.293,  132.905,  137.327,
                   138.905,  140.116,  140.907,  144.242,     145.,   150.36,  151.964,   157.25,
                   158.925,  162.500,  164.930,  167.259,  168.934,  173.045, 174.9668,  178.486,
                   180.947,   183.84,  186.207,   190.23,  192.217,  195.084,  196.966,  200.592,
                    204.38,    207.2,  208.980,     209.,     210.,     222.,     223.,     226.,
                      227., 232.0377,  231.035,  238.028,     237.,     244.,     243.,     247.,
                      247.,     251.,     252.,     257.,     258.,     259.,     262.,     267.,
                      270.,     269.,     270.,     270.,     278.,     281.,     281.,     285.,
                      286.,     289.,     289.,     293.,     293.,     294.]
class Structure:





    def __init__(self, elements, ats, energy=None, charges=None, totalCharge=None, lattice=None, forces=None, fhiAimsProperties=None, atomTypes=None, extra={}):
        self.extra = extra
        if fhiAimsProperties is None:
            fhiAimsProperties = {}
        self.nat = len(elements)
        newels = []
        for e in elements:
            if type(e) == str:
                newels.append(elementSymbolToNumber[e.strip()])
            else:
                newels.append(e)
        elements = newels
        self.distinctElements = set(elements)
        self.elements = elements
        self.ats = np.array(ats)
        self.energy = energy
        self.charges = charges
        self.totalCharge = totalCharge
        self.atomTypes = atomTypes
        if (lattice is None):
            self.lattice = None
        else:
            self.lattice = np.array(lattice)
        if forces is not None:
            self.forces = np.array(forces)
        else:
            self.forces = None
        self.fhiAimsProperties = fhiAimsProperties

        if self.lattice is not None:
            if len(self.lattice) == 0:
                self.lattice = None

      #  if self.charges is not None:
      #      qt = sum(self.charges)
      #      qt = (self.totalCharge - qt) / self.nat
      #      #print(self.totalCharge, sum(self.charges), qt - self.totalCharge)
      #      self.charges = [q + qt for q in self.charges]

        if self.charges is not None:
            if self.totalCharge is None:
                print('WARNING: Total charge not set')
            else:
                if abs(sum(self.charges)-self.totalCharge) > 0.01:
                    print('WARNING: Total charge != sum of charges. {}'.format(sum(self.charges) - self.totalCharge))


     #       if self.totalCharge is None:
     #           self.totalCharge = sum(self.charges)
     #       elif abs(sum(self.charges)-self.totalCharge) > 0.01:
     #           print("ERROR: Total charge is not sum of charges!")
     #           print(sum(self.charges), self.totalCharge)
     #           a = 1.0 / 0.0
     #           exit(1)


    def translate(self, t):
        self.ats += t


    def rotate(self, a, b, c):
        r = R.from_euler('xyz', [a,b,c])
        self.ats = r.apply(self.ats)

    def copy(self):
        return copy.deepcopy(self)

    def addFHIaims_spin(self, index, spin):
        if not index in self.fhiAimsProperties:
            self.fhiAimsProperties[index] = []
        self.fhiAimsProperties[index] += ['initial_moment {}'.format(spin)]

    def addFHIaims_charge(self, index, spin):
        if not index in self.fhiAimsProperties:
            self.fhiAimsProperties[index] = []
        self.fhiAimsProperties[index] += ['initial_charge {}'.format(spin)]

    def addFHIaims_constraint(self, index):
        if not index in self.fhiAimsProperties:
            self.fhiAimsProperties[index] = []
        self.fhiAimsProperties[index] += ['constrain_relaxation .true. ']

    def removeAtoms(self, indices):
        ats = []
        charges = []
        elems = []
        fhiprops = {}
        j = 0
        for i in range(self.nat):
            if i in indices:
                continue
            ats.append(self.ats[i])
            if self.charges is not None:
                charges.append(self.charges[i])
            elems.append(self.elements[i])
            if i in self.fhiAimsProperties:
                fhiprops[j] = self.fhiAimsProperties[i]
            j += 1

        if len(charges) == 0:
            charges=None
        return Structure(elems, np.array(ats), charges=charges, fhiAimsProperties=fhiprops, lattice=self.lattice)


    def addAtom(self, el, pos):
        self.nat += 1
        els = list(self.elements)
        els.append(el)
        self.elements = np.array(els)
        ats = list(self.ats)
        ats.append(pos)
        self.ats = np.array(ats)
        self.energy = None
        self.charges = None




    def getTotalCharge(self):
        return sum(self.charges)

    def compareElements(self, other):
        if self.nat != other.nat:
            return False
        for e1,e2 in zip(self.elements, other.elements):
            if e1.strip() != e2.strip():
                return False
        return True

    def __add__(self, other):
        if self.compareElements(other):
            return Structure(self.elements, ats=self.ats+other.ats)
        else:
            raise Exception("Structures dont match")



    # deprecated
    def removeAtom_old(self, index):
        def removeOne(thing):
            if thing is not None:
                del thing[index]
        if self.charges is not None and self.totalCharge is not None:
            self.totalCharge = self.totalCharge - self.charges[index]
        removeOne(self.elements)
        if isinstance(self.ats, list):
            removeOne(self.ats)
        elif isinstance(self.ats, np.ndarray):
            self.ats = np.delete(self.ats, index, axis=0)
        else:
            print('ERROR: Unknown type of ats')
            exit(1)
        removeOne(self.charges)
        removeOne(self.forces)

        self.distinctElements = set(self.elements)
        self.nat -= 1

    @staticmethod
    def readFhiAimsGeometry(fname):
        f = open(fname, 'r')
        pos = []
        els = []
        lat = []

        for line in f:
            s = line.split()
            if 'lattice_vector' in line:
                lat.append([float(x) * ang2bohr for x in s[1:4]])
            if 'atom' in line:
                pos.append([float(x) * ang2bohr for x in s[1:4]])
                els.append(elementSymbolToNumber[s[4]])
        f.close()
        return Structure(els, np.array(pos), lattice=(None if len(lat) == 0 else np.array(lat)))

    @staticmethod
    def readFhiAims(f):
        #print(f)
        f = open(f, 'r')

        structs = []
        hasLat = False
        lat = []
        qtot = 0.
        extra = {}
        qs = None

        while True:
            l = f.readline()
            if 'Input file geometry.in not found.' in l:
                return None
            if 'Charged system requested:' in l:
                qtot = float(l.split()[5][:-1])
                #print('qtot', qtot)
                qtot = round(qtot)
            if 'The structure contains' in l:
                nat = int(l.split()[3])
                #print('nat=', nat)
                break
        while True:
            l = f.readline()
            if 'Input geometry:' in l:
                while not 'Atomic structure' in l:
                    l = f.readline()
                    if 'Unit cell' in l:
                        for i in range(3):
                            l = f.readline()
                            lat.append([float(x) * ang2bohr for x in l.split()[1:4]])

                ats = []
                elems = []
                f.readline()
                for i in range(nat):
                    l = f.readline()
                    elems.append(l.split()[3])
                    ats.append([float(x) * ang2bohr for x in l.split()[4:7]])
                    # print('elems=', elems)
            if 'Writing Kohn-Sham eigenvalues.' in l: # why are they printed before the converged statement. Makes no sense!
                l = f.readline()
                l = f.readline()
                evs = []
                while True:
                    l = f.readline()
                    if l.strip() == '':
                        break
                    s = l.split()
                    # state, occupation, eigenvalue [Ha], eigenvalue [eV]
                    evs.append((int(s[0]), float(s[1]), float(s[2]), float(s[3])))
                    extra['KS-eigenvalues'] = evs
            if 'Electronic self-consistency reached' in l:
                break
            if not l:
                return None
            if 'SCF cycle not converged' in l:
                print('Not converged!')
                return None

        while True:
            l = f.readline()
            if not l:
                break
            if 'Performing Hirshfeld analysis of fragment charges and moments' in l:
                qs = []
                while len(qs) < nat:
                    l = f.readline()
                    if 'Hirshfeld charge' in l:
                        qs.append(float(l.split()[4]))
                # print('qs=',qs)
            if '| Total charge [e] ' in l:
                qtot = float(l.split()[5])  # todo: round to next int
                qtot = round(qtot)
                # print('qtot=',qtot)
            if 'Total atomic forces (unitary ' in l:
                force = []
                for i in range(nat):
                    l = f.readline()
                    force.append([float(x) * ev2ha / ang2bohr for x in l.split()[2:5]])
                    # print('f=',force)
            if 'Energy and forces in a compact form:' in l:
                l = f.readline()
                e = float(l.split()[5]) * ev2ha
                # print('e=',e)

            if 'structure (and velocities) as used in the preceding time step:' in l:
                ats = []
                elems = []
                l = f.readline()
                for i in range(nat):
                    l = f.readline()
                    ats.append([float(x) * ang2bohr for x in l.split()[1:4]])
                    elems.append(l.split()[4])
                    l =f.readline()
                structs.append(Structure(elems, ats, energy=e, charges=qs, totalCharge=qtot, lattice=lat, forces=force, extra=extra))
                ats = []
                    #velocity.append([float(x) * ev2ha / ang2bohr for x in l.split()[1:4]])
        # if 'tomic structure (and velocities) as used in the preceding time step:' in l:
           #     ats = []
           #     elems = []
           #     lat = []
           #     while len(ats) < nat:
           #         l = f.readline()
           #         if 'atom' in l:
           #             elems.append(l.split()[4])
           #             ats.append([float(x) * ang2bohr for x in l.split()[1:4]])
           #         elif 'lattice_vector' in l:
           #             lat.append([float(x) * ang2bohr for x in l.split[1:4]])

            if 'Updated atomic structure:' in l:
                ats = []
                elems = []
                force = None
                e = None
                charges = None
                lat = []
                while len(ats) < nat:
                    l = f.readline()
                    if 'atom' in l:
                        elems.append(l.split()[4])
                        ats.append([float(x) * ang2bohr for x in l.split()[1:4]])
                    elif 'lattice_vector' in l:
                        lat.append([float(x) * ang2bohr for x in l.split()[1:4]])

            if 'Geometry optimization: Attempting to predict improved coordinates' in l:
                structs.append(Structure(elems, ats, energy=e, charges=qs, totalCharge=qtot, lattice=lat, forces=force, extra=extra))
                ats = []

            if 'Have a nice day.' in l:
                if ats:
                    structs.append(Structure(elems, ats, energy=e, charges=qs, totalCharge=qtot, lattice=lat, forces=force, extra=extra))


            #if 'Self-consistency cycle converged' in l:
            #    qerr = sum(qs) - qtot
            #    qs = [q - qerr / len(qs) for q in qs]
            #    print('*')
            #    structs.append(Structure(elems, ats, energy=e, charges=qs, totalCharge=qtot, lattice=lat, forces=force))
            if 'SCF cycle not converged' in l or 'Input file geometry.in not found.' in l:
                print(l)
                print('- - - - - - - - - - - - - - - - - - - - - - - - - - -')
                return None

        if ats:
            structs.append(Structure(elems, ats, energy=e, charges=qs, totalCharge=qtot, lattice=lat, forces=force, extra=extra))

        f.close()

        return structs

    @staticmethod
    def readFhiAimsMD(fname):
        structs = []
        with open(fname, 'r') as f:

            line = f.readline()
            while line:
                if 'Self-consistency cycle converged.' in line:
                    charges = []
                    forces = []
                    positions = []
                    elements = []
                if 'Hirshfeld charge        :' in line:
                    charges.append(float(line.split()[4]))
                if 'Total energy uncorrected ' in line:
                    energy = float(line.split()[5])
                if 'Total atomic forces' in line:
                    line = f.readline()
                    while '|' in line:
                        forces.append([float(x) for x in line.split()[2:5]])
                        line = f.readline()
                if 'Atomic structure (and velocities) as used in the preceding time step:' in line:
                    line = f.readline() # skip header
                    line = f.readline()
                    while 'atom' in line:
                        positions.append([float(x) for x in line.split()[1:4]])
                        elements.append(elementSymbolToNumber[line.split()[4]])
                        line = f.readline() # skip velocity
                        line = f.readline()
                    structs.append(Structure(elements, np.array(positions) * ang2bohr, energy=energy * ev2ha, charges=charges,
                                 forces=np.array(forces) * ev2ha / ang2bohr))
                    print(len(structs))
                line = f.readline()
        print('done')
        return structs




    @staticmethod
    def readYamlFull(fname):
        with open(fname, 'r') as stream:
            structs = []
            ydata = yaml.load_all(stream)
            for d in ydata:
                lat = None
                conf = d['conf']
                bc = conf['bc']
                if bc == 'free':
                    lat = None
                elif bc == 'bulk':
                    # todo: check array order
                    lat = np.array([[float(x) for x in v] for v in conf['cell']])
                else:
                    raise Exception('Boundary condition {} not supported'.format(bc))
                e = float(conf['epot'])
                ats = np.array([[float(x) for x in v[:3]] for v in conf['coord']])
                if 'force' in conf:
                    force = np.array([[float(x) for x in v] for v in conf['force']])
                else:
                    force = None
                nat = int(conf['nat'])
                els = [x[3] for x in conf['coord']]
                if (conf['units_length'] == 'angstrom'):
                    ats *= ang2bohr
                    # todo: assuming eV here. Is this correct?
                    if force is not None:
                        force *= ev2ha / ang2bohr
                    e *= ev2ha
                else:
                    raise Exception('Unknown units {}'.format(conf['units_length']))

                print('l')
                s = Structure(els, ats, energy=e, charges=None, totalCharge=None, lattice=lat, forces=force)
                structs.append(s)
                yield(s)
           # return structs
        raise Exception('Parsing YAML failed')



    @staticmethod
    def readExtXYZ(fname, n=-1):
        structs = []
        with open(fname, 'r') as f:
            while True:
                atomic = False
                energy = None
                forces = []
                l = f.readline()
                if not l:
                    return structs
                else:
                    nat = int(l.split()[0])
                    if len(l.split()) > 1:
                        energy = float(l.split()[1]) * ev2ha

                props = f.readline()
                if 'atomic' in props.lower():
                    atomic = True
                s = shlex.split(props)
                pdict = {}
             #   for p in s:
             #       [k, v] = p.split('=')
             #       pdict[k] = v
             #   energy = float(pdict['energy'])
                els = []
                ats = []
                fs = []
                for i in range(nat):
                    s = f.readline().split()
                    els.append(elementSymbolToNumber[s[0]])
                    ats.append([float(x) * ang2bohr for x in s[1:4]])
                    if len(s) > 4:
                        forces.append([float(x) * ev2ha / ang2bohr for x in s[7:10]])
             #       fs.append([float(x) for x in s[4:7]])
                if atomic:
                    ats = np.array(ats) / ang2bohr
                    forces = np.array(forces) / ev2ha * ang2bohr
                    energy /= ev2ha
                else:
                    ats = np.array(ats)
                    forces = np.array(forces)

                structs.append(Structure(els, ats, energy=energy, forces=forces)) #, energy=energy, forces=np.array(fs)))
                if n > 0:
                    if len(structs) >= n:
                        return structs


    @staticmethod
    def readXYZ(xyzFile):
        with open(xyzFile, 'r') as f:
            nat = int(f.readline().split()[0])
            f.readline()
            ats = []
            elems = []
            for i in range(nat):
                s = f.readline().split()
                elems.append(s[0])
                ats.append([float(x) * ang2bohr for x in s[1:4]])
            ats = np.array(ats)
        return Structure(elems, ats, None, None, None)

    @staticmethod
    def readData(dataFile):
        return Structure.readDataFull(dataFile)[-1]
        with open(dataFile, 'r') as f:
            nat = 0
            ats = []
            elems = []
            charges = []
            lat = []
            energy = 0.0
            totalCharge = 0.0
            for line in f:
                if 'atom' in line:
                    s = line.split()
                    elems.append(s[4])
                    ats.append([float(x) for x in s[1:4]])
                    charges.append(float(s[5]))
                if 'energy' in line:
                    s = line.split()
                    energy = float(s[1])
                if 'lattice' in line:
                    s = line.split()
                    lat.append([float(x) for x in s[1:4]])
                if 'charge' in line:
                    s = line.split()
                    totalCharge = float(s[1])
            nat = len(ats)
            ats = np.array(ats)
        return Structure(elems, ats, energy, charges, totalCharge, lattice=lat)

    # reads all structures in the file, while readData only reads one
    @staticmethod
    def readDataFull(dataFile):
        structs = []
        with open(dataFile, 'r') as f:
            while True:
                line = f.readline()
                if not 'begin' in line:
                    break
                nat = 0
                ats = []
                forces = []
                elems = []
                lat = []
                charges = []
                energy = 0.0
                totalCharge = 0.0
                line = f.readline()
                while not 'end' in line:
                    if 'lattice' in line:
                        s = line.split()
                        lat.append([float(x) for x in s[1:4]])
                    if 'atom' in line:
                        s = line.split()
                        elems.append(s[4])
                        ats.append([float(x) for x in s[1:4]])
                        forces.append([float(x) for x in s[7:10]])
                        charges.append(float(s[5]))
                    if 'energy' in line:
                        s = line.split()
                        energy = float(s[1])
                    line = f.readline()
                    if 'charge' in line:
                        s = line.split()
                        totalCharge = float(s[1])
                nat = len(ats)
                ats = np.array(ats)
                forces = np.array(forces)
                structs.append(Structure(elems, ats, energy, charges, totalCharge, lattice=lat, forces=forces))
        return structs

    # todo: test this
    def repeat(self, n):
        ats = []
        els = []

        for x in range(n[0] + 1):
            for y in range(n[1] + 1):
                for z in range(n[2] + 1):
                    dlat = self.lattice.transpose() @ np.array((x,y,z))
                    ats.extend(self.ats + dlat)
                    els.extend(self.elements)
        return Structure(els, np.array(ats), lattice=(self.lattice.transpose() * (np.array(n)+1)).transpose())

    def center(self):
        self.ats -= np.mean(self.ats, 0)

    def __str__(self):
        s = ''
        if self.lattice is not None:
            for l in self.lattice:
                s += 'lattice {:17.10e} {:17.10e} {:17.10e}\n'.format(*l)

        for i in range(self.nat):
            s += '{:<2}: {:17.10e} {:17.10e} {:17.10e}\n'.format(self.elements[i], *self.ats[i])
        return s


    def toXyz(self):
        s = ''
        s += str(self.nat) + ' angstroem\n'
        s += '\n'
        for i in range(self.nat):
            s += '{:<2} {:17.10e} {:17.10e} {:17.10e}{}'.format(elementSymbols[self.elements[i]], *(np.array(self.ats[i]) / ang2bohr), '\n' if i<self.nat-1 else '')
        return s + '\n'

    def toFHIaims(self):
        s = ''
        if self.lattice is not None:
            for i in range(3):
                s += 'lattice_vector {:17.10e} {:17.10e} {:17.10e}\n'.format(*(np.array(self.lattice[i]) / ang2bohr))

        for i in range(self.nat):
            s += 'atom {:17.10e} {:17.10e} {:17.10e} {:<2}\n'.format(*(np.array(self.ats[i]) / ang2bohr), elementSymbols[self.elements[i]]) #''\n' if i<self.nat-1 else '')
            if i in self.fhiAimsProperties:
                s += '\n'.join(self.fhiAimsProperties[i]) + '\n'
        return s

    def toFHIaimsWithAtomNames(self):
        s = ''
        if self.lattice is not None:
            for i in range(3):
                s += 'lattice_vector {:17.10e} {:17.10e} {:17.10e}\n'.format(*(np.array(self.lattice[i]) / ang2bohr))

        for i in range(self.nat):
            s += 'atom {:17.10e} {:17.10e} {:17.10e} {:<2}\n'.format(*(np.array(self.ats[i]) / ang2bohr), elementSymbols[self.elements[i]]) #''\n' if i<self.nat-1 else '')
            if i in self.fhiAimsProperties:
                s += '\n'.join(self.fhiAimsProperties[i]) + '\n'
        return s

    def toData(self):
        s = 'begin\n'
        if self.lattice is not None:
            for l in self.lattice:
                s += 'lattice {:17.10e} {:17.10e} {:17.10e}\n'.format(*l)
        for i in range(self.nat):
            s += 'atom {:17.10e} {:17.10e} {:17.10e} {:<2} {:17.10e} 0.0 {:17.10e} {:17.10e} {:17.10e}\n'.format(
                    *self.ats[i,:],
                    elementSymbols[self.elements[i]],
                    self.charges[i] if self.charges is not None else 0.0,
                    *(self.forces[i,:] if self.forces is not None else [0.0, 0.0, 0.0])
                    )
        s += 'energy {:17.10e}\n'.format(self.energy if self.energy is not None else 0.0)
        s += 'charge {:17.10e}\n'.format(self.totalCharge if self.totalCharge is not None else 0.0)
        s += 'end\n'
        return s


    def save(self, path: str):
        f = open(path, 'w')
        if path.endswith('.xyz'):
            f.write(self.toXyz())
        elif path.endswith('.fhiaims') or path.endswith('.in'):
            f.write(self.toFHIaims())
        elif path.endswith('.data'):
            f.write(self.toData())
        elif path.endswith('.ascii'):
            f.write(self.toAscii())
        elif path.endswith('.vasp'):
            f.write(self.toVASP())
        elif path.endswith('POSCAR'):
            f.write(self.toVASP())
        else:
            raise Exception('unknown file type')

        f.close()

    def saveCharge(self, path: str):
        f = open(path, 'w')
        for c in self.charges:
            f.write('{}\n'.format(c))
        f.close()

    @staticmethod
    def saveDataFull(structures, path: str):
        f = open(path, 'w')
        for s in structures:
            f.write(s.toData())
        f.close()


    def rescaleCharges(self, totalCharge=None):
        tc = totalCharge if totalCharge is not None else self.totalCharge
        ce = sum(self.charges) - tc
        self.charges = [x - ce / self.nat for x in self.charges]


    def combine(self, other):
        return Structure(np.concatenate([self.elements, other.elements]), np.concatenate([self.ats, other.ats]))


    # todo: check if lattice needs to be transposed
    def getFractionalCoordinate(self):
      #  if (np.sum((self.lattice - self.lattice.transpose())**2) > 1.e-6):
#            raise Exception('Lattice not orthonormal. Check if it needs to be transposed!')

        invlat = np.linalg.inv(self.lattice.transpose())
        return (invlat @ self.ats.transpose()).transpose()

    def fromFractionalCoordinates(self, fracc):
        self.ats = (self.lattice.transpose() @ fracc.transpose()).transpose()

    @staticmethod
    def readBigDft(fname):
        print('did you check the output file for not converged warnings?')
        # todo: print these warnings during parsing
        # grep '\- Self-consistent cycle did not meet convergence criteria' out.txt
        with open(fname, 'r') as stream:
            # this is a very hacky solution to fix the fact that
            filtered_string = ''
            good = True
            for line in stream:
                if '--- Self-Consistent Cycle' in line: # filter out the scf part since it is not a valid yaml
                    good = False
                if '---- Forces Calculation' in line:
                    good = True
                if good:
                    filtered_string += line

            ydata = yaml.load_all(StringIO(filtered_string), Loader=yaml.FullLoader)
            for d in ydata:
                if d['Atomic structure']['units'] != 'angstroem':
                    raise Exception('Wrong untis!')
                els = []
                ats = []
                for at in d['Atomic structure']['positions']:
                    e = list(at.keys())[0]
                    els.append(elementSymbolToNumber[e])
                    ats.append(at[e])
                ats = np.array(ats) * ang2bohr

                forces = []
                for f in d['Atomic Forces (Ha/Bohr)']:
                    e = list(f.keys())[0]
                    forces.append(f[e])
                energy = d['Energy (Hartree)']

                return Structure(els, ats, energy=energy, forces=forces)


    @staticmethod
    def readAscii(aFile):
        f = open(aFile, 'r')
        l = f.readline()
        nat = int(l.split()[0])
        lat = np.zeros((3,3))
        l = f.readline()
        s = [float(x) for x in l.split()]
        lat[0,0] = s[0]
        lat[1,:2] = s[1:3]
        l = f.readline()
        s = [float(x) for x in l.split()]
        lat[2,:] = s[:]
        lat *= ang2bohr
        ats = []
        els = []
        for i in range(nat):
            l = f.readline()
            s = l.split()
            ats.append([float(x) * ang2bohr for x in s[:3]])
            els.append(elementSymbolToNumber[s[3]])
        return Structure(els, ats, lattice=lat)




    @staticmethod
    def readGulp(gFile):
        f = open(gFile, 'r')
        while True:
            l = f.readline()
            if not l:
                break
            if 'Final fractional coordinates of atoms' in l:
                ats = []
                els = []
                atTypes = []
                for i in range(5):
                    l = f.readline()
                while True:
                    l = f.readline()
                    if '--' in l:
                        for i in range(3):
                            l = f.readline()
                        lat = []
                        for i in range(3):
                            l = f.readline()
                            s = l.split()
                            lat.append([float(x) for x in s[:3]])
                        ats = np.array(ats)
                        lat = np.array(lat)
                        ats = (lat @ ats.transpose()).transpose()
                        return Structure(els, ats, lattice=lat, atomTypes=atTypes)

                        return
                    s = l.split()
                    els.append(elementSymbolToNumber[''.join([x for x in s[1] if not x.isdigit()])])
                    atTypes.append(''.join([x for x in s[1] if x.isdigit()]))

                    ats.append([float(x) for x in s[3:6]])

    def toAscii(self):
        s = ''
        q, r = np.linalg.qr(self.lattice.transpose())
        if np.linalg.det(q) < 0.0:
            q *= -1
            r *= -1
        self.lattice = r.transpose()
        r = r.transpose()
        self.ats = np.array(self.ats)
        self.ats = (q.transpose() @ self.ats.transpose()).transpose()

        s += str(self.nat) + ' \n'
        s += str(self.lattice[0,0]/ang2bohr) + ' ' + str(self.lattice[1,0]/ang2bohr) + ' ' + str(self.lattice[1,1]/ang2bohr) + ' \n'
        s += str(self.lattice[2,0]/ang2bohr) + ' ' + str(self.lattice[2,1]/ang2bohr) + ' ' + str(self.lattice[2,2]/ang2bohr) + ' \n'

        for i in range(self.nat):
            s += ' '.join([str(x/ang2bohr) for x in self.ats[i,:]]) + ' ' + elementSymbols[self.elements[i]] + ' \n'
            #s += ' '.join([str(x) for x in self.ats[i, :]]) + ' ' + self.elements[i] + ' \n'
        return s


    def toVASP(self):
        invlat = np.linalg.inv(self.lattice.transpose())
        relats = (invlat @ self.ats.transpose()).transpose()

        lats = []
        for i in range(self.nat):
            lats.append((self.elements[i], relats[i,:]))
        lats = sorted(lats, key=lambda x: x[0], reverse=True)

        dels = sorted(list(set(self.elements)), reverse=True)

        s = '\n'
        s += '1.0 \n'
        for i in range(3):
            s += ' '.join(['{:17.10e}'.format(x) for x in self.lattice[i,:] / ang2bohr]) + '\n'
        for e in dels:
            s += elementSymbols[e] + ' '
        s += '\n'
        for e in dels:
            s += str(len([x for x in self.elements if x == e])) + ' '
        s += '\n'
        s += 'Direct \n'
        for el, at in lats:
            s += ' '.join(['{:17.10e}'.format(x) for x in at]) + '\n'

        return s

    @staticmethod
    def readVaspPoscar(fname):
        f = open(fname, 'r')
        f.readline()
        scaling = float(f.readline())
        lat = []
        for i in range(3):
            lat.append([float(x) for x in f.readline().split()])
        els = f.readline().split()
        nels = [int(x) for x in f.readline().split()]
        elements = [elementSymbolToNumber[x] for a,b in zip(els, nels) for x in [a] * b]
        l = f.readline()
        if l.strip() != 'Direct':
            print("ERROR: Not Direct")
            return
        nat = len(elements)
        ats = []
        for i in range(nat):
            ats.append([float(x) for x in f.readline().split()])
        ats = np.mod(np.array(ats), 1.0)
        lat = np.array(lat) * ang2bohr
        ats = (lat.transpose() @ ats.transpose()).transpose()
        return Structure(elements, ats, lattice=lat)

    @staticmethod
    def readVaspOutput(fname):
        print('does VASP reading work for geopt output?')
        if fname[-1] == '/':
            fname = fname[:-1]
        ref = Structure.readVaspPoscar(fname + '/POSCAR')
        f = open(fname + '/OUTCAR')
        ats = []
        forces = []
        while 'TOTAL-FORCE' not in f.readline():
            pass
        f.readline()
        for i in range(ref.nat):
            s = f.readline().split()
            ats.append([float(x) * ang2bohr for x in s[:3]])
            forces.append([float(x) / ang2bohr * ev2ha for x in s[3:]])
        ats = np.array(ats)
        print(np.linalg.norm(ref.ats - ats))

        ref.forces = np.array(forces)

        l = ''
        while 'TOTEN' not in l:
            l = f.readline()
        energy = float(l.split()[4]) * ev2ha
        ref.energy = energy
        fc = open(fname + '/charges.txt')
        charges = []
        for i in range(ref.nat):
            j, e, q = fc.readline().split()
            q = float(q)
            if elementSymbolToNumber[e] != ref.elements[i]:
                print(elementSymbolToNumber[e], ref.elements[i])
                raise Exception('Wrong element in charges.txt on line {}'.format(i+1))
            charges.append(q)
        fc.close()
        ref.charges = np.array(charges)
        return ref

    def moveIntoCell(self):
        fracc = self.getFractionalCoordinate()
        fracc = fracc % 1.
        self.fromFractionalCoordinates(fracc)




