python -m numpy.f2py -c -L/usr/lib/ -lopenblas fingerprint.f90 -m fingerprint --fcompiler=gfortran --f90flags=" -O3"
python -m numpy.f2py -c -L/usr/lib/ -lopenblas fingerprint-all.f90 -m fingerprint_all --fcompiler=gfortran --f90flags=" -O3"
