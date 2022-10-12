# A multipoint vorticity mixed finite element method for incompressible Stokes flow

Source code and examples for the paper <br>
"*A multipoint vorticity mixed finite element method for incompressible Stokes flow*" by Wietse M. Boon and Alessio Fumagalli.<br>
See [arXiv pre-print](https://arxiv.org/abs/2208.13540).

# Abstract
We propose a mixed finite element method for Stokes flow with one degree of freedom per element and facet of simplicial grids. The method is derived by considering the vorticity-velocity-pressure formulation and eliminating the vorticity locally through the use of a quadrature rule. The discrete solution is pointwise divergence-free and the method is pressure robust. The theoretically derived convergence rates are confirmed by numerical experiments.

# Citing
If you use this work in your research, we ask you to cite the following publication [arXiv pre-print](https://arxiv.org/abs/2208.13540).

# PyGeoN version
If you want to run the code you need to install [PyGeoN](https://github.com/compgeo-mox/pygeon) and might revert it to the following tag
v0.1.0. <br>
Newer versions of PyGeoN may not be compatible with this repository.

# PorePy version
If you want to run the code you need to install [PorePy](https://github.com/pmgbergen/porepy) and might revert it to the following hash
f68781ee88530b44440465f4d083243a326e08a7. <br>
Newer versions of PorePy may not be compatible with this repository.

# License
See [license](./LICENSE).
