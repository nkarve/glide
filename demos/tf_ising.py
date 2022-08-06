import sys
sys.path.append('../')

from dmrg.dmrg import DMRG
from dmrg.mpo import IsingHamiltonian
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from timeit import default_timer as timer

plt.style.use('ggplot')


lattice_size = 50
max_bond_dim = 10
nsweeps = 5

"""
Phases of 1D tranverse-field Ising model

critical:                      J = g, entanglement entropy shows CFT-like spectrum
symmetry-broken ground state:  J = 0, entanglement entropy = 0 (product state), <Sz> = ±g, <E/site> = -g
symmetric ground state:        J = ∞, entanglement entropy = 0 (product state), <Sz> = 0,  <Sx> = -J 

"""

J = 2.
g = 2.

start = timer()
ising_mpo = IsingHamiltonian(lattice_size, J, g) 
dmrg = DMRG(ising_mpo, max_bond_dim)
dmrg.sweep(nsweeps)
end = timer()

print(end - start)

# Take expectation values for system and divide by lattice size for one-site expectation values, from translational symmetry
print('Expectation value of magnetization =', dmrg.get_xval(ising_mpo.mag) / lattice_size) 
print('Expectation value of Sx =', dmrg.get_xval(ising_mpo.Sx) / lattice_size)
print('Energy per site =', dmrg.energies[-1] / lattice_size)


plt.figure()

plt.title('Ground State Energy')
plt.xlabel('Iteration')
plt.ylabel('Computed ground state energy after n iterations')
plt.plot(dmrg.energies)


plt.figure()

plt.title('Site-site entanglement entropies')
plt.xlabel('Site-site bond')
plt.ylabel('Entanglement entropy')

x = np.arange(1, lattice_size)
dx = np.linspace(1, lattice_size-1, 200)

plt.scatter(x, dmrg.entropies[1:-1], marker='*', label='DMRG predictions', color='blue')

# CFT prediction of entanglement entropy for critical TFIM, see hep-th/0405152 [Cardy]
if abs(J - g) < 1e-6:
    f = lambda x, c, k: c/6 * np.log(2 * lattice_size / np.pi * np.sin(np.pi * x / lattice_size)) + k
    fit_params = curve_fit(f, x, dmrg.entropies[1:-1])
    plt.plot(dx, f(dx, *fit_params[0]), label=f'Fit with central charge = {fit_params[0][0]}')

plt.legend(loc='best')

plt.show()
