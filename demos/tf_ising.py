import sys
sys.path.append('../')

from dmrg.dmrg import DMRG
from dmrg.mpo import IsingMPO
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.style.use('ggplot')

lattice_size = 50
max_bond_dim = 10
nsweeps = 2

ising_mpo = IsingMPO(lattice_size, J=1., g=1.) # critical tranverse-field 1D Ising model
dmrg = DMRG(ising_mpo, max_bond_dim)

dmrg.sweep(nsweeps)

print('Expectation value of magnetization =', dmrg.get_xval(ising_mpo.mag) / lattice_size)
print('Expectation value of Sx =', dmrg.get_xval(ising_mpo.Sx) / lattice_size)


plt.figure()

plt.title('Ground State Energy')
plt.xlabel('Iteration')
plt.ylabel('Computed ground state energy after n iterations')
plt.plot(dmrg.energies)


plt.figure()

plt.title('Site-site entanglement entropies')
plt.xlabel('Site-site bond')
plt.ylabel('Entanglement entropy')

# CFT prediction of entanglement entropy for critical TFIM, see Cardy 0405152
f = lambda x, c, k: c/6 * np.log(2 * lattice_size / np.pi * np.sin(np.pi * x / lattice_size)) + k
x = np.arange(1, lattice_size)
fit_params = curve_fit(f, x, dmrg.entropies[1:-1])

dx = np.linspace(1, lattice_size-1, 200)
plt.plot(dx, f(dx, *fit_params[0]), label=f'Fit with central charge = {fit_params[0][0]}') # central charge should be 0.5
plt.scatter(x, dmrg.entropies[1:-1], marker='*', label='DMRG predictions', color='blue')

plt.legend(loc='best')


plt.show()
