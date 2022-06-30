import itertools
import numpy as np
from opt_einsum import contract, contract_expression, get_symbol
import scipy.sparse.linalg as sla
from .mps import MPS

class DMRG:
    def __init__(self, mpo, Dmax) -> None:
        self.L = mpo.L # Number of sites
        self.Dmax = Dmax # Maximum bond dimension

        self.bond_dims = np.full(self.L + 1, mpo.d) ** np.hstack((np.arange(0, self.L / 2), np.arange(self.L / 2, -1, -1)))

        self.bond_dims[self.bond_dims > self.Dmax] = self.Dmax
        self.bond_dims = np.nan_to_num(self.bond_dims, posinf=self.Dmax, nan=self.Dmax).astype(int)

        self.mpo = mpo
        self.mps = MPS(self)
        
        self.init_xval_paths()

    # Initialise the list of left and right block tensors
    def init_edges(self):
        self.redge = [np.ones((1, 1, 1))]
        for i in range(self.L - 1, 0, -1):
            edge = contract('ijk,lmi,njmo,kop', self.redge[-1], self.mps[i], self.mpo[i], self.mps[i].conj().T)
            self.redge.append(edge)

        self.redge = self.redge[::-1]

        self.ledge = [np.ones((1, 1, 1))]
        for i in range(0, self.L - 1):
            edge = contract('ijk,ilm,jnlo,pok', self.ledge[i], self.mps[i], self.mpo[i], self.mps[i].conj().T)
            self.ledge.append(edge)

    # MPO Hamiltonian operator projected into single site basis, reshaped into matrix
    def get_effective_hamiltonian(self, i):
        H = contract('ijkl,min,ojp', self.mpo[i], self.ledge[i], self.redge[i])
        H = np.transpose(H, (2, 0, 4, 3, 1, 5))
        dim = self.mpo.d * self.bond_dims[i] * self.bond_dims[i+1]
        return H.reshape(dim, dim)

    # returns expectation value of single-site effective Hamiltonian
    def get_effective_energy(self, i):
        return contract('iln,ijk,jolm,nop,pmk', self.mps[i], self.ledge[i], self.mpo[i], self.redge[i], self.mps[i].conj().T).squeeze()

    # find the smallest eigenvalue of the single-site effective Hamiltonian using the Lanczos method in ARPACK
    def eigsolve(self, i):
        H = self.get_effective_hamiltonian(i)
        M = self.mps[i]

        mshp = M.shape
        guess = M.reshape(-1)
        eigval, eigvec = sla.eigen.arpack.eigsh(H, k=1, which='SA', v0=guess)
        return eigval[0], eigvec.reshape(mshp)

    # perform DMRG optimisation from left to right; stores entropies and incremental energies
    def sweep_right(self):
        for i in range(0, self.L):
            energy_upd, M_upd = self.eigsolve(i)
            self.energies.append(energy_upd)

            self.mps[i] = M_upd
            A, S, Vh = self.mps.get_site_lnorm(i, trunc=self.Dmax)
            self.mps[i] = A

            self.entropies[i+1] = -np.sum((S * S) * np.log(S * S))

            if i < self.L - 1:
                self.mps[i+1] = contract('ij,jk,klm', np.diag(S), Vh, self.mps[i+1])
                self.ledge[i+1] = contract('ijk,ilm,jnlo,pok', self.ledge[i], self.mps[i], self.mpo[i], self.mps[i].conj().T)

    # perform DMRG optimisation from right to left; stores entropies and incremental energies
    def sweep_left(self):      
        for i in range(self.L - 1, -1, -1):
            energy_upd, M_upd = self.eigsolve(i)
            self.energies.append(energy_upd)
            
            self.mps[i] = M_upd
            U, S, B = self.mps.get_site_rnorm(i, trunc=self.Dmax) 
            self.mps[i] = B

            self.entropies[i] = -np.sum((S * S) * np.log(S * S))

            if i > 0:
                self.mps[i-1] = contract('ijk,kl,lm', self.mps[i-1], U, np.diag(S))
                self.redge[i-1] = contract('ijk,lmi,njmo,kop', self.redge[i], self.mps[i], self.mpo[i], self.mps[i].conj().T)

    # Sweep back and forth a fixed number of times
    def sweep(self, times):
        self.mps.right_normalise()
        self.init_edges()

        self.energies = []
        self.entropies = [0.] * (self.L + 1)

        for i in range(times):
            print(f'Sweeping right ({i+1}/{times})')
            self.sweep_right()
            print(f'Sweeping left  ({i+1}/{times})')
            self.sweep_left()
    
    # Sweep back and forth until the energy converges
    def sweep_until(self, tol=1e-8):
        self.mps.right_normalise()
        self.init_edges()

        self.energies = []
        self.entropies = [0.] * (self.L + 1)

        while len(self.energies) < 2 or not np.allclose(self.energies[-1], self.energies[-2], rtol=tol):
            self.sweep_right()
            self.sweep_left()

    # compute the einsum contraction format for fast MPO expectation value calculation
    def init_xval_paths(self):
        self.paths = dict()
        self.einsum_str = 'ade,bgdf,hfc,'

        for i in range(8, 1 + 8 + 5 * (self.L - 2), 5):
            c, f, h, i, j, k, l, m = (get_symbol(i + off) for off in (-4, -2, -1, 0, 1, 2, 3, 4))
            self.einsum_str += f'{c}{i}{j},{f}{l}{i}{k},{m}{k}{h},'

        self.einsum_str = self.einsum_str[:-1]

    def cache_mpo_path(self, mpo):
        shapes = []
        for i in range(self.L):
            shapes.append(self.mps[i].shape)
            shapes.append(mpo[i].shape)
            shapes.append(self.mps[i].shape[::-1])

        expr = contract_expression(self.einsum_str, *shapes, memory_limit=-1)
        self.paths[mpo] = expr
    
    # compute the expectation value of an MPO with the current MPS
    def get_xval(self, mpo):
        if mpo not in self.paths:
            self.cache_mpo_path(mpo)

        mps_conj = [m.conj().T for m in self.mps.M]
        msorted = list(itertools.chain(*zip(self.mps.M, mpo, mps_conj)))
        return self.paths[mpo](*msorted).squeeze()
