from opt_einsum import contract
import scipy.linalg as la
import numpy as np

class MPS:
    def __init__(self, dmrg) -> None:
        self.L = dmrg.L
        self.M = [np.random.randn(dmrg.bond_dims[i], dmrg.mpo.d, dmrg.bond_dims[i+1]) for i in range(self.L)]

    def __getitem__(self, i):
        return self.M[i]

    def __setitem__(self, k, v):
        self.M[k] = v

    # turns MPS into left-canonical form via QR decomposition
    def left_normalise(self):
        B = []

        for i in range(self.L - 1):
            a, d, b = self.M[i].shape
            m = self.M[i].reshape((a * d, b))

            Q, R = la.qr(m, mode='economic')
            B.append(Q.reshape(a, d, b))
            
            self.M[i+1] = contract('ij,jkl', R, self.M[i+1])
        
        B.append(self.M[-1])

        self.M = B

    # turns MPS into right-canonical form via RQ decomposition
    def right_normalise(self):
        B = []
    
        for i in range(self.L - 1, 0, -1):
            a, d, b = self.M[i].shape
            m = self.M[i].reshape((a, d * b))

            R, Q = la.rq(m, mode='economic')
            B.append(Q.reshape(a, d, b))
        
            self.M[i-1] = contract('ijk,kl', self.M[i-1], R)

        B.append(self.M[0])

        self.M = B[::-1]
    
    # performs Schmidt decomposition on MPS site to be left-normalised; truncates singular values
    def get_site_lnorm(self, i, trunc, tol=1e-8):
        l, d, r = self.M[i].shape
        temp = self.M[i].reshape(l * d, r)
        U, S, Vh = la.svd(temp, full_matrices=False)

        n = min(trunc, S.shape[0]) 
        U, S, Vh = U[:, 0:n], S[0:n], Vh[0:n, :]

        S /= la.norm(S)

        return U.reshape((l, d, n)), S, Vh

    # performs Schmidt decomposition on MPS site to be right-normalised; truncates singular values
    def get_site_rnorm(self, i, trunc, tol=1e-8):
        l, d, r = self.M[i].shape
        temp = self.M[i].reshape(l, d * r)
        U, S, Vh = la.svd(temp, full_matrices=False)

        n = min(trunc, S.shape[0])
        U, S, Vh = U[:, 0:n], S[0:n], Vh[0:n, :]

        S /= la.norm(S)

        return U, S, Vh.reshape((n, d, r))

    # returns whether each MPS site is left-normalised
    def is_lnormed(self):
        res = []
        for m in self.M:
            temp = contract('ijk,kjl', m.conj().T, m)
            res.append(np.allclose(temp, np.eye(temp.shape[0])))
        return res
    
    # returns whether each MPS site is right-normalised
    def is_rnormed(self):
        res = []
        for m in self.M: 
            temp = contract('ijk,kjl', m, m.conj().T)
            res.append(np.allclose(temp, np.eye(temp.shape[0])))
        return res