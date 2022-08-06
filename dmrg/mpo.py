import numpy as np

# Pauli operators
sx = np.array([[0., 1.], [1.,  0.]])
sy = np.array([[0., 1j], [-1j, 0.]])
sz = np.array([[1., 0.], [0., -1.]])
sp = np.array([[0., 1.], [0.,  0.]])
sm = np.array([[0., 0.], [1.,  0.]])
id = np.identity(2)

class MPO:
    def __getitem__(self, i):
        return self.w[i]

    def __setitem__(self, k, v):
        self.w[k] = v

    @staticmethod
    def from_graph(adj, map, L):
        mpo = MPO()
        mpo.L = L
        mpo.d = map[0].shape[0]
        mpo.D = len(adj)
        print(map)

        mpo.w = []

        wl = np.zeros((1, mpo.D, mpo.d, mpo.d))
        wm = np.zeros((mpo.D, mpo.D, mpo.d, mpo.d))
        wr = np.zeros((mpo.D, 1, mpo.d, mpo.d))

        for i, l in enumerate(adj):
            for j, c in l:
                wm[i, j] = map[c]

        for i in range(mpo.D): wl[0, i], wr[i, 0] = wm[0, i], wm[i, mpo.D - 1]

        mpo.w.append(wl)
        for _ in range(L-2): mpo.w.append(wm)
        mpo.w.append(wr)

        return mpo    

class NNMPO(MPO):
    def __init__(self, L, args) -> None:
        self.L = L
        self.d = args[0].shape[0]
        self.D = (1 + len(args)) // 2

        self.w = []

        wl = np.zeros((1, self.D, self.d, self.d))
        wm = np.zeros((self.D, self.D, self.d, self.d))
        wr = np.zeros((self.D, 1, self.d, self.d))

        for i in range(self.D): wl[0, i] = wm[0, i] = args[i]
        for i in range(self.D): wm[i, self.D - 1] = wr[i, 0] = args[self.D + i - 1]

        self.w.append(wl)
        for _ in range(L-2): self.w.append(wm)
        self.w.append(wr)

class SpinX(NNMPO):
    def __init__(self, L, J) -> None: 
        self.J = J 

        super().__init__(L, [id, -J * sx, id])

class Magnetisation(NNMPO):
    def __init__(self, L, g) -> None:
        self.g = g 

        super().__init__(L, [id, -g * sz, id])
        
class IsingHamiltonian(NNMPO):
    def __init__(self, L, J, g) -> None:
        self.J = J 
        self.g = g  
    
        super().__init__(L, [id, sx, -g * sz, -J * sx, id])

        self.mag = Magnetisation(L, self.g) 
        self.Sx = SpinX(L, self.J)   

class XXZ_Hamiltonian(NNMPO):
    def __init__(self, L, J, h, A) -> None:
        self.J = J
        self.h = h
        self.A = A

        super().__init__(L, [id, sp, sm, 0.5 * sz, -h * 0.5 * sz, 0.5 * J * sm, 0.5 * J * sp, 0.5 * J * A * sz, id])
        
        self.mag = Magnetisation(L, self.h) 
        self.Sx = SpinX(L, self.J)   

class HeisenbergHamiltonian(XXZ_Hamiltonian):
    def __init__(self, L, J) -> None:
        super().__init__(L, J, 0, 1)
