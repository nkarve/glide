import numpy as np

# MPO for the spin-x operator
class SxMPO:
    def __init__(self, L, J) -> None:
        self.L = L # Number of sites
        self.J = J # Sx coupling constant
        self.d = 2 # Hilbert space dimension
        self.D = 2 # MPO internal dimension
        
        sx = np.array([[0., 1.], [1., 0.]])
        id = np.identity(2)

        self.w = []

        wl = np.zeros((1, self.D, self.d, self.d))
        wm = np.zeros((self.D, self.D, self.d, self.d))
        wr = np.zeros((self.D, 1, self.d, self.d))

        wl[0, 0], wl[0, 1] = -J * sx, id
        wm[0, 0], wm[1, 0], wm[1, 1] = id, -J * sx, id
        wr[0, 0], wr[1, 0] = id, -J * sx

        self.w.append(wl)
        for _ in range(L-2): self.w.append(wm)
        self.w.append(wr)

    def __getitem__(self, i):
        return self.w[i]

    def __setitem__(self, k, v):
        self.w[k] = v

class MagMPO:
    def __init__(self, L, g) -> None:
        self.L = L # Number of sites
        self.g = g # Sz coupling constant
        self.d = 2 # Hilbert space dimension
        self.D = 2 # MPO internal dimension
        
        sz = np.array([[1., 0.], [0., -1.]])
        id = np.identity(2)

        self.w = []

        wl = np.zeros((1, self.D, self.d, self.d))
        wm = np.zeros((self.D, self.D, self.d, self.d))
        wr = np.zeros((self.D, 1, self.d, self.d))

        wl[0, 0], wl[0, 1] = -g * sz, id
        wm[0, 0], wm[1, 0], wm[1, 1] = id, -g * sz, id
        wr[0, 0], wr[1, 0] = id, -g * sz

        self.w.append(wl)
        for _ in range(L-2): self.w.append(wm)
        self.w.append(wr)

    def __getitem__(self, i):
        return self.w[i]

    def __setitem__(self, k, v):
        self.w[k] = v

class IsingMPO:
    def __init__(self, L, J, g) -> None:
        self.L = L # Number of sites
        self.J = J # Sx coupling constant
        self.g = g # Sz coupling constant 
        self.d = 2 # Hilbert space dimension
        self.D = 3 # MPO internal dimension
        
        sx = np.array([[0., 1.], [1., 0.]])
        sz = np.array([[1., 0.], [0., -1.]])
        id = np.identity(2)

        self.w = []

        wl = np.zeros((1, self.D, self.d, self.d))
        wm = np.zeros((self.D, self.D, self.d, self.d))
        wr = np.zeros((self.D, 1, self.d, self.d))

        wl[0, 0], wl[0, 1], wl[0, 2] = id, sx, -g * sz
        wm[0, 0], wm[0, 1], wm[0, 2], wm[1, 2], wm[2, 2] = id, sx, -g * sz, -J * sx, id
        wr[0, 0], wr[1, 0], wr[2, 0] = -g * sz, -J * sx, id

        self.w.append(wl)
        for _ in range(L-2): self.w.append(wm)
        self.w.append(wr)

        self.mag = MagMPO(self.L, self.g) # Associated magnetisation MPO
        self.Sx = SxMPO(self.L, self.J)   # Associated Sx MPO

    def __getitem__(self, i):
        return self.w[i]

    def __setitem__(self, k, v):
        self.w[k] = v
