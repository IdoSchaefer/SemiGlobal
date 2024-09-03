# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:49:48 2022

@author: Ido Schaefer
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.fftpack import ifft
from scipy.linalg import eig, norm
from SG1funs import SemiGlobal1
from FourierGrid import Hpsi, xp_grid

def gsNLHdiag(H0, Vnl, x, tol):
    """
The function finds the ground state of a non-linear Hamiltonian by an iterative
process.
Input:
H0: A 2D ndarray; the linear part of the Hamiltonian, represented as a matrix.
Vnl: Function object of the form: Vnl(u, x); the nonlinear purterbation, where
u is the state (1D ndarray) and x is the x grid (1D ndarray)
x: 1D ndarray; the space grid
tol: The desired tolerance of the iterative process
Output:
gs: 1D ndarray; the resulting ground state
niter: The required number of iterations
"""
    eigval, P = eig(H0)
    iminE = np.argmin(eigval)
    gs = P[:, iminE]
    H = H0 + np.diag(Vnl(gs, x))
    Hgs = H@gs
    niter = 1
    maxNiter = 100
    while np.abs(Hgs@Hgs - (gs@Hgs)**2)>tol and niter<=maxNiter:
        eigval, P = eig(H)
        iminE = np.argmin(eigval)
        gs = P[:, iminE]
        H = H0 + np.diag(Vnl(gs, x))
        niter += 1
        Hgs = H@gs
    if niter>maxNiter:
        print('The program has failed to achieve the desired tolerance.')
    return gs, niter

# The program tests the efficiency of Hillel Tal-Ezer's
# semi-global propagator, for a forced harmonic oscillator.
# You may play with the following parameters:
T = 10; Nts = 200; Nt_ts = 9; Nfm = 9; tol=1e-5;
# Constructing the grid:
L = 16*np.sqrt(np.pi)
Nx = 128
dx = L/Nx
x, p = xp_grid(L, Nx)
xcolumn = x[:, np.newaxis]
# The kinetic energy matrix diagonal in the p domain:
K = p**2/2
# The potential energy matrix diagonal in the x domain:
V = x**2/2
# We have to construct the H0 matrix, to find the ground state (it's unnecessary
# for the propagation process).
# The potential energy matrix:
Vmat = np.diag(V)
# The kinetic energy matrix in the x domain:
Kmat = Nx*np.conj(ifft(np.conj(ifft(np.diag(K))).T)).T
# The Hamiltonian:
H = Kmat + Vmat
# The ground state, found by an iterative process:
gs, _ = gsNLHdiag(H, lambda u, x: np.conj(u)*u, x, 2e-12)
# The output time grid:
dt = 0.1
t=np.r_[0:(T + dt):dt]
def Gop(u, t, v):
    return -1j*Hpsi(K, V + x*np.cos(t) + np.conj(u)*u, v)
def Gdiff_op(u1, t1, u2, t2):
    u2column = u2[:, None]
    return -1j*(xcolumn*(np.cos(t1) - np.cos(t2)) + np.conj(u1)*u1 - np.conj(u2column)*u2column)*u1
print('Chebyshev algorithm:')
U, history = SemiGlobal1(Gop, Gdiff_op, 0, gs, t, Nts, Nt_ts, Nfm, tol, ev_domain=np.r_[-188*1j, 1j])
print("The mean number of iterations per time-step:", history['mniter'])
# (should be close to 1, for ideal efficiency)
print('The number of matrix vector multiplications:', history['matvecs'])
print('\nArnoldi algorithm:')
Uar, history_ar = SemiGlobal1(Gop, Gdiff_op, 0, gs, t, Nts, Nt_ts, Nfm, tol)
print("The mean number of iterations per time-step:", history_ar['mniter'])
# (should be close to 1, for ideal efficiency)
print('The number of matrix vector multiplications:', history_ar['matvecs'])
print('\nComputing error from the RK45 solution for a tiny tolerance parameter.')
def RKfun(t, u):
    return -1j*Hpsi(K, V + x*np.cos(t) + np.conj(u)*u, u)
solution = solve_ivp(RKfun, (0, T), gs, method='RK45', t_eval=t, atol=1e-13, rtol=1e-13)
URK = solution.y[:, -1]
errorCheb = norm(U[:, -1] - URK)/norm(URK)
errorAr = norm(Uar[:, -1] - URK)/norm(URK)
print('Chebyshev algorithm error:', errorCheb)
print('Arnoldi algorithm error:', errorAr)