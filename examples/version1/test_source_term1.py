# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 13:30:41 2024

@author: Ido Schaefer
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import norm
from SG1funs import SemiGlobal1, SGdata
from FourierGrid import Hpsi, xp_grid
# The program tests the efficiency of Hillel Tal-Ezer's
# semi-global propagator for a forced harmonic oscillator with an additional
# arbitrary source term.
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
fi0 = (np.pi**(-1/4)*np.exp(-x**2/2)*np.sqrt(dx)).astype(np.complex128)
# The output time grid:
dt = 0.1
t = np.r_[0:(T + dt):dt]
def Gop(u, t, v):
    return -1j*Hpsi(K, V + x*np.cos(t), v)
def Gdiff_op(u1, t1, u2, t2):
    return -1j*(xcolumn*(np.cos(t1) - np.cos(t2)))*u1
def ihfun(t):
    return np.exp(-xcolumn**2/2)*np.cos(t)
dat = SGdata(13, 15, 10, 16)
print('Semi-global Chebyshev algorithm:')
U, history = SemiGlobal1(Gop, Gdiff_op, 0, fi0, t, Nts, Nt_ts, Nfm, tol, ihfun, ev_domain=np.r_[-188*1j, 1j], data=dat)
print("The mean number of iterations per time-step:", history['mniter'])
# (should be close to 1, for ideal efficiency)
print('The number of matrix vector multiplications:', history['matvecs'])
print('\nSemi-global Arnoldi algorithm:')
Uar, history_ar = SemiGlobal1(Gop, Gdiff_op, 0, fi0, t, Nts, Nt_ts, Nfm, tol, ihfun, data=dat)
print("The mean number of iterations per time-step:", history_ar['mniter'])
# (should be close to 1, for ideal efficiency)
print('The number of matrix vector multiplications:', history_ar['matvecs'])
print('\nComputing error from the RK45 solution for a tiny tolerance parameter.')
def RKfun(t, u):
    return -1j*Hpsi(K, V + x*np.cos(t), u) + np.exp(-x**2/2)*np.cos(t)
solution = solve_ivp(RKfun, (0, T), fi0, method='RK45', t_eval=t, atol=1e-12, rtol=1e-11)
URK = solution.y[:, -1]
errorCheb = norm(U[:, -1] - URK)/norm(URK)
errorAr = norm(Uar[:, -1] - URK)/norm(URK)
print('Chebyshev algorithm error:', errorCheb)
print('Arnoldi algorithm error:', errorAr)