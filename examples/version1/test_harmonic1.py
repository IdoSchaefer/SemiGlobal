# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 19:05:02 2024

@author: Ido Schaefer
"""

import numpy as np
from scipy.linalg import norm
from SG1funs import SemiGlobal1, SGdata
from FourierGrid import Hpsi, xp_grid


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
fi0 = np.pi**(-1/4)*np.exp(-x**2/2)*np.sqrt(dx)
# The output time grid:
dt = 0.1
t = np.r_[0:(T + dt):dt]
def Gop(u, t, v):
    return -1j*Hpsi(K, V + x*np.cos(t), v)
def Gdiff_op(u1, t1, u2, t2):
    return -1j*(xcolumn*(np.cos(t1) - np.cos(t2)))*u1
dat = SGdata(13, 15, 10, 16)
print('Chebyshev algorithm:')
U1, history1 = SemiGlobal1(Gop, Gdiff_op, 0, fi0, t, Nts, Nt_ts, Nfm, tol, ev_domain=np.r_[-188*1j, 1j], data=dat)
print("The mean number of iterations per time-step:", history1['mniter'])
# (should be close to 1, for ideal efficiency)
print('The number of matrix vector multiplications:', history1['matvecs'])
# Computation of the maximal error - the deviation from the analytical
# result of the expectation value.
# Computing the analytic expectation value of x at all the time points:
mx_ex = (-0.5*np.sin(t)*t)
# Computing the analytic expectation value of p in all the time points:
mp_ex = -0.5*(np.sin(t) + t*np.cos(t))
angle_analytic = (t/2 - (np.sin(2*t)/4 - t*np.cos(2*t)/2)/8)
# The exact analytic solution at all time-points:
Uex_phase = np.pi**(-1/4)*np.exp(-1j*angle_analytic[np.newaxis, :])*np.exp(1j*(mp_ex[np.newaxis, :]*(x[:, np.newaxis] - mx_ex[np.newaxis, :]/2)) - (x[:, np.newaxis] - mx_ex[np.newaxis, :])**2/2)*np.sqrt(dx)
Uex_fHO = Uex_phase[:, -1]
error_anal = norm(U1[:, -1] - Uex_fHO)/norm(Uex_fHO)
print('The error of the final solution from the analytic solution:', error_anal)
print('Relative difference of the estimated error and the exact error:', (history1['est_errors']['total'] - error_anal)/error_anal)
print('\nArnoldi algorithm:')
Uar1, history_ar1 = SemiGlobal1(Gop, Gdiff_op, 0, fi0, t, Nts, Nt_ts, Nfm, tol, data=dat)
print("The mean number of iterations per time-step:", history_ar1['mniter'])
# (should be close to 1, for ideal efficiency)
print('The number of matrix vector multiplications:', history_ar1['matvecs'])
# Computation of the maximal error - the deviation from the analytical
# result of the expectation value.
error_anal_ar = norm(Uar1[:, -1] - Uex_fHO)/norm(Uex_fHO)
print('The error of the final solution from the analytic solution:', error_anal_ar)
print('Relative difference of the estimated error and the exact error:', (history_ar1['est_errors']['total'] - error_anal_ar)/error_anal_ar)
