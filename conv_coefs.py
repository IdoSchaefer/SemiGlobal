# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:13:59 2024

@author: Ido Schaefer
"""

import numpy as np
from Crecurrence import recurrence_coefs

def convCtexp(M, p_max): # Chebyshev coefficients for the SG propagator, time-expansion
    """
The function computes the Chebyshev coefficients of g(xi) for the convergence
process of the semi-global propagator for the time expansion error.
The function returns the coefficients of all iterations up to p_max.
Output: 2D ndarray; contains the coefficients of the p'th iteration in the 
(p + 1)'th row, where different columns represent the coefficients of
different polynomial orders.
"""
    def f_rec(previousC, p):
        Nmax = previousC.size - 1
        newC = np.zeros(Nmax + 1)
        N = M + 2*p + 1
        # The previous n:
        prev_n = np.r_[3:(N - 1)]
        # The index of the order j coefficient is j
        newC[0] = (7/4)*previousC[0] + previousC[1]/6 - (35/48)*previousC[2] \
            - np.sum(previousC[prev_n]*(prev_n**2 - 7)/((prev_n**2 - 4)*(prev_n**2 - 1)))
        newC[1] = -2*previousC[0] + previousC[1]/4 + previousC[2] - previousC[3]/4
        newC[2] = previousC[0]/4 - previousC[1]/2 + previousC[3]/2 - previousC[4]/8
        ni = np.r_[3:(N - 3)]
        newC[ni] = ((previousC[ni - 2] - previousC[ni + 2])/4 - previousC[ni - 1] + previousC[ni + 1])/ni
        newC[N - 3] = (previousC[N - 5]/4 - previousC[N - 4] + previousC[N - 2])/(N - 3)
        newC[(N - 2):N] = (previousC[(N - 4):(N - 2)]/4 - previousC[(N - 3):(N - 1)])/np.array([N - 2, N - 1])
        newC[N] = previousC[N - 2]/(4*N)
        newC = newC/4
        return newC

    Cinit = np.zeros(M + 2*p_max + 2)
    if M == 3:
        Cinit[0] = 3/8
    elif M%2 == 0:
        Cinit[0] = 4/((M**2 - 1)*(M - 3))
    else:
        Cinit[0] = -4/((M**2 - 1)*(M - 3))
    Cinit[M - 1] = -1/(M - 1)
    Cinit[M + 1] = 1/(2*(M + 1))
    if M != 3:
        # This accounts also for the case of M = 2:
        Cinit[abs(M - 3)] = Cinit[abs(M - 3)] + 1/(2*(M - 3))
    Cinit = Cinit/4**M
    return recurrence_coefs(Cinit, p_max, f_rec)


def convCfm(MplusK, p_max): # Polynomial coefficients for the SG propagator, function of matrix
    """
The function computes the (xi + 1)**j coefficients of h(xi) for the convergence
process of the semi-global propagator for the function of matrix error.
The function returns the coefficients of all iterations up to p_max.
Input:
MplusK (int): Equals M + K.
p_max (int): The maximal iteration number 
Output: 2D ndarray; contains the coefficients of the p'th iteration in the 
p'th row, where different columns represent the coefficients of
different polynomial orders.
"""
    def f_rec(previousC, p):
        newC = np.zeros(previousC.size)
        # The previous n:
        prev_n = np.r_[0:(MplusK + 2*p - 1)]
        # The index of the order j coefficient is j.
        newC[0] = np.sum(previousC[prev_n]*(prev_n + 4)/((prev_n + 1)*(prev_n + 2)))/2
        newC[1] = -1.5*previousC[0]
        ni = np.r_[2:(MplusK + 2*p + 1)]
        newC[ni] = (previousC[ni - 2] - 1.5*previousC[ni - 1])/ni
        return newC
    

    Cinit = np.zeros(MplusK + 2*p_max + 1)
    Cinit[MplusK] = 1
    return recurrence_coefs(Cinit, p_max, f_rec)
   

def f_rec_convC1st(previousC, p): # Recurrence function for the use of the function convC1st
    """

"""
    newC = np.zeros(previousC.size)
    n = np.r_[(p + 1):(2*p + 2)]
    newC[n] = (previousC[n - 2] - previousC[n - 1]/2)/n
    return newC
 

def convC1st(p_max):
    """
The function computes the power coefficients of l(xi) for the convergence
process of the semi-global propagator for the first time-step. The
function returns the coefficients of all iterations up to p_max.
Output: 2D ndarray; contains the coefficients of the p'th iteration in the 
p'th row, where different columns represent the coefficients of
different orders of polynomials (the n'th column represents the coefficient
of xi**n).
"""
    Cinit = np.zeros(2*p_max + 2)
    Cinit[1] = 1
    return recurrence_coefs(Cinit, p_max, f_rec_convC1st)


def conv_gfuns(M, p_max, xivals):
    """
The function computes the values of g^{(p)}(xi) at values xivals, for all
p up to the maximal iteration number p_max.
Input:
M (int): The M value
p_max (int): The maximal iteration number
xivals (ndarray/int/float/complex): The xi values to be computed
"""
    from Chebyshev import cheb_pols
    Ctexp = convCtexp(M, p_max)
    Mcheb = cheb_pols(2*xivals + 1, M + 2*p_max + 1)
    return Ctexp@Mcheb


def conv_hfuns(MplusK, p_max, xivals):
    """
The function computes the values of h^{(p)}(xi) at values xivals, for all
p up to the maximal iteration number p_max.
Input:
MplusK (int): The value of M+K
p_max (int): The maximal iteration number
xivals (ndarray/int/float/complex): The xi values to be computed    
"""
    
    Cfm = convCfm(MplusK, p_max)
    if isinstance(xivals, np.ndarray):
        if xivals.dtype.type is np.int32:
            xivals = xivals.astype(float)
        Mxi_plus1 = (xivals + 1)**np.r_[0:(MplusK + 2*p_max + 1)][:, np.newaxis]
    else:
        if type(xivals) is int:
            xivals = float(xivals)
        Mxi_plus1 = (xivals + 1)**np.r_[0:(MplusK + 2*p_max + 1)]        
    return Cfm@Mxi_plus1


def conv_lfuns(p_max, xivals):
    """
The function computes the values of l^{(p)}(xi) at values xivals, for all
p up to the maximal iteration number p_max.
Input:
p_max (int): The maximal iteration number
xivals (ndarray/int/float/complex): The xi values to be computed    
"""

    C1st = convC1st(p_max)
    if isinstance(xivals, np.ndarray):
        if xivals.dtype.type is np.int32:
            xivals = xivals.astype(float)
        Mxi = xivals**np.r_[0:(2*p_max + 2)][:, np.newaxis]
    else:
        if type(xivals) is int:
            xivals = float(xivals)
        Mxi = xivals**np.r_[0:(2*p_max + 2)]
    return C1st@Mxi


def conv_ratios_texp(Mmax, p_max): # Convergence ratios for the time-expansion error
    """
The function computes the convergence ratios of the time-expansion
extrapolation error. The ratios are computed for all 2=<M<=Mmax and for
all iteration numbers up to p_max.
The coefficients are defined for M>=2, so M = 1 isn't included in the output array.
Mmax (int): Maximal M value
p_max (int): Maximal iteration number
Output (2D ndarray): The covergence ratios; different M's
are represented by different columns, and different iteration numbers by
different rows.
   """
    all_gfuns = np.empty((p_max + 1, Mmax - 1))
    for Mi in range(0, Mmax - 1):
        all_gfuns[:, Mi] = conv_gfuns(Mi + 2, p_max, 1)
    return all_gfuns[1:(p_max + 1), :]/all_gfuns[0:p_max, :]


def conv_ratios_fm(MplusKmax, p_max): # Convergence ratios for the function of matrix computation error
    """
The function computes the convergence ratios of the function of matrix
extrapolation error. The ratios are computed for all 2=<M+K<=MplusKmax and for
all iteration numbers up to p_max.
The coefficients are defined for M>=2, so M+K=1 (M = 1, K = 0) isn't included in
the output array.
MplusKmax (int): Maximal M+K value
p_max (int): Maximal iteration number
Output (2D ndarray): The covergence ratios; different M's
are represented by different columns, and different iteration numbers by
different rows.
    """
    all_hfuns = np.empty((p_max + 1, MplusKmax - 1))
    for MplusKi in range(0, (MplusKmax - 1)):
        all_hfuns[:, MplusKi] = conv_hfuns(MplusKi + 2, p_max, 1)
    return all_hfuns[1:(p_max + 1), :]/all_hfuns[0:p_max, :]


def conv_ratios_1st(p_max): # Convergence ratios for the first time-step
    """
The function computes the convergence ratios of the first time-step.
The ratios are computed for all iteration numbers up to p_max.
Input:
p_max (int): The maximal iteration number
Output (1D ndarray): The covergence ratios for all iteration numbers
    """
    lfuns = conv_lfuns(p_max, 1)
    return lfuns[1:(p_max + 1)]/lfuns[0:p_max]