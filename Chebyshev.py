# -*- coding: utf-8 -*-
"""
Functions for a Chebyshev polynomial approximation.
Author: Ido Schaefer
"""
import numpy as np
from scipy.fftpack import dct


def chebc(f, leftb, rightb, N): # Chebyshev coefficient computation of a given function
    """
    The function returns the Chebychev coefficients of the function f in a given domain.
    f: A function object of the form: f(x).
    leftb, rightb: The boundaries of the approximation domain, [leftb, rightb].
    N: The number of Chebychev coefficients.
    Output: An ndarray containing the Chebyshev coefficients
    """
    # The Chebyshev points in the Chebyshev polynomial domain, [-1, 1]:
    xcheb = np.cos((np.arange(1, N + 1)*2 - 1)*np.pi/(2*N))
    # The Chebyshev points transformed to the approximation domain, [leftb, rightb]:
    x = 0.5*(xcheb*(rightb - leftb) + rightb + leftb)
    c = dct(f(x))/N
    c[0] = c[0]/2
    return c


def chebcM(fM): # Chebyshev coefficient computation from sampling points
    """
    The function computes the Chebychev coefficients of a set of functions
    from their samplings at the Chebychev points.
    fM: Two dimensional ndarray; contains the sampled values of several functions in
    its columns. For faster computation, fM should be stored in the memory
    in a column major fasion (Fortran like).
    Output: Two dimensional ndarray; the Chebyshev coefficiets of each function
    are the corresponding columns of the output ndarray.
    """
    
    # The number of Chebyshev sampling points:
    N = fM.shape[0]
    C = dct(fM, axis = 0)/N
    C[0, :] = C[0, :]/2
    return C

def chebcbv(fv): # Chebyshev coefficient computation of a function sampled at the boundary including Chebyshev points
    """
The function computes the Chebychev coefficients of a
function sampled at the Chebychev points that include the boundary of the
domain.

    Parameters
    ----------
    fv : 1D ndarray
        The sampled values.

    Returns
    -------
    1D ndarray
        The Chebyshev coefficients.

    """
    N = fv.size - 1
    c = dct(fv, type=1)/N
    c[0] = c[0]/2
    c[N] = c[N]/2
    return c


def chebc2result(Ccheb, xdomain, xresult): # Computation of the Chebyshev approximation from the coefficients
    """
    The function computes the Chebyshev polynomial approximation of a function
    from the corresponding Chebyshev coefficients, at a given set of points.
    Ccheb: The Chebyshev coefficients of the function (see the function chebc); ndarray
    xdomain: The approximation domain; ndarray of the form: np.array([xmin, xmax])
    xresult: An ndarray; the set of points in which the function is to be evaluated
    Output: An ndarray of the shape of xresult with the approximated function
    values at xresult
    """
    
    # Transforming xresult to the Chebyshev domain [-1 1]:
    xrcheb = (2*xresult - xdomain[0] - xdomain[1])/(xdomain[1] - xdomain[0])
    m = Ccheb.size
    # Tk represents the Chebyshev polynomial of the k'th degree.
    # T0 represents the Chebyshev polynomial of the (k-2)'th degree.
    # T1 represents the Chebyshev polynomial of the (k-1)'th degree.
    # Computing the Chebyshev polynomial terms iteraively by the Chebyshev
    # polynomial recurrence relation: 
    T0 = np.ones(xresult.shape)
    T1 = xrcheb.copy()
    result = Ccheb[0]*T0 + Ccheb[1]*T1
    for k in range(2, m):
          Tk = 2*xrcheb*T1 - T0
          result = result + Ccheb[k]*Tk
          T0 = T1
          T1 = Tk
    return result


def vchebMop(operator, u0, leftb, rightb, Ncheb, data_type=np.complex128):
    """
The function computes the vectors: v_k = T_k(operator)u0, where u0 is a vector,
T_k(x) is the k'th degree Chebyshev polynomial, and operator is a linear operator.
These vectors can be used for construction of a Chebyshev expansion of any
function of operator which operates on the vector u0, as follows:
f(operator)u0 \approx \sum_{k=0}^{Ncheb - 1} c_k*v_k,
where the c_k's are the Chebyshev coefficients (see chebc).
The c_k's are f dependent, and can be computed by the program chebc.
The program is useful when it is required to compute several functions
of the same operator, which operate on the same vector.
operator: A function object of the form: operator(v); the function returns 
the operation of the operator on the one-dimensional ndarray v.
u0: A one dimensional ndarray; mathematically defined above
Ncheb: The number of expansion terms/Chebyshev coefficients/output vectors
leftb, rightb: Scalars; define the boundaries of the eigenvalue domain of the
operator, [leftb, rightb].
Output: Two dimensional ndarray which contains the v_k vectors defined above.
Let M be a view of the output ndarray; its columns M[:, k] represent the corresponding v_k's.
"""
    # Defining the operator transformed to the domain of the Chebyshev
    # polynomial approximation, [-1, 1]
    chebop = lambda v: (2*operator(v) - (leftb + rightb)*v)/(rightb - leftb)
    dim = u0.size
    M = np.empty((dim, Ncheb), dtype=data_type, order='F')
    M[:, 0] = u0.copy()
    M[:, 1] = chebop(u0)
    for k in range(2, Ncheb):
        M[:, k] = 2*chebop(M[:, k-1]) - M[:, k-2]
    return M


def cheb_pols(x, N):
    """
The function computes the Chebyshev polynomials up to order N, evaluated
at values x. The computation is performed by the recursive definition of
the Chebyshev polynomials.
"""
    if isinstance(x, np.ndarray):
        Nx = x.size
        allT = np.zeros((N + 1, Nx))
        allT[0, :] = np.ones(Nx)
        allT[1, :] = x
        for Ti in range(2, N + 1):
            allT[Ti, :] = 2*x*allT[Ti - 1, :] - allT[Ti - 2, :]
    else:
        # If x is a regular numeric variable:
        allT = np.zeros(N + 1)
        allT[0] = 1
        allT[1] = x
        for Ti in range(2, N + 1):
            allT[Ti] = 2*x*allT[Ti - 1] - allT[Ti - 2]
    return allT


def cheb_rec_fun(prevCs, order_prev): # Calculation of the power coefficients of a Chebyshev polynomial
    """
The function computes the power coefficients of a particular Chebyshev
polynomial from the power coefficients of the two previous orders.
Intended for the use of the function Crecurrence.recurrence_coefs.
Input:
prevCs (2D ndarray): Containing the power coefficients of the two previous
orders; each order is represented by a row, and each power is represented
by a column. Column j represents the (j-1)'th power coefficients.
order_prev (int): The order of prevCs; not in use, but necessary for the function
recurrence_coefs.
Output (1D ndarray): The power coefficients of the new Chebyshev polynomial.
"""
    N = prevCs.shape[1]
    C = np.zeros(N)
    C[1:N] = 2*prevCs[1, 0:(N - 1)]
    C[0:(N - 2)] = C[0:(N - 2)] - prevCs[0, 0:(N - 2)]
    return C


def chebweights(N, lengthD): # Chebyshev integration weights for boundary including Chebyshev points
    """
The program computes the weights of the function values sampled on
Chebyshev points that include the boundary, used for integration.
For details, see Master's thesis (arXiv:1202.6520), Appendix B.5.
Exact for a polynomial of degree N-1.

    Parameters
    ----------
    N : int
        The number of Chebyshev points.
    lengthD : int/float
        The length of the domain.

    Returns
    -------
    1D ndarray
        The integration weights

    """
    integT = np.zeros(N)
    n_even = np.r_[0:N:2]
    integT[n_even] = -1/(n_even**2 - 1)
    return lengthD*chebcbv(integT)
