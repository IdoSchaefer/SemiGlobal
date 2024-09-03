# -*- coding: utf-8 -*-
"""
Semi-global propagator functions

see: Ido Schaefer, Hillel Tal-Ezer and Ronnie Kosloff,
"Semi-global approach for propagation of the time-dependent Schr\"odinger 
equation for time-dependent and nonlinear problems", JCP 2017


Author: Ido Schaefer

ido.schaefer@gmail.com
"""

import numpy as np
from scipy.linalg import norm, eig
from Chebyshev import chebcM, chebc2result, vchebMop
from Arnoldi import createKrop, getRvKr
from NewtonIpln import divdif, dvd2fun, get_capacity


def r2Taylor4(x, Dsize): # Computation of conversion coefficients from Newton basis polynomials to Taylor polynomials
    """
The function computes the conversion coefficients from Newton basis polynomials
to Taylor polynomials. The Taylor coefficients contain the 1/n! factor.
The Newton approximation is performed in a length 4 domain (capacity 1).
(see: "Semi-global approach for propagation of the time-dependent Schr\"odinger 
equation for time-dependent and nonlinear problems", Appendix C.1.
Relevant equations: (226)-(228))
Input:
x: An ndarray; contains the sampling points.
Dsize: size of the x domain
Output: 2D ndarray which contains the conversion coefficients; the row index 
indexes the Newton basis polynomials, and the column index indexes the Taylor
polynomials. The dimension of the output array is (NC, NC), where NC is the 
number of sampling points. 
"""

    # Conversion factor from the original x domain to a length 4 domain:
    Dfactor = 4/Dsize
    NC = x.size
    output_type = (x[0] + 0.).dtype.type
    Q = np.zeros((NC, NC), dtype = output_type)
    # First row:
    Q[0, 0] = 1
    # All the rest of the rows:
    for ri in range(1, NC):
        Q[ri, 0] = -Dfactor*x[ri - 1]*Q[ri - 1, 0]
        for Taylori in range(1, ri):
            Q[ri, Taylori] = (Q[ri - 1, Taylori - 1] - x[ri - 1]*Q[ri - 1, Taylori])*Dfactor
        Q[ri, ri] = Q[ri - 1, ri - 1]*Dfactor
    return Q


def Cpoly2Ctaylor(Cpoly, Cp2t):
    """
The function computes the Taylor-like coefficients from the coefficients
of a polynomial set, using the conversion coefficients Cp2t.
Input:
Cpoly: 2D ndarray; contains the vector coefficients of the polynomial set
in separate columns.
Cp2t: 2D ndarray; contains the conversion coefficients from the
polynomial set to the Taylor polynomials (see, for example, r2Taylor4).
Output:
Ctaylor: 2D ndarray; contains the Taylor-like coefficients, where
different orders are represented by different columns.
"""

    Nu, NC = Cpoly.shape
    output_type = (Cpoly[0, 0] + Cp2t[0, 0]).dtype.type
    Ctaylor = np.zeros((Nu, NC), dtype=output_type)
    Ctaylor[:, 0] = Cpoly[:, 0]
    for polyi in range(1, NC):
        Ctaylor[:, 0:(polyi + 1)] += Cpoly[:, polyi][:, np.newaxis]*Cp2t[polyi, 0:(polyi + 1)]
    return Ctaylor



def guess0(ui, Np):
    """
The function returns the zeroth order approximation for the guess of the first
time-step.
"""
    return np.tile(ui[:, np.newaxis], (1, Np))


def maketM(t, Nt_ts): # Computation of transpose Vandermonde matrix of time points
    """
Computation of the matrix of time Taylor polynomials. It is equivalent to the 
transpose Vandermonde matrix for the time points specified by t. The output
2D ndarray represents a matrix with the following jeneral term:
timeM[i, j] = t[j]**i
t: 1D ndarray containing the time-points
Nt_ts: The number of rows, which is the number of time sampling points in the 
time step; the maximal degree of t is represented by the last row of the output
matrix and is equivalent to Nt_ts - 1.
"""
    Nt = t.size
    timeM = np.empty((Nt_ts, Nt))
    timeM[0, :] = 1
    for vi in range(1, Nt_ts):
        timeM[vi, :] = t*timeM[vi - 1, :]
    return timeM


def f_fun(z, t, Nt_ts, tol, factorialNt_ts): # Computation of the function \tilde{f}_{Nt_ts}(z, t)
    """
The function computes the \tilde{f}_{Nt_ts}(z, t) function (see: "Semi-global 
approach for propagation of the time-dependent Schr\"odinger equation for
time-dependent and nonlinear problems", Eq. (82), where Nt_ts stands for m in
the paper).
z: 1D ndarray with the required points in the eigenvalue domain of \tilde{G}
t: 1D ndarray with the required time-points
Nt_ts: The number of interior time-points in the time-step; specifies the 
computed function \tilde{f}_{Nt_ts}(z, t).
tol: The required tolerance of the computation
factorialNt_ts: The value of Nt_ts!; required for the error estimation.
Output: 2D ndarray with the function values; the row index indexes different
z values, and the column index indexes different t values.
    """
    
    Nt = t.size
    Nz = z.size
    zt = z[:, np.newaxis]@t[np.newaxis, :]
    output_type = (zt[0, 0] + 0.).dtype.type
    # Condition for estimating if \tilde{f}_{Nt_ts}(z, t) should be computed 
    # directly or by a "tail" of a Taylor expansion (see supplementary material):
    is_big = factorialNt_ts*np.spacing(1)/np.abs(zt)**Nt_ts < tol
    result = np.ones((Nz, Nt), dtype=output_type, order='F')
    # First, we compute \tilde{f}_{Nt_ts}(z, t)/(t^Nt_ts), which is a function of zt.
    # A direct computation for large arguments:
    result[is_big] = np.exp(zt[is_big])
    for polyi in range(1, Nt_ts + 1):
        result[is_big] = polyi*(result[is_big] - 1)/zt[is_big]
    # Computation by a Taylor form for small arguments:
    is_not_converged = np.logical_not(is_big)
    term = is_not_converged.astype(output_type)
    polydeg = 1
    while is_not_converged.max():
        term[is_not_converged] = zt[is_not_converged]*term[is_not_converged]/(polydeg + Nt_ts)
        result[is_not_converged] += term[is_not_converged]
        polydeg += 1
        is_not_converged[is_not_converged] = \
            np.abs(term[is_not_converged])/np.abs(result[is_not_converged]) > np.spacing(1)
    # Obtaining the required function \tilde{f}_{Nt_ts}(z, t):
    result *= t**Nt_ts
    return result


def f_chebC(t, Nz, Nt_ts, leftb, rightb, tol, factorialNt_ts):
    """
The function computes the Chebyshev coefficients of \tilde{f}_{Nt_ts}(z, t),
where z is the argument of the Chebyshev expansion, and t serves as a parameter.
Input:
t: 1D ndarray of time-values
Nz: The number of Chebyshev sampling points for the z argument
Nt_ts: Defines the function as above
leftb: The minimum boundary of the approximation domain
rightb: The maximum boundary of the approximation domain
tol: The tolerance parameter for the computation of \tilde{f}_{Nt_ts}(z, t)
Output: 2D ndarray containing the Chebyshev coefficients, where different
t values are represented by separate columns.
"""
    
    # The Chebyshev sampling points in the Chebyshev domain, [-1 1]:
    zsamp_cheb = np.cos((np.r_[1:(Nz + 1)]*2 - 1)*np.pi/(2*Nz))
    # Transformation to the approximation domain:
    zsamp = 0.5*(zsamp_cheb*(rightb - leftb) + rightb + leftb)
    f_zt = f_fun(zsamp, t, Nt_ts, tol, factorialNt_ts)
    Ccheb_f = chebcM(f_zt)
    return Ccheb_f


def Ufrom_vCheb(v_tpols, timeM, Vcheb, Ccheb_f, f_error=None):
    """
The function computes the solution for the Chebyshev algorithm at all
time points specified by the transpose Vandermonde matrix timeM.
Input:
v_tpols: 2D ndarray; The v_j vectors excluding the last one, j=0,1,...,Nt_ts-1, in
seperate columns; the vector coefficients of the Taylor time-polynomials
in the solution equation
timeM: 2D ndarray; represents the matrix of the t powers for the required time-points
Vcheb: 2D ndarray; the T_k(\tilde{G})v_{Nt_ts} vectors, k=0,1,...,Nt_ts-1, in
separate columns
Ccheb_f: 2D ndarray; the Chebyshev coefficients of \tilde{f}_{Nt_ts}(z, t) in the
required time-points specified by timeM, as computed by the function f_chebC
f_error: The estimated relative error of the computation of \tilde{f}_{Nt_ts}(G,t[Nt_ts - 1])v_vecs[:, Nkr];
required for the computation of the error of the solution induced by the 
function of matrix computation error. When this computation is unrequired, no
value is assigned to f_error.
Output:
U: 2D ndarray; the computed solution at the required time-points
fUerror: Estimation of the error resulting from the Chebyshev approximation
for the computation of the function of matrix; returned as the second term in
a tuple only when f_error is specified.
"""

    fGv = Vcheb@Ccheb_f
    U = v_tpols@timeM + fGv
    if not f_error is None:
        fUerror = f_error*norm(fGv[:, -2], check_finite=False)/norm(U[:, -2], check_finite=False)
        return U, fUerror
    # Otherwise, this line is executed:
    return U



def Ufrom_vArnoldi(v_tpols, timeM, Upsilon_without_bar, RvKr, samplingp, capacity,
                   Nt_ts, tol, factorialNt_ts, estimate_error=False):
    """
The function computes the solution for the Arnoldi algorithm at all
time points specified by the transpose Vandermonde matrix timeM.
Input:
v_tpols: 2D ndarray; The columns are the the v_j vectors excluding the last one,
j=0,1,...,Nt_ts-1; the vector coefficients of the Taylor time-polynomials in 
the solution equation
timeM: 2D ndarray; represents the matrix of the t powers for the required 
time-points (see the function maketM).
Upsilon_without_bar: 2D ndarray, containing the orthonormalized Krylov space
vectors which participate in the approximation
RvKr: 2D ndarray of the vectors computed by Arnoldi.getRvKr
samplingp: The sampling points for the Newton expansion in the Krylov space
capacity: The capacity of the approximation domain
Nt_ts: Number of internal Chebyshev time-points
tol: The tolerance parameter for computation of \tilde{f}_{Nt_ts}(z, t)
factorialNt_ts: The value of Nt_ts!
estimate_error: A boolean; True means that the error of the computation of
\tilde{f}_{Nt_ts}(G,t[Nt_ts - 1])v_vecs[:, Nkr] and the resulting error of U is
required and is to be returned. Otherwise, estimate_error=False. 
Output:
U: 2D ndarray; the solution in the required time points, where the different
time-points are represented by seperate columns
f_error: The estimated relative error of the computation of \tilde{f}_{Nt_ts}(G,t[Nt_ts - 1])v_vecs[:, Nkr];
returned as the second term in a tuple only when estimate_error=True
fUerror: The estimated relative error of U resulting from the computational
error of \tilde{f}_{Nt_ts}(\tilde{G},t[Nt_ts - 1])v_vecs[:, Nkr];
returned as the third term in a tuple only when estimate_error=True
    """
    
    # The size of the Krylov space which is used for the approximation:
    Nfm = samplingp.size - 1        
    # The required time-points are given by their first power:
    tp = timeM[1, :]
    # Computing the divided differences for the Newton expansion of 
    # \tilde{f}_{Nt_ts}(G, t), for all time-points:
    fdvd_ts, _ = divdif(samplingp/capacity, f_fun(samplingp, tp, Nt_ts, tol, factorialNt_ts).T)
    fdvd_ts = fdvd_ts.T
    # The computation of the Newton expansion of \tilde{f}_{Nt_ts}(G, t)v_vecs[:, Nkr]
    # in the Krylov space, for all tp:
    fGv_kr = RvKr[0:Nfm, :]@fdvd_ts
    # Computing the solution in all tp: 
    U = v_tpols@timeM + Upsilon_without_bar@fGv_kr
    if estimate_error:
        # The absolute error:
        f_error_abs = abs(fdvd_ts[Nfm, Nt_ts - 2])*norm(RvKr[:, Nfm])
        # The relative error:
        f_error = f_error_abs/norm(fGv_kr[:, Nt_ts - 2], check_finite=False)
        # The relative error of U, resulting from this computation:
        fUerror = f_error_abs/norm(U[:, Nt_ts - 2], check_finite=False)
        return U, f_error, fUerror
    # Otherwise, this line is executed:
    return U


def SemiGlobal(Gop, Gdiff_op, Gdiff_matvecs,
    ui, tgrid, Nts, Nt_ts, Nfm, tol, ihfun=None, ev_domain=None, Niter=10, Niter1st=16,
    test_tpoint=None, data_type=np.complex128, display_mode=True, save_memory=False, *args):
    """
The program solves time-dependent Schroedinger equation for a time-dependent,
nonlinear Hamiltonian, using the semi-global propagator by Hillel Tal-Ezer.
An inhomogeneos source term can be included.
Input:
Gop: A function object of the form Gop(u, t, v, *args), where:
    u: A 1D ndarray which represents the state vector
    t: The time variable (scalar)
    v: A 1D ndarray which represents any vector of the dimension of u
    args: Optional additional arguments
Gop returns G(u, t)v (in quantum mechanics, G(u, t) = -iH(u, t)/hbar). The 
optional arguments included in args will be written in the place of *args in
the current function, separated by commas.
Gdiff_op: A function object of the form Gdiff_op(u1, t1, u2, t2, *args), where:
    u1: 2D ndarray; represents the state vector in several time points, where
    different time-points are resresented by separate columns.
    t1: 1D ndarray; represents the time points in which the corresponding
    columns of u1 are evaluated.
    u2: 1D ndarray; represents the state vector in a particular time-point.
    t1: scalar; represents the time-point in which u2 is evaluated.
    args: As above; the optional arguments must be the same as for Gop.
Gdiff_op returns a 2D ndarray, where the j'th column represents
(G(u1[:, j], t1[j]) - G(u2, t2))u1[:, j],
where t1[j] is the j'th time point and u1[:, j] is the j'th column of u1.
Gdiff_matvecs: The number of matrix-vector multiplications required for
the computation of the operation of Gdiff_op for each time-point (usually less than 2).
(The number of matrix-vector multiplications is counted as the number of large 
scale operations with the highest scaling with the dimension of the problem.)
ui: 1D ndarray; represents the initial state vector.
tgrid: The time grid of the desired output solution (see U in the output).
Should be ordered in an increasing order for a forward propagation, and a 
decreasing order for a backward propagation.
Nts: The number of time steps of the propagation. 
Nt_ts: The number of interior Chebyshev time points in each time-step, used during
the computational process (M in the paper).
Nfm: The number of expansion terms for the computation of the function of 
matrix (K in the paper).
tol: The desired tolerance of the convergence (epsilon in the paper)
ihfun: A function object of the form: ihfun(t, *args), where:
    t: 1D ndarray; represents several time-points.
    args: As above; the optional arguments must be the same as for Gop.
ihfun returns a 2D ndarray which represents the inhomogeneous source term
(s(t) in the paper) in the time points specified by t. Different time-points 
are represented by separate columns. The number of rows of the output of ihfun
is identical to the dimension of the state vector.
The default None value means that there is no inhomogeneous term (s(t) \equiv 0).
ev_domain: ndarray/list/tuple of 2 terms; the (estimated) boundaries of the eigenvalue
domain of G(u, t); required when a Chebyshev algorithm is used for the computation of
the function of matrix.
The default None value means that the Arnoldi algorithm is employed instead.
The general form for an ndarray is np.r_[lambda_min, lambda_max], where
lambda_min is the (estimated) lowest eigenvalue of G(u, t), and lambda_max is
the (estimated) highest eigenvalue.
Since G(u, t) is time-dependent, its eigenvalue domain is also time-dependent.
Hence, ev_domain has to cover all the eigenvalue domains of G(u, t) throughout
the propagation process.
Niter: The maximal allowed number of iterations for all time-steps
excluding the first
Niter1st: The maximal allowed number of iterations for the first time-step
test_tpoint: Represents the test point for the time-expansion error computation;
defined as the difference between the test point and the beginning of the time
step. It is the same for all time steps. The default None value means that the 
default test point is computed by the program.
data_type: The data-type object of the output solution
display_mode: A boolean variable; True (default) means that warnings are displayed
during the propagation. False means that warnings are displayed only
before and after the propagation.
save_memoty: A boolean variable; False (default) means that the solution in all propagation
grid (history['U'] in the output) and test points (history['Utestp']) is stored
in the memory and is contained in the output dictionary of the propagation
details, history. True means the opposite.
args: Optional additional arguments for the input functions: Gop, Gdiff_op,
and ihfun; they should be written in the place of *args separated by commas.

Output:
U: 2D ndarray; contains the solution at the time points specified in tgrid; 
the different time-points are respresented by separate columns.
history: A dictionary with the details of the propagation process; contains the
following keys:
    mniter: The mean number of iteration for a time step, excluding the first step,
    which typically requires more iterations. Usually, mniter should be 1 for ideal
    efficiency of the algorithm.
    matvecs: The number of G(u, t) operations on a vector
    t: 1D ndarray; the time grid of propagation
    U: 2D ndarray; the solution at t, where different time-points are represented
    by separate columns
    Utestp: 2D ndarray; the solution at all test-points in all time-steps,
    where different time steps are represented by separate columns
    texp_error: 1D ndarray; the estimations for the error of U resulting from the 
    time-expansion, for all time-steps
    f_error: 1D ndarray; the estimations for the error of the computation of the
    function of matrix for all time-steps (for the Arnoldi approximation only)
    fUerror: 1D ndarray; the estimations for the error of U, resulting from the
    computation of the function of matrix, for all time-steps
    conv_error: 1D ndarray; the estimations for the convergence errors for all time-steps
    niter: 1D ndarray; the number of iterations for each time-steps 
    max_errors: A nested dictionary which contains the maximal estimated errors;
    contains the following keys:
        texp: The maximal estimated error of U, resulting from the time-expansions in
        each time-step
        fU: The maximal estimated error of U, resulting from the computation of
        the function of matrix
        f: The maximal estimated error of the function of matrix
        computation itself
        conv: The maximal estimated convergence error
"""

    # Creating the dictionary of the output details of the propagation process:
    history = {'max_errors': {}}
    # If the eigenvalue domain is not specified, the Arnoldi approach is employed.
    Arnoldi = ev_domain is None
    # In order to detect if the propagation is a forward or a backward propagation:
    direction = np.sign(tgrid[1] - tgrid[0])
    Nt = tgrid.size
    tinit = tgrid[0]
    tf = tgrid[Nt - 1]
    # The length of the time interval of the whole propagation (can be negative):
    T = tf - tinit
    # If Nts is a float, it has to be converted to an integer such that it can be
    # used for indexing:
    if not isinstance(Nts, int):
        Nts = int(np.round(Nts))
    # The length of the time step interval:
    Tts = T/Nts
    # The index of the middle term in the time step:
    tmidi = Nt_ts//2
    # This is actually unrequired, but I inserted this to make the Python code 
    # consistent with my MATLAB code:
    if Nt_ts%2 == 0:
        tmidi = tmidi - 1
    ui = ui.astype(data_type)
    Nu = ui.size
    U = np.zeros((Nu, Nt), dtype=data_type, order = 'F')
    U[:, 0] = ui
    # The Chebyshev points for expansion in time, in the domain in which the Chebyshev expansion
    # is defined: [-1 1]
    tcheb = -np.cos(np.r_[0:Nt_ts]*np.pi/(Nt_ts - 1))
    # The Chebyshev points for expansion in time, in the domain of the time
    # variable:
    t_ts = 0.5*(tcheb + 1)*Tts
    if test_tpoint is None:
        # Default test point for error estimation:
        test_tpoint = (t_ts[tmidi] + t_ts[tmidi + 1])/2
    # The interior time points of the current time step for interpolation,
    # and the next time step for extrapolation of the guess solution in the next step:
    t_2ts = np.r_[t_ts, test_tpoint, Tts + t_ts[1:Nt_ts], Tts + test_tpoint]    
    # The full propagation grid:
    propagation_grid = np.r_[np.kron(np.r_[tinit:(tf - 2*np.spacing(tf)):Tts], np.ones(Nt_ts)) + \
                             np.tile(np.r_[t_ts[0:(Nt_ts - 1)], test_tpoint], Nts), tf]
    # The -2*np.spacing(tf) is in order to avoid undesirable outcomes of roundoff errors.
    # Necessary for error estimation of \tilde{f}_{Nt_ts}(z, t):
    factorialNt_ts = np.math.factorial(Nt_ts)
    if not Arnoldi:
        # If the eigenvalue domain is specified, a Chebyshev approximation
        # for the function of matrix is employed.
        min_ev = ev_domain[0]
        max_ev = ev_domain[1]        
        # Computing the coefficients for the Chebyshev expansion of the
        # function of matrix, in all the interior time points.
        # CchebFts contains the coefficients of the current time step, and
        # CchebFnext contains the coefficients of the next one.
        Ccheb_f_comp = f_chebC(t_2ts[1:(2*Nt_ts + 1)], Nfm, Nt_ts, min_ev, max_ev, tol, factorialNt_ts)
        Ccheb_f_ts = Ccheb_f_comp[:, 0:Nt_ts]
        Ccheb_f_next = Ccheb_f_comp[:, Nt_ts:(2*Nt_ts)]   
        # Computing evenly spaced sampling points for the error test of the
        # function of matrix error:
        dz_test = (max_ev - min_ev)/(Nu + 1)
        ztest = min_ev + dz_test*np.r_[1:(Nu + 1)]
        fztest = np.squeeze(f_fun(ztest, np.array([Tts]), Nt_ts, tol, factorialNt_ts))
        f_error = np.max(np.abs(chebc2result(Ccheb_f_ts[:, Nt_ts - 2], ev_domain, ztest) - fztest)/np.abs(fztest))
        history['max_errors']['f'] = f_error
        if f_error>1e-5:
            print(f'Warning: The estimated error of the computation of the function of matrix ({f_error}) is larger than 1e-5.\nInstability in the propagation process may occur.')
    if not save_memory:
        history['U'] = np.zeros((Nu, Nts*(Nt_ts - 1) + 1), dtype=data_type, order = 'F')
        history['U'][:, 0] = ui
        history['Utestp'] = np.zeros((Nu, Nts), dtype=data_type, order = 'F')
    # The propagation grid without the test points:
    history['t'] = np.delete(propagation_grid, np.r_[(Nt_ts - 1):(Nts*Nt_ts):Nt_ts])
    history['texp_error'] = np.zeros(Nts)
    if Arnoldi:
        # If the Arnoldi approximation is employed, the error of the function
        # of matrix computation is different for each time-step:
        history['f_error'] = np.zeros(Nts)
    history['fUerror'] = np.zeros(Nts)
    history['conv_error'] = np.zeros(Nts)
    history['niter'] = np.zeros(Nts)
    # Computing the matrix of the time Taylor polynomials.
    # timeMts contains the points in the current time step, and timeMnext
    # contains the points in the next time step:
    timeMcomp = maketM(t_2ts[1:(2*Nt_ts + 1)], Nt_ts)
    timeMts = timeMcomp[:, 0:Nt_ts]
    timeMnext = timeMcomp[:, Nt_ts:(2*Nt_ts)]    
    # Computing the coefficients of the transformation from the Newton
    # interpolation polynomial terms, to a Taylor like form:
    Cr2t = r2Taylor4(t_ts, Tts)
    # The extended "inhomogeneous" vectors:
    s_ext = np.zeros((Nu, Nt_ts + 1), dtype=data_type, order = 'F')
    # The indices for the computation of s_ext (excluding tmidi):
    s_ext_i = np.r_[0:tmidi, (tmidi + 1):(Nt_ts + 1)]
    # The v vectors are defined recursively, and contain information about
    # the time dependence of the s_ext vectors:
    v_vecs = np.empty((Nu, Nt_ts + 1), dtype=data_type, order = 'F')
    # If there is no inhomogeneous term in the equation, ihfun == None, and there_is_ih == False.
    # If there is, there_is_ih == true.
    there_is_ih = (ihfun != None)
    if there_is_ih:
        s = np.empty((Nu, Nt_ts + 1), dtype=data_type, order = 'F')
        s[:, 0] = ihfun(tinit, *args).squeeze()
    # The 0'th order approximation is the first guess for the first time step.
    # Each column represents an interior time point in the time step:
    Uguess = guess0(ui, Nt_ts + 1)
    Unew = np.empty((Nu, Nt_ts + 1), dtype=data_type, order = 'F')
    allniter = 0
    # These variables are used to determine which points in tgrid are in the computed time-step.  
    tgrid_lowi = 1
    tgrid_upi = 0
    for tsi in range(0, Nts):
        # The time of the interior time points within the time-step:
        t = propagation_grid[tsi*Nt_ts + np.r_[np.r_[0:(Nt_ts - 1)], Nt_ts, Nt_ts - 1]]
        # The first guess for the iterative process, for the convergence of the u
        # values. Each column represents an interior time point in the time step:
        Ulast = Uguess.copy()
        Unew[:, 0] = Ulast[:, 0]
        v_vecs[:, 0] = Ulast[:, 0]
        if there_is_ih:
            # Computing the inhomogeneous term:
            s[:, 1:(Nt_ts + 1)] = ihfun(t[1:(Nt_ts + 1)], *args)
        # Starting an iterative process, until convergence:
        niter = 0
        reldif = tol + 1
        while (reldif>tol and ((tsi>0 and niter<Niter) or (tsi == 0 and niter<Niter1st))):
            # Calculation of the inhomogeneous s_ext vectors. Note that
            # s_ext[:, tmidi] is equivalent to s[:, tmidi], and therefore not
            # calculated.
            s_ext[:, s_ext_i] = Gdiff_op(Ulast[:, s_ext_i], t[s_ext_i], Ulast[:, tmidi], t[tmidi], *args)
            # If there is an inhomogeneous term, we add it to the s_ext 
            # vectors:
            if there_is_ih:
                s_ext[:, tmidi] = s[:, tmidi]
                s_ext[:, s_ext_i] = s_ext[:, s_ext_i] + s[:, s_ext_i]
            # Calculation of the coefficients of the form of Taylor
            # expansion, from the coefficients of the Newton
            # interpolation at the points t_ts.
            # The divided differences are computed by the function divdif.
            # For numerical stability, we have to transform the time points
            # in the time step, to points in an interval of length 4:
            Cnewton, _ = divdif(t_ts*4/Tts, s_ext[:, 0:Nt_ts])
            # Calculating the Taylor like coefficients:
            Ctaylor = Cpoly2Ctaylor(Cnewton, Cr2t)
            # Calculation of the v vectors:
            for polyi in range(1, Nt_ts + 1):
                v_vecs[:, polyi] = (Gop(Ulast[:, tmidi], t[tmidi], v_vecs[:, polyi-1], *args)
                                    + Ctaylor[:, polyi - 1])/polyi
            if not np.min(np.isfinite(v_vecs[:, Nt_ts])):
                # It means that the algorithm diverges.
                # In such a case, change Nts, Nt_ts and/or Nfm.
                print(f'Error: The algorithm diverges (in time step No. {tsi + 1}).')
                history['mniter'] = allniter/tsi
                return U, history
            if Arnoldi:
                # Creating the Krylov space by the Arnodi iteration procedure,
                # in order to approximate \tilde{f}_{Nt_ts}(G, t)v_vecs[:, Nt_ts]:
                Upsilon, Hessenberg = createKrop(lambda v: Gop(Ulast[:, tmidi], t[tmidi], v, *args),
                                                 v_vecs[:, Nt_ts], Nfm, data_type)
                # Obtaining eigenvalues of the Hessenberg matrix:
                eigval, _ = eig(Hessenberg[0:Nfm, 0:Nfm])
                # The test point is the average point of the eigenvalues:
                avgp = np.sum(eigval)/Nfm
                samplingp = np.r_[eigval, avgp]
                capacity = get_capacity(eigval, avgp)
                # Obtaining the expansion vectors for a Newton approximation of
                # \tilde{f}_{Nt_ts}(G, t)v_vecs[:, Nt_ts] in the reduced Krylov space:
                RvKr = getRvKr(Hessenberg, v_vecs[:, Nt_ts], samplingp, Nfm, capacity)
                # Calculation of the solution in all time points
                # within the time step:
                Unew[:, 1:(Nt_ts + 1)], f_error, fUerror = \
                    Ufrom_vArnoldi(v_vecs[:, 0:Nt_ts], timeMts, Upsilon[:, 0:Nfm],
                                   RvKr, samplingp, capacity, Nt_ts, tol, factorialNt_ts, estimate_error=True)
            else:
                # Employing a Chebyshev approximation for the function of
                # matrix computation.
                # Vcheb is a 2D ndarray. It contains the following vectors in 
                # separate columns:
                # T_n(G(Ulast[:, tmidi], t[tmidi]))*v_vecs[: ,Nt_ts],  n = 0, 1, ..., Nfm-1
                # where the T_n(z) are the Chebyshev polynomials.
                # The n'th vector is the column of Vcheb with index n.
                Vcheb = vchebMop(lambda v: Gop(Ulast[:, tmidi], t[tmidi], v, *args),
                                 v_vecs[:, Nt_ts], min_ev, max_ev, Nfm, data_type)
                # Calculation of the solution in all the time points
                # within the time step:
                Unew[:, 1:(Nt_ts + 1)], fUerror = Ufrom_vCheb(v_vecs[:, 0:Nt_ts], timeMts, Vcheb, Ccheb_f_ts, f_error)
            if not np.min(np.isfinite(Unew[:, Nt_ts - 1])):
                # It means that the algorithm diverges.
                # In such a case, change Nts, Nt_ts and/or Nfm.
                print(f'Error: The algorithm diverges (in time step No. {tsi + 1}).')
                history['mniter'] = allniter/tsi
                return U, history
            # Error estimation for the time expansion;
            # A Newton interpolation of s_ext at the test time-point:
            s_ext_testpoint = dvd2fun((4/Tts)*t_ts, Cnewton, (4/Tts)*test_tpoint)
            # The error estimation of the time-expansion:
            texpansion_error = norm(s_ext_testpoint - s_ext[:, Nt_ts])*np.abs(Tts)/norm(Unew[:, Nt_ts])
            # Convergence check of u:
            reldif = norm(Unew[:, Nt_ts - 1] - Ulast[:, Nt_ts - 1])/norm(Ulast[:, Nt_ts - 1])
            Ulast = Unew.copy()
            niter += 1
        if display_mode and reldif>tol:
            print(f'Warning: The program has failed to achieve the desired tolerance in the iterative process (in time step No. {tsi + 1}).')
            # In such a case, change Nts, Nt_ts and/or Nfm.
        if display_mode and texpansion_error>tol:
            print(f'Warning: The estimated error of the time expansion ({texpansion_error}) is larger than the requested tolerance.\nThe solution might be inaccurate (in time step No. {tsi + 1}).')
        if display_mode and fUerror>tol: 
            print(f'Warning: The estimation of the error resulting from the function of matrix ({fUerror}) is larger than the requested tolerance.\nThe solution might be inaccurate (in time step No. {tsi + 1}).')
        if display_mode and Arnoldi and f_error>1e-5:
            print(f'Warning: The estimated error of the function of matrix ({f_error}) is larger than 1e-5.\nInstability in the propagation process may occur (in time step No. {tsi + 1}).')
        history['texp_error'][tsi] = texpansion_error
        if Arnoldi:
            history['f_error'][tsi] = f_error
        history['fUerror'][tsi] = fUerror
        history['conv_error'][tsi] = reldif
        history['niter'][tsi] = niter
        if tsi != 0:
            allniter = allniter + niter
        else:
            if Arnoldi:
                history['matvecs'] = niter*(Nt_ts*(1 + Gdiff_matvecs) + Nfm)
            else:
                history['matvecs'] = niter*(Nt_ts*(1 + Gdiff_matvecs) + Nfm - 1)
        # Computation of the solution at the tgrid points.
        # Finding the indices of the tgrid points within the time step (the indices of the points
        # to be computed are between tgrid_lowi and tgrid_upi):
        while tgrid_upi<(Nt - 1) and (t[Nt_ts - 1] - tgrid[tgrid_upi + 1])*direction>np.spacing(np.abs(t[Nt_ts - 1]))*10:
            tgrid_upi = tgrid_upi + 1
        # Calculating the solution at the tgrid points: 
        if tgrid_lowi<=tgrid_upi:
            timeMout = maketM(tgrid[tgrid_lowi:(tgrid_upi + 1)] - t[0], Nt_ts)
            if Arnoldi:
                U[:, tgrid_lowi:(tgrid_upi + 1)] = \
                    Ufrom_vArnoldi(v_vecs[:, 0:Nt_ts], timeMout, Upsilon[:, 0:Nfm],
                                   RvKr, samplingp, capacity, Nt_ts, tol, factorialNt_ts)
            else:
                Ccheb_f_out = f_chebC(tgrid[tgrid_lowi:(tgrid_upi + 1)] - t[0],
                                      Nfm, Nt_ts, min_ev, max_ev, tol, factorialNt_ts)
                U[:, tgrid_lowi:(tgrid_upi + 1)] = Ufrom_vCheb(v_vecs[:, 0:Nt_ts], timeMout, Vcheb, Ccheb_f_out)
            tgrid_lowi = tgrid_upi + 1
        # If one of the points in tgrid coincides with the point of the
        # propagation grid:
        if np.abs(t[Nt_ts - 1] - tgrid[tgrid_upi + 1])<=np.spacing(np.abs(t[Nt_ts - 1]))*10:
            tgrid_upi = tgrid_upi + 1
            U[:, tgrid_upi] = Unew[:, Nt_ts - 1]
            tgrid_lowi = tgrid_upi + 1
        if not save_memory:
            history['U'][:, (tsi*(Nt_ts - 1) + 1):((tsi + 1)*(Nt_ts - 1) + 1)] = Unew[:, 1:Nt_ts]
            history['Utestp'][:, tsi] = Unew[:, Nt_ts]
        # The new guess is an extrapolation of the solution within the
        # previous time step:
        Uguess[:, 0] = Unew[:, Nt_ts - 1]
        if Arnoldi:
            Uguess[:, 1:(Nt_ts + 1)] = \
                Ufrom_vArnoldi(v_vecs[:, 0:Nt_ts], timeMnext, Upsilon[:, 0:Nfm],
                               RvKr, samplingp, capacity, Nt_ts, tol, factorialNt_ts)
        else:
            Uguess[:, 1:(Nt_ts + 1)] = Ufrom_vCheb(v_vecs[:, 0:Nt_ts], timeMnext, Vcheb, Ccheb_f_next)
        if there_is_ih:
            s[:, 0] = s[:, Nt_ts - 1]
    history['max_errors']['texp'] = history['texp_error'].max()
    if Arnoldi:
        history['max_errors']['f'] = history['f_error'].max()
    history['max_errors']['fU'] = history['fUerror'].max()    
    history['max_errors']['conv'] = history['conv_error'].max()
    if history['max_errors']['texp']>tol:
        print(f"Warning: The maximal estimated error of the time expansion ({history['max_errors']['texp']}) is larger than the requested tolerance.\nThe solution might be inaccurate.")    
    if history['max_errors']['fU']>tol:
        print(f"Warning: The maximal estimated error resulting from the function of matrix computation ({history['max_errors']['fU']}) is larger than the requested tolerance.\nThe solution might be inaccurate.")
    if Arnoldi and history['max_errors']['f']>1e-5:
        print(f"Warning: The maximal estimated error of the function of matrix ({history['max_errors']['f']}) is larger than 1e-5.\nInstability in the propagation process is possible.")
    if history['max_errors']['conv']>tol:
        print(f"Warning: The maximal estimated error resulting from the iterative process ({history['max_errors']['conv']}) is larger than the requested tolerance.\nThe solution might be inaccurate.")
    history['mniter'] = allniter/(Nts - 1)
    if Arnoldi:
        history['matvecs'] += allniter*(Nt_ts*(1 + Gdiff_matvecs) + Nfm)
    else:
        history['matvecs'] += allniter*(Nt_ts*(1 + Gdiff_matvecs) + Nfm - 1)
    return U, history