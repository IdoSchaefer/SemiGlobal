# -*- coding: utf-8 -*-
"""
Semi-global propagator functions for the June 2024 version.

See: Ido Schaefer, Hillel Tal-Ezer and Ronnie Kosloff,
"Semi-global approach for propagation of the time-dependent Schr\"odinger 
equation for time-dependent and nonlinear problems", JCP 2017


Author: Ido Schaefer

ido.schaefer@gmail.com
"""

import numpy as np
import math
from scipy.linalg import norm, eig
from Chebyshev import chebc2result, vchebMop, chebweights
from Arnoldi import createKrop, getRvKr
from NewtonIpln import divdif, new_divdif, get_capacity
from SGfuns import r2Taylor4, Cpoly2Ctaylor, guess0, maketM, f_fun, f_chebC
from conv_coefs import conv_ratios_texp, conv_ratios_fm, conv_ratios_1st

def SGdata(Nt_ts_max, Nfm_max, Niter, Niter1st): # Data computation for the function SemiGlobal1
    """
    Input:
    Nt_ts_max (int): Maximal number of internal Chebyshev time-point
    Nfm_max (int): Maximal number of expansion terms for the computation of the function of 
        matrix.
    Niter (int): Maximal iteration number
    Niter1st (int): Maximal iteration number for the first time-step

    Returns:
    (dictionary): Contains the following keys:
        correct_texp: The correction coefficients for the time-expansion error
        correct_fm: The correction coefficients for the function of matrix error
        correct_1st: The correction coefficients for the first time-step
    """
    return {'correct_texp': 2*conv_ratios_texp(Nt_ts_max, Niter),
            'correct_fm': 2*conv_ratios_fm(Nt_ts_max + Nfm_max, Niter),
            'correct_1st': 2*conv_ratios_1st(Niter1st)
            }


def Ufrom_vCheb1(v_tpols, timeM, Vcheb, Ccheb_f, f_scalar_error=None):
    """
The function computes the solution for the Chebyshev algorithm at all
time points specified by the transpose Vandermonde matrix timeM.
Input:
v_tpols (2D ndarray): The v_j vectors in seperate columns; the vector 
coefficients of the Taylor time-polynomials in the solution equation
timeM (2D ndarray): Represents the matrix of the t powers for the required time-points
Vcheb: 2D ndarray; the T_k(\tilde{G})v_{Nt_ts} vectors, k=0,1,...,Nt_ts-1, in
separate columns
Ccheb_f (2D ndarray): The Chebyshev coefficients of \tilde{f}_{Nt_ts}(z, t) in the
required time-points specified by timeM, as computed by the function f_chebC
f_scalar_error (float): The estimated maximal error of the computation of \tilde{f}_{Nt_ts}(z,t[Nt_ts - 1]);
required for the computation of the error of the solution induced by the 
function of matrix computation error. When this computation is unrequired, no
value is assigned to f_scalar_error.
Output:
U (2D ndarray): the computed solution at the required time-points
fUerror (float): Estimation of the error resulting from the Chebyshev approximation
for the computation of the function of matrix; returned as the second term in
a tuple only when f_error is specified.
"""

    # Number of internal Chebyshev time-points:
    Nt_ts = v_tpols.shape[1] - 1
    fGv = Vcheb@Ccheb_f
    U = v_tpols[:, 0:Nt_ts]@timeM + fGv
    if not f_scalar_error is None:
        f_abs_error = f_scalar_error*norm(v_tpols[:, Nt_ts], check_finite=False)
        fUerror = f_abs_error/norm(U[:, Nt_ts - 2], check_finite=False)
        return U, fUerror
    # Otherwise, this line is executed:
    return U



def Ufrom_vArnoldi1(v_tpols, timeM, Upsilon_without_bar, RvKr, samplingp, capacity,
                    tol, factorialNt_ts, estimate_error=False):
    """
The function computes the solution for the Arnoldi algorithm at all
time points specified by the transpose Vandermonde matrix timeM.
Input:
v_tpols (2D ndarray): The columns are the the v_j vectors; the vector 
coefficients of the Taylor time-polynomials in the solution equation
timeM (2D ndarray): represents the matrix of the t powers for the required 
time-points (see the function maketM).
Upsilon_without_bar (2D ndarray): contains the orthonormalized Krylov space
vectors which participate in the approximation
RvKr (2D ndarray): The vectors computed by Arnoldi.getRvKr
samplingp (1D ndarray): The sampling points for the Newton expansion in the Krylov space
capacity (float): The capacity of the approximation domain
tol (float): The tolerance parameter for computation of \tilde{f}_{Nt_ts}(z, t)
factorialNt_ts (int): The value of Nt_ts!
estimate_error (boolean): True means that the error of the computation of
\tilde{f}_{Nt_ts}(G,t[Nt_ts - 1])v_vecs[:, Nkr] and the resulting error of U is
required and is to be returned. Otherwise, estimate_error=False (default). 
Output:
U (2D ndarray): the solution in the required time points, where the different
time-points are represented by seperate columns
f_abs_error: The estimated absolute error of the computation of \tilde{f}_{Nt_ts}(G,t[Nt_ts - 1])v_vecs[:, Nkr];
returned as the second term in a tuple only when estimate_error=True
fUerror (float): The estimated relative error of U resulting from the computational
error of \tilde{f}_{Nt_ts}(\tilde{G},t[Nt_ts - 1])v_vecs[:, Nkr];
returned as the third term in a tuple only when estimate_error=True
    """
    
    # The size of the Krylov space which is used for the approximation:
    Nfm = samplingp.size - 1
    # Number of internal Chebyshev time-points:
    Nt_ts = v_tpols.shape[1] - 1  
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
    U = v_tpols[:, 0:Nt_ts]@timeM + Upsilon_without_bar@fGv_kr
    if estimate_error:
        # The absolute error:
        f_abs_error = np.abs(fdvd_ts[Nfm, Nt_ts - 2])*norm(RvKr[:, Nfm])
        # The relative error of U, resulting from this computation:
        fUerror = f_abs_error/norm(U[:, Nt_ts - 2], check_finite=False)
        return U, f_abs_error, fUerror
    # Otherwise, this line is executed:
    return U


def SemiGlobal1(Gop, Gdiff_op, Gdiff_matvecs, ui, tgrid, Nts, Nt_ts, Nfm, tol,
                ihfun=None, ev_domain=None, Niter=10, Niter1st=16, data=None,
                data_type=np.complex128, display_mode=False, save_memory=False,
                tol_f=None, tol1st=None, comp_conv_cheb=None, comp_texp_odd=True,
                *args):
    """
The program solves time-dependent Schroedinger equation for a time-dependent,
nonlinear Hamiltonian, using the semi-global propagator by Hillel Tal-Ezer.
An inhomogeneos source term can be included.
The current version produces accurate error estimations. The convergence error
estimation is used as the criterion for determination of the number of iterations
in each time-step. There is also a reliable criterion for an instability alert.
Input:
Gop (function): A function object of the form Gop(u, t, v, *args), where:
    u (1D ndarray): Represents the state vector
    t (float): The time variable (scalar)
    v (1D ndarray): Represents any vector of the dimension of u
    args: Optional additional arguments
Gop returns G(u, t)v (in quantum mechanics, G(u, t) = -iH(u, t)/hbar). The 
optional arguments included in args will be written in the place of *args in
the current function, separated by commas.
Gdiff_op (function): A function object of the form Gdiff_op(u1, t1, u2, t2, *args), where:
    u1(2D ndarray): Represents the state vector in several time points, where
    different time-points are resresented by separate columns.
    t1 (1D ndarray): Represents the time points in which the corresponding
    columns of u1 are evaluated.
    u2 (1D ndarray): Represents the state vector in a particular time-point.
    t1 (float): Scalar; represents the time-point in which u2 is evaluated.
    args: As above; the optional arguments must be the same as for Gop.
Gdiff_op returns a 2D ndarray, where the j'th column represents
(G(u1[:, j], t1[j]) - G(u2, t2))u1[:, j],
where t1[j] is the j'th time point and u1[:, j] is the j'th column of u1.
Gdiff_matvecs (int): The number of matrix-vector multiplications required for
the computation of the operation of Gdiff_op for each time-point (usually less than 2).
(The number of matrix-vector multiplications is counted as the number of large 
scale operations with the highest scaling with the dimension of the problem.)
ui (1D ndarray): Represents the initial state vector.
tgrid (1D ndarray): The time grid of the desired output solution (see U in the output).
Should be ordered in an increasing order for a forward propagation, and a 
decreasing order for a backward propagation. Has to contain at least two time-points: The
initial and final point.
Nts (int): The number of time steps of the propagation. 
Nt_ts (int): The number of interior Chebyshev time points in each time-step, used during
the computational process (M in the paper).
Nfm (int): The number of expansion terms for the computation of the function of 
matrix (K in the paper).
tol (float/None): The desired tolerance of the entire propagation process 
(note the difference from the function SemiGlobal in the module SGfuns.py).
For a manual mode with a predefined iteration number, insert None instead. In 
this case, the iteration number of the first time-step is given by Niter1st,
and the iteration number for all other time-steps is given by Niter.
ihfun (function): A function object of the form: ihfun(t, *args), where:
    t (1D ndarray): Represents several time-points
    args: As above; the optional arguments must be the same as for Gop.
ihfun returns a 2D ndarray which represents the inhomogeneous source term
(s(t) in the paper) in the time points specified by t. Different time-points 
are represented by separate columns. The number of rows of the output of ihfun
is identical to the dimension of the state vector.
The default None value means that there is no inhomogeneous term (s(t) \equiv 0).
ev_domain (ndarray/list/tuple): The (estimated) boundaries of the eigenvalue
domain of G(u, t); required when a Chebyshev algorithm is used for the computation of
the function of matrix.
The default None value means that the Arnoldi algorithm is employed instead.
The general form for an ndarray is np.r_[lambda_min, lambda_max], where
lambda_min is the (estimated) lowest eigenvalue of G(u, t), and lambda_max is
the (estimated) highest eigenvalue.
Since G(u, t) is time-dependent, its eigenvalue domain is also time-dependent.
Hence, ev_domain has to cover all the eigenvalue domains of G(u, t) throughout
the propagation process.
Niter (int): The maximal allowed number of iterations for all time-steps
excluding the first
Niter1st (int): The maximal allowed number of iterations for the first time-step
data (dictionary/None): The required data computed as a preparation stage. Can be produced by the
function SGdata.
The default None value means that the dictionary is computed in the program.
The default maximal Nt_ts is 13, and the maximal Nfm is 15.
Important note: If the propagation itself consumes a small amount of time, but the
overall computation consists of many propagations (e.g. in optimal control
problems), it's not recommended to use the default None value.
data_type (type): The data-type object of the output solution
display_mode (boolean): True means that all warnings are 
displayed, even during propagation, including warnings of error that
exceeds the tolerance. False (default) means that only warnings of possibility
of failure of the entire process are displayed during propagation.
Warnings of error that exceeds the tolerance are displayed only at
the end of the propagation.
save_memory (boolean): False (default) means that the solution at all propagation
grid time-points is stored in memory and is contained in the output dictionary 
history (key 'U'). True means the opposite.
tol_f (float/None): The tolerance of the f_fun computation; None (default) means
that it is set to the default value: The tolerance per time-step if tol is assigned
a numerical value, and np.spacing(1) if tol is None.
tol1st (float/None): The tolerance of the iterative process in the first time-step,
if it is desired to treat it differently than the other time-steps; applies
also when tol=None, and the manual mode is applied for the other time-steps.
None (default) means that it is set to the default value of the tolerance per
time-step (tol_ts in the program).
comp_conv_cheb (boolean/None): True means that the convergence error estimation
by integration at the Chebyshev time-points is computed. False means the opposite.
None (default, recommended) means that this is determined by the value of Gdiff_matvecs:
True if Gdiff_matvecs==0, and False otherwise.
comp_texp_odd (boolean): True (default) means that the exact time-expansion
error estimation for odd Nt_ts is computed. False means that the cheap and 
inaccurate estimation is computed only.
args: Optional additional arguments for the input functions: Gop, Gdiff_op,
and ihfun; they should be written in the place of *args separated by commas.

Output:
U (2D ndarray): Contains the solution at the time points specified in tgrid; 
the different time-points are respresented by separate columns.
history: A dictionary with the details of the propagation process; contains the
following keys:
    mniter (float): The mean number of iteration for a time step, excluding the first step,
    which typically requires more iterations. Usually, mniter should be 1 for ideal
    efficiency of the algorithm.
    matvecs (dictionary): The number of G(u, t) operations on a vector; a nested
    dictionary with the following keys:
        propagation (int): The number of Hamiltonian operations required for the
        propagation process;
        error (int): The number of Hamiltonian operations required for the error
        estimation;
        total (int): The total number of Hamiltonian operations (the sum of the two
        other keys).
    t (1D ndarray): The time grid of propagation
    U (2D ndarray): The solution at t, where different time-points are represented
    by separate columns. Computed if save_memory==False.
    niter (1D ndarray): the number of iterations for each time-steps 
    est_errors (dictionary): Contains the total estimated relative errors, based 
    on the assumption of additive error accumulation during the propagtion.
    It has the following keys:
        texp_cheap (float): The estimated relative error resulting from the time-expansions in
        each time-step; the estimation is relatively cheap numerically, but
        overestimates the error, typically by 2-3 orders of magnitude.
        texp_exact (float): The estimated relative error resulting from the time-expansions in
        each time-step; the resulting estimation is much more precise than 
        texp_cheap. However, in the case of odd Nt_ts the estimation is more
        expansive numerically. Not computed for odd Nt_ts if comp_texp_odd==False
        (the default is True).
        texp_cheap_odd (float; for odd Nt_ts only): The estimated relative error 
        resulting from the time-expansions in each time-step; more precise than
        texp_cheap, but less safe - it might underestimate the error. Less
        precise than texp_exact.
        fm (float): The estimated relative error resulting from the
        computation of the function of matrix.
        conv (float): The total estimated relative convergence error. It is computed
        based on the different convergence error estimations
        for each time-step. The sums of the different estimations are represented
        by the following fields.
        conv_cheb (float): The total estimated relative convergence error based on
        numerical integration at the Chebyshev points. Becomes demanding for
        Gdiff_matvecs>0. The default is not to perform this computation in this
        case.
        conv_texp (float): Based on multiplication of the integration by the end-point
        of the interval by a correction factor; the correction factor is based
        on the assumption that the extrapolation error is dominated by the
        time-expansion error.
        conv_fm (float): Based on multiplication of the integration by the end-point
        of the interval by a correction factor; the correction factor is based
        on the assumption that the extrapolation error is dominated by the
        function of matrix computation error.
        total (float): The total estimated error; it is the sum of the keys
        texp_exact, fm and conv. In case that texp_exact isn't computed, it is
        replaced by texp_cheap.
        stab_factor (float/ndarray): The stability factor; it is a scalar for 
        the Chebyshev algorithm. For the Arnoldi algorithm it is an ndarray,
        where its value is computed separately for each time-step.
        Problematic values are in the order of ~0.1-1 or higher.
    all_errors (dictionary): Contains the estimated errors for all the individual
    time-steps. It has the following keys:
        texp_cheap, texp_exact, texp_cheap_odd (1D ndarray for all): The 
        time-expansion error estimations for all time-steps (see the description
        for the fields of the nested dictionary est_errors).
        fm_error (1D ndarray): The function of matrix error estimation for all time-steps
        conv_error, conv_error_cheb, conv_error_texp, conv_error_fm (1D ndarray for all):
        The convergence error estimations for all time-steps (see the description
        for the fields of the nested dictionary est_errors).
        total (1D ndarray): The total error estimation for all time-steps
        reldif (1D ndarray): The relative difference of the solution from the 
        previous iterated solution for all time-steps. Provides an estimation
        of the error of the previous iterated solution.
"""
    if data is None:
#   If the data dictionary isn't provided:
        data = SGdata(13, 15, Niter, Niter1st)    
#   Setting defaults:
    if not tol is None:
        # The tolerance is specified by the user; allowed error per time-step:
        tol_ts = tol/Nts
        # The number of iterations is determined adaptively by the allowed tolerance:
        tol_mode = True
    else:
        # Employing a manual mode for determination of the number of
        # iterations. The number of iterations is predefined,
        # Niter1st for the 1st time-step, and Niter for
        # the rest of the time-steps.
        tol_mode = False
    if tol_f is None:
        # The default tolerance of the f_{Nt_ts}(z, t) computation:
        if tol_mode:
            tol_f = tol_ts
        else:
            tol_f = np.spacing(1)
    if tol1st is None:
        tol1st_mode = False
    else:
        # The 1st time-step is treated differently than the rest of the
        # time-steps with its own tolerance parameter:
        tol1st_mode = True
    if comp_conv_cheb is None:
        # The default for the computation of the convergence error by
        # integration at Chebyshev points; the computation won't be
        # performed if the evaluation of the extended "inhomogeneous" vector costs
        # non-negligible computational effort:
        if Gdiff_matvecs == 0:
            comp_conv_cheb = True
        else:
            comp_conv_cheb = False
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
    if comp_conv_cheb:
        # Chebyshev integration weights for caculation of history['est_errors']['conv_cheb']:
        wcheb = chebweights(Nt_ts, Tts)[:, np.newaxis]
    # The parity of Nt_ts:
    Nt_ts_is_even = (Nt_ts%2 == 0)
    if Nt_ts_is_even:
        # The index of the time-point in the middle of the time step; for
        # even Nt_ts it is the test-point:
        tmidi = Nt_ts
        # For time-expansion error computation:
        Etexp_factor_even = 8*np.abs(Tts/((Nt_ts**2 - 1)*(Nt_ts - 3)))
        # s_ext evaluation time points for error estimation (just one point
        # for even Nt_ts):
        texp_er_tpoints = Tts/2
        # The index of the test-point:
        texp_er_i = Nt_ts
        # The indices for the computation of \bar{G}(t)u(t) in s_ext (without the test-points;
        # note that this differs from SGfuns.SemiGlobal):
        s_ext_i = np.r_[0:Nt_ts]
    else:
        # For odd Nt_ts, the middle time-point is also in the middle of the
        # time-step:
        tmidi = Nt_ts//2
        # For the exact time-expansion error computation; used also in a variant of
        # the cheap estimation:
        Etexp_factor_odd = 4*np.abs(Tts*(Nt_ts - 1)/(Nt_ts*(Nt_ts**2 - 4)*(Nt_ts - 4)))
        if comp_texp_odd:
            # If the exact time-expansion error computation is performed:
            # The error estimaiton for odd Nt_ts requires two test points:
            texp_er_tpoints = np.array([(t_ts[tmidi] + t_ts[tmidi + 1])/2, (t_ts[tmidi] + t_ts[tmidi - 1])/2])
            # The indices of the test-points:
            texp_er_i = np.array([Nt_ts, Nt_ts + 1])
        else:
            # Only the cheap time-expansion error estimation is performed: 
            texp_er_tpoints = np.array([(t_ts[tmidi] + t_ts[tmidi + 1])/2])
            texp_er_i = Nt_ts
        # If Nt_ts is odd, the midpoint needn't be computed:
        s_ext_i = np.r_[0:tmidi, (tmidi + 1):Nt_ts]
        if comp_conv_cheb:
            # For odd Nt_ts, the value of the integrand at the middle point is
            # zero, and thus not included in the Chebyshev weights.
            wcheb = np.delete(wcheb, tmidi, axis=0)
    # For the cheap time-expansion error estimation:
    Etexp_factor_cheap = 4*Tts
    # The total number of internal time-points, including test-points:
    total_Nt_ts = Nt_ts + texp_er_tpoints.size
    # The number of points in s_ext_i:
    Ns_ext_i = s_ext_i.size
    # The interior time points of the current time step for interpolation,
    # and the next time step for extrapolation of the guess solution into the next step:
    t_2ts = np.r_[t_ts, texp_er_tpoints, Tts + t_ts[1:Nt_ts], Tts + texp_er_tpoints]    
    # The full propagation grid:
    propagation_grid = np.r_[np.kron(np.r_[tinit:(tf - 2*np.spacing(tf)):Tts], np.ones(total_Nt_ts - 1)) + \
                             np.tile(np.r_[t_ts[0:(Nt_ts - 1)], texp_er_tpoints], Nts), tf]
    # The -2*np.spacing(tf) is in order to avoid undesirable outcomes of roundoff errors.
    history = {
        't': np.delete(propagation_grid, np.r_[(Nt_ts - 1):(Nts*Nt_ts):Nt_ts]),
        'niter': np.zeros(Nts),
        'matvecs': {},
        'est_errors': {},
        'all_errors': {
            'texp_cheap': np.zeros(Nts),
            'fm': np.zeros(Nts),
            'conv': np.zeros(Nts),
            'conv_texp': np.zeros(Nts),
            'conv_fm': np.zeros(Nts),
            'reldif': np.zeros(Nts)
            }
        }
    if not save_memory:
        history['U'] = np.zeros((Nu, Nts*(Nt_ts - 1) + 1), dtype=data_type, order = 'F')
        history['U'][:, 0] = ui
    if Nt_ts_is_even or comp_texp_odd:
        history['all_errors']['texp_exact'] = np.zeros(Nts)
    if comp_conv_cheb:
        history['all_errors']['conv_cheb'] = np.zeros(Nts)
    # Necessary for error estimation of f_{Nt_ts}(z, t):
    factorialNt_ts = math.factorial(Nt_ts)
    if not Arnoldi:
        # If the eigenvalue domain is specified, a Chebyshev approximation
        # for the function of matrix is employed.
        min_ev = ev_domain[0]
        max_ev = ev_domain[1]        
        # Computing the coefficients for the Chebyshev expansion of the
        # function of matrix, in all the interior time points.
        # CchebFts contains the coefficients of the current time step, and
        # CchebFnext contains the coefficients of the next one.
        Ccheb_f_comp = f_chebC(t_2ts[1:(2*total_Nt_ts - 1)], Nfm, Nt_ts, min_ev, max_ev, tol_f, factorialNt_ts)
        Ccheb_f_ts = Ccheb_f_comp[:, 0:(total_Nt_ts - 1)]
        Ccheb_f_next = Ccheb_f_comp[:, (total_Nt_ts - 1):(2*total_Nt_ts - 2)]   
        # Computing sampling points for the error test of the
        # function of matrix error at the maxima of the Chebyshev polynomial:
        cheb_testp = -np.cos(np.r_[0:(Nfm + 1)]*np.pi/Nfm)
        ztest = (cheb_testp*(max_ev - min_ev) + min_ev + max_ev)/2
        fztest = np.squeeze(f_fun(ztest, np.array([Tts]), Nt_ts, tol_f, factorialNt_ts))
        f_scalar_error = np.max(np.abs(chebc2result(Ccheb_f_ts[:, Nt_ts - 2], ev_domain, ztest) - fztest))
        # Stability factor:
        history['stab_factor'] = f_scalar_error*np.max(np.abs(ev_domain))**Nt_ts/factorialNt_ts
        # Estimated stability criterion (a boolean variable):
        instability = history['stab_factor']>0.1
        if instability:
            print('Warning: Instability in the propagation process may occur.')
    else:
        # For the Arnoldi algorithm, the stability test is performed in
        # each time step.
        instability = False
        history['stab_factor'] = np.zeros(Nts)
    # A boolean that indicates if the convergence has failed in at least one
    # time-step:
    conv_failure = False
    # Creating the dictionary of the output details of the propagation process:
    # Computing the matrix of the time Taylor polynomials.
    # timeMts contains the points in the current time step, and timeMnext
    # contains the points in the next time step:
    timeMcomp = maketM(t_2ts[1:(2*total_Nt_ts - 1)], Nt_ts)
    timeMts = timeMcomp[:, 0:(total_Nt_ts - 1)]
    timeMnext = timeMcomp[:, (total_Nt_ts - 1):(2*total_Nt_ts - 2)]    
    # Computing the coefficients of the transformation from the Newton
    # interpolation polynomial terms, to a Taylor like form:
    Cr2t = r2Taylor4(t_ts, Tts)
    # The extended "inhomogeneous" vectors:
    s_ext = np.zeros((Nu, total_Nt_ts), dtype=data_type, order = 'F')
    # Newton coefficients for the time-expansion of s_ext:
    Cnewton = np.zeros((Nu, total_Nt_ts), dtype=data_type, order = 'F')
    # The v vectors are defined recursively, and contain information about
    # the time dependence of the s_ext vectors:
    v_vecs = np.empty((Nu, Nt_ts + 1), dtype=data_type, order = 'F')
    # If there is no inhomogeneous term in the equation, ihfun == None, and there_is_ih == False.
    # If there is, there_is_ih == true.
    there_is_ih = (ihfun != None)
    if there_is_ih:
        s = np.empty((Nu, total_Nt_ts), dtype=data_type, order = 'F')
        s[:, 0] = ihfun(tinit, *args).squeeze()
    # The 0'th order approximation is the first guess for the first time step.
    # Each column represents an interior time point in the time step:
    Uguess = guess0(ui, total_Nt_ts)
    Unew = np.empty((Nu, total_Nt_ts), dtype=data_type, order = 'F')
    # These variables are used to determine which points in tgrid are in the computed time-step.  
    tgrid_lowi = 1
    tgrid_upi = 0
    for tsi in range(0, Nts):
        # The time of the interior time points within the time-step:
        t = propagation_grid[tsi*(total_Nt_ts - 1) + np.r_[np.r_[0:(Nt_ts - 1)], total_Nt_ts - 1, texp_er_i - 1]]
        # The first guess for the iterative process, for the convergence of the u
        # values. Each column represents an interior time point in the time step:
        Ulast = Uguess.copy()
        Unew[:, 0] = Ulast[:, 0]
        v_vecs[:, 0] = Ulast[:, 0]
        if there_is_ih:
            # Computing the inhomogeneous term:
            s[:, 1:total_Nt_ts] = ihfun(t[1:total_Nt_ts], *args)
        if comp_conv_cheb:
            # Computation of \bar{G}u at the Chebyshev points for
            # integration. Note that for odd Nt_ts, \bar{G}u = 0, and therefore not
            # calculated.
            Gdiff_u_new = Gdiff_op(Ulast[:, s_ext_i], t[s_ext_i], Ulast[:, tmidi], t[tmidi], *args)
        else:
            # Computation of \bar{G}u at the end-point of the time-step
            # only:
            Gdiff_u_new_end =\
                Gdiff_op(Ulast[:, Nt_ts - 1][:, np.newaxis], t[Nt_ts - 1], Ulast[:, tmidi], t[tmidi], *args).squeeze()
            # The first input has to be a 2D ndarray. The output is a 2D ndarray,
            # and has to be squeezed to 1D.
        # Starting an iterative process until convergence:
        niter = 0
        conv_error = np.inf
        while (tsi>0 and niter<Niter and ((tol_mode and conv_error>tol_ts) or not tol_mode)) or\
              (tsi == 0 and niter<Niter1st and\
                 ((not tol1st_mode and ((tol_mode and conv_error>tol_ts) or not tol_mode)) or\
                    (tol1st_mode and conv_error>tol1st))):
            # Setting the inhomogeneous s_ext vectors. 
            if comp_conv_cheb:
                s_ext[:, s_ext_i] = Gdiff_u_new
            else:
                s_ext[:, s_ext_i] =\
                    np.c_[Gdiff_op(Ulast[:, s_ext_i[0:(Ns_ext_i - 1)]], t[s_ext_i[0:(Ns_ext_i - 1)]],\
                                   Ulast[:, tmidi], t[tmidi], *args),\
                          Gdiff_u_new_end]
                # Note that for odd Nt_ts, s_ext[:, tmidi] is equivalent to
                # s[:, tmidi], and therefore not calculated.
            if there_is_ih:
                # Saving the previous \bar{G}u for calculation of the
                # convergence error. Required only when s_ext is different
                # than \bar{G}u - when there's a source term.
                if comp_conv_cheb:
                    Gdiff_u = Gdiff_u_new.copy()
                else:
                    Gdiff_u_end = Gdiff_u_new_end.copy()
                if not Nt_ts_is_even:
                    # For odd Nt_ts, the extended "source term" at the middle
                    # time-point is just the source term:
                    s_ext[:, tmidi] = s[:, tmidi]
                # The inhomogeneous term as added it to the s_ext vectors:
                s_ext[:, s_ext_i] = s_ext[:, s_ext_i] + s[:, s_ext_i]
            # Calculation of the coefficients of the form of Taylor
            # expansion, from the coefficients of the Newton
            # interpolation at the points t_ts.
            # The divided differences are computed by the function divdif.
            # For numerical stability, we have to transform the time points
            # in the time step, to points in an interval of length 4:
            Cnewton[:, 0:Nt_ts], diagonal = divdif(t_2ts[0:Nt_ts]*4/Tts, s_ext[:, 0:Nt_ts])
            # Calculating the Taylor like coefficients:
            Ctaylor = Cpoly2Ctaylor(Cnewton[:, 0:Nt_ts], Cr2t)
            # Calculation of the v vectors:
            for polyi in range(1, Nt_ts + 1):
                v_vecs[:, polyi] = (Gop(Ulast[:, tmidi], t[tmidi], v_vecs[:, polyi-1], *args)
                                    + Ctaylor[:, polyi - 1])/polyi
            if not np.min(np.isfinite(v_vecs[:, Nt_ts])):
                # It means that the algorithm diverges.
                # In such a case, change Nts, Nt_ts and/or Nfm.
                print(f'Error: The algorithm diverges (in time step No. {tsi + 1}).')
                history['niter'][tsi] = niter
                if tsi>0:
                    history['mniter'] = history['niter'][1:(tsi + 1)].sum()/tsi
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
                # Calculation of the solution at all time points
                # within the time step:
                Unew[:, 1:total_Nt_ts], f_abs_error, fUerror = \
                    Ufrom_vArnoldi1(v_vecs, timeMts, Upsilon[:, 0:Nfm],
                                   RvKr, samplingp, capacity, tol_f, factorialNt_ts, estimate_error=True)
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
                # Calculation of the solution at all time points
                # within the time step:
                Unew[:, 1:total_Nt_ts], fUerror = Ufrom_vCheb1(v_vecs, timeMts, Vcheb, Ccheb_f_ts, f_scalar_error)
            # The relative difference from the previous solution in the
            # iterative process:
            reldif = norm(Unew[:, Nt_ts - 1] - Ulast[:, Nt_ts - 1])/norm(Ulast[:, Nt_ts - 1])
            if niter == 0:
                reldif1st = reldif
            # Convergence error estimation:
            if comp_conv_cheb:
                # Calculation of the new \bar{G}u vectors:
                Gdiff_u_new = Gdiff_op(Unew[:, s_ext_i], t[s_ext_i], Unew[:, tmidi], t[tmidi], *args)
                if there_is_ih:
                    # The cheap estimation is based on the end-point of the
                    # integration interval:
                    conv_error_cheap =\
                        norm(Gdiff_u_new[:, Ns_ext_i - 1] - Gdiff_u[:, Ns_ext_i - 1])*Tts/norm(Unew[:, Nt_ts - 1])
                    # An accurate estimation based on integration at the
                    # Chebyshev points:
                    conv_error_cheb = norm((Gdiff_u_new - Gdiff_u)@wcheb)/norm(Unew[:, Nt_ts - 1])
                else:
                    # In the homogeneous case, s_ext is equivalent to the
                    # previous \bar{G}u, which needn't be stored in memory:
                    conv_error_cheap = norm(Gdiff_u_new[:, Ns_ext_i - 1] - s_ext[:, Nt_ts - 1])*Tts/norm(Unew[:, Nt_ts - 1])
                    conv_error_cheb = norm((Gdiff_u_new - s_ext[:, s_ext_i])@wcheb)/norm(Unew[:, Nt_ts - 1])
                if tsi > 0:
                    # Computation of the convergence error based on
                    # multiplication of the cheap estimation by correction
                    # coefficients:
                    conv_error_texp = conv_error_cheap*data['correct_texp'][niter, Nt_ts - 2]
                    conv_error_fm = conv_error_cheap*data['correct_fm'][niter, Nt_ts + Nfm - 2]
                    # The accepted convergence error is the maximum of the
                    # error by Chebyshev integration and the error based on
                    # multiplication by the correction coefficient. It is
                    # assumed that the correction coefficient that yields
                    # the closest estimation to conv_error_cheb is the
                    # right one.
                    if np.abs(conv_error_texp - conv_error_cheb)<=np.abs(conv_error_fm - conv_error_cheb):
                        # It is assumed that the extrapolation error is
                        # dominated by the time-expansion error:
                        conv_error = max(conv_error_texp, conv_error_cheb)
                    else:
                        # It is assumed that the extrapolation error is
                        # dominated by the function of matrix error:
                        conv_error = max(conv_error_fm, conv_error_cheb)
                    # Note: Actually, it's possible to estimate both the
                    # time-expansion extrapolation error and the function
                    # of matrix extrapolation error. Then, it's possible to
                    # determine the right correction coefficient. I have no
                    # time right now for that. Hopefully in a future
                    # version.
                else:
                    # Estimation for the first time-step based on
                    # correction of the cheap estimation. It isn't related
                    # to extrapolation error, and thus conv_error_texp
                    # and conv_error_fm are set to the same value:
                    conv_error_texp = conv_error_cheap*data['correct_1st'][niter]
                    conv_error_fm = conv_error_texp.copy()
                    conv_error = max(conv_error_texp, conv_error_cheb)
            else:
                # The convergence error by Chebyshev integration isn't
                # computed. The estimation is solely based on the cheap
                # estimation multiplied by a correction coefficient.
                Gdiff_u_new_end =\
                    Gdiff_op(Unew[:, Nt_ts - 1][:, np.newaxis], t[Nt_ts - 1], Unew[:, tmidi], t[tmidi], *args).squeeze()
                if there_is_ih:
                    conv_error_cheap = norm(Gdiff_u_new_end - Gdiff_u_end)*Tts/norm(Unew[:, Nt_ts - 1])
                else:
                    conv_error_cheap = norm(Gdiff_u_new_end - s_ext[:, Nt_ts - 1])*Tts/norm(Unew[:, Nt_ts - 1])
                if tsi > 0:
                    conv_error_texp = conv_error_cheap*data['correct_texp'][niter, Nt_ts - 2]
                    conv_error_fm = conv_error_cheap*data['correct_fm'][niter, Nt_ts + Nfm - 2]
                    conv_error = max(conv_error_texp, conv_error_fm)
                else:
                    conv_error_texp = conv_error_cheap*data['correct_1st'][niter]
                    conv_error_fm = conv_error_texp.copy()
                    conv_error = conv_error_texp.copy()
            # The solution before the last one is stored for
            # computation of the time-expansion error after the end of the
            # iterative process:
            Ulast2 = Ulast.copy()
            Ulast = Unew.copy()
            niter += 1
        if Arnoldi:
            # Stability factor computation:
            f_scalar_error = f_abs_error/norm(v_vecs[:, Nt_ts])
            history['stab_factor'][tsi] = f_scalar_error*np.max(np.abs(eigval))**Nt_ts/factorialNt_ts
            if not instability:
            # If a possibility of instability hasn't been detected yet, the
            # stability criterion is checked:
                instability = history['stab_factor'][tsi]>0.1
                if instability:
                    print(f'Warning: Instability in the propagation process may occur (detected in time step No. {tsi + 1}).')
        # Detection of the first appearance of convergence failure:
        if niter>1 and reldif>reldif1st and not conv_failure:
            conv_failure = True
            print(f'Warning: Convergence failure (first occured in time step No. {tsi + 1}).')
            # In such a case, change Nts, Nt_ts and/or Nfm.
        # Time-expansion error estimation.
        # Computing the extended source term at the test time-points:
        if Nt_ts_is_even:
            if there_is_ih:
                # The extended "source term" at the midpoint of the time-step is just the source term: 
                s_ext[:, texp_er_i] = s[:, texp_er_i]
            # If there's no source term, it remains 0 throughout the propagation.
        else:
            if comp_texp_odd:
                s_ext[:, texp_er_i] = Gdiff_op(Ulast2[:, texp_er_i], t[texp_er_i], Ulast2[:, tmidi], t[tmidi], *args)
            else:
                # texp_er_i has one term only, which requires dimensional adjustments
                # for the input and output of Gdiff_op:
                s_ext[:, texp_er_i] =\
                    Gdiff_op(Ulast2[:, texp_er_i][:, np.newaxis], t[texp_er_i], Ulast2[:, tmidi], t[tmidi], *args).squeeze()
            if there_is_ih:
                s_ext[:, texp_er_i] = s_ext[:, texp_er_i] + s[:, texp_er_i]
        # Computing the additional divided differences for the test-points:
        if Nt_ts_is_even or not comp_texp_odd:
            # The first term in the output tuple of new_divdif is squeezed for
            # dimensional match:
            Cnewton[:, texp_er_i] =\
                new_divdif(t_2ts[0:total_Nt_ts]*4/Tts, s_ext[:, texp_er_i][:, np.newaxis], diagonal)[0].squeeze()
        else:
            Cnewton[:, texp_er_i], _ = new_divdif(t_2ts[0:total_Nt_ts]*4/Tts, s_ext[:, texp_er_i], diagonal)
        normUnew = norm(Unew[:, Nt_ts - 1])
        normCnewton_t_er = norm(Cnewton[:, Nt_ts])
        texp_error_cheap = Etexp_factor_cheap*normCnewton_t_er/normUnew
        if Nt_ts_is_even:
            texp_error_exact = Etexp_factor_even*normCnewton_t_er/normUnew
        elif comp_texp_odd:
            # The odd estimation is computed only if this is specified in the
            # options dictionary.
            texp_error_exact =\
                Etexp_factor_odd*norm(4*Cnewton[:, Nt_ts + 1] -\
                                     Tts*Gop(Ulast[:, tmidi], t[tmidi], Cnewton[:, Nt_ts], *args))/normUnew
        if display_mode and tol_mode:
            if conv_error>tol_ts:
                print(f'Warning: The estimated error of the iterative process ({conv_error}) is larger than the requested tolerance.\nThe solution might be inaccurate (in time step No. {tsi + 1}).')
            if Nt_ts_is_even or comp_texp_odd:
                if texp_error_exact>tol_ts: 
                    print(f'Warning: The estimated error of the time expansion ({texp_error_exact}) is larger than the requested tolerance.\nThe solution might be inaccurate (in time step No. {tsi + 1}).')
            elif texp_error_cheap>tol_ts:
                print(f'Warning: The estimated error of the time expansion ({texp_error_cheap}) is larger than the requested tolerance.\nThe solution might be inaccurate (in time step No. {tsi + 1}).')
            if fUerror>tol_ts:
                print(f'Warning: The estimation of the error resulting from the function of matrix ({fUerror}) is larger than the requested tolerance.\nThe solution might be inaccurate (in time step No. {tsi + 1}).')
        history['all_errors']['texp_cheap'][tsi] = texp_error_cheap
        if Nt_ts_is_even or comp_texp_odd:
            history['all_errors']['texp_exact'][tsi] = texp_error_exact
        history['all_errors']['fm'][tsi] = fUerror
        history['all_errors']['reldif'][tsi] = reldif
        history['all_errors']['conv'][tsi] = conv_error
        history['all_errors']['conv_texp'][tsi] = conv_error_texp
        history['all_errors']['conv_fm'][tsi] = conv_error_fm
        if comp_conv_cheb:
            history['all_errors']['conv_cheb'][tsi] = conv_error_cheb
        history['niter'][tsi] = niter
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
                    Ufrom_vArnoldi1(v_vecs, timeMout, Upsilon[:, 0:Nfm],
                                   RvKr, samplingp, capacity, tol_f, factorialNt_ts)
            else:
                Ccheb_f_out = f_chebC(tgrid[tgrid_lowi:(tgrid_upi + 1)] - t[0],
                                      Nfm, Nt_ts, min_ev, max_ev, tol_f, factorialNt_ts)
                U[:, tgrid_lowi:(tgrid_upi + 1)] = Ufrom_vCheb1(v_vecs, timeMout, Vcheb, Ccheb_f_out)
            tgrid_lowi = tgrid_upi + 1
        # If one of the points in tgrid coincides with the point of the
        # propagation grid:
        if np.abs(t[Nt_ts - 1] - tgrid[tgrid_upi + 1])<=np.spacing(np.abs(t[Nt_ts - 1]))*10:
            tgrid_upi = tgrid_upi + 1
            U[:, tgrid_upi] = Unew[:, Nt_ts - 1]
            tgrid_lowi = tgrid_upi + 1
        if not save_memory:
            history['U'][:, (tsi*(Nt_ts - 1) + 1):((tsi + 1)*(Nt_ts - 1) + 1)] = Unew[:, 1:Nt_ts]
        # The new guess is an extrapolation of the solution within the
        # previous time step:
        Uguess[:, 0] = Unew[:, Nt_ts - 1]
        if Arnoldi:
            Uguess[:, 1:total_Nt_ts] = \
                Ufrom_vArnoldi1(v_vecs, timeMnext, Upsilon[:, 0:Nfm],
                               RvKr, samplingp, capacity, tol_f, factorialNt_ts)
        else:
            Uguess[:, 1:total_Nt_ts] = Ufrom_vCheb1(v_vecs, timeMnext, Vcheb, Ccheb_f_next)
        if there_is_ih:
            s[:, 0] = s[:, Nt_ts - 1]
    if not Nt_ts_is_even:
        # A variant of the cheap estimation for odd Nt_ts only; in general
        # more accurate than history['est_errors']['texp_cheap'], but less safe:
        history['all_errors']['texp_cheap_odd'] = history['all_errors']['texp_cheap']*Etexp_factor_odd/Etexp_factor_cheap
        history['est_errors']['texp_cheap_odd'] = history['all_errors']['texp_cheap_odd'].sum()
    if Nt_ts_is_even or comp_texp_odd:
        history['all_errors']['total'] =\
            history['all_errors']['texp_exact'] + history['all_errors']['fm'] + history['all_errors']['conv']
    else:
        history['all_errors']['total'] =\
            history['all_errors']['texp_cheap_odd'] + history['all_errors']['fm'] + history['all_errors']['conv']
    history['est_errors']['texp_cheap'] = history['all_errors']['texp_cheap'].sum()
    if Nt_ts_is_even or comp_texp_odd:
        history['est_errors']['texp_exact'] = history['all_errors']['texp_exact'].sum()
    history['est_errors']['fm'] = history['all_errors']['fm'].sum()
    history['est_errors']['conv'] = history['all_errors']['conv'].sum()
    history['est_errors']['conv_texp'] = history['all_errors']['conv_texp'].sum()
    history['est_errors']['conv_fm'] = history['all_errors']['conv_fm'].sum()
    if comp_conv_cheb:
        history['est_errors']['conv_cheb'] = history['all_errors']['conv_cheb'].sum()
    if Nt_ts_is_even or comp_texp_odd:
        history['est_errors']['total'] =\
            history['est_errors']['texp_exact'] + history['est_errors']['fm'] + history['est_errors']['conv']
    else:
        history['est_errors']['total'] =\
            history['est_errors']['texp_cheap'] + history['est_errors']['fm'] + history['est_errors']['conv']
    if tol_mode:
        if Nt_ts_is_even or comp_texp_odd:
            if history['est_errors']['texp_exact']>tol:
                print(f"Warning: The total estimated error of the time expansion ({history['est_errors']['texp_exact']}) is larger than the requested tolerance.\nThe solution might be inaccurate.")
        elif history['est_errors']['texp_cheap']>tol:
            print(f"Warning: The total estimated error of the time expansion ({history['est_errors']['texp_cheap']}) is larger than the requested tolerance.\nThe solution might be inaccurate.")
        if history['est_errors']['fm']>tol:
            print(f"Warning: The total estimated error resulting from the function of matrix computation ({history['est_errors']['fm']}) is larger than the requested tolerance.\nThe solution might be inaccurate.")
        if history['est_errors']['conv']>tol:
            print(f"Warning: The total estimated error resulting from the iterative process ({history['est_errors']['conv']}) is larger than the requested tolerance.\nThe solution might be inaccurate.")
        if history['est_errors']['total']>tol:
            print(f"Warning: The total estimated error ({history['est_errors']['total']}) is larger than the requested tolerance.\nThe solution might be inaccurate.")
    # The overall number of interations:
    allniter = history['niter'].sum()
    # The mean number of iterations, where the first time-step is excluded:
    history['mniter'] = (allniter - history['niter'][0])/(Nts - 1)
    # The number of Hamiltonian operations required for the propagation:
    if Arnoldi:
        history['matvecs']['propagation'] = allniter*(Nt_ts + Ns_ext_i*Gdiff_matvecs + Nfm)
    else:
        history['matvecs']['propagation'] = allniter*(Nt_ts + Ns_ext_i*Gdiff_matvecs + Nfm - 1)
    # The number of Hamiltonian operations required for the error estimation:
    if comp_conv_cheb:
        history['matvecs']['error'] = Nts*Gdiff_matvecs*Ns_ext_i
        # The situation that this is more than 0 isn't recommended.
    else:
        history['matvecs']['error'] = Nts*Gdiff_matvecs
    if Nt_ts_is_even or not comp_texp_odd:
        history['matvecs']['error'] = history['matvecs']['error'] + Nts*Gdiff_matvecs
    else:
        history['matvecs']['error'] = history['matvecs']['error'] + Nts*(2*Gdiff_matvecs + 1)
    history['matvecs']['total'] = history['matvecs']['propagation'] + history['matvecs']['error']
    return U, history