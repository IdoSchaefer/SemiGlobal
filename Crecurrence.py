# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 08:03:14 2024

@author: Ido Schaefer
"""

import numpy as np

def recurrence_coefs(Cinit, Norders, f_rec, order_init=0, data_type=None):# Recurrence coefficient computation
    """
The function computes a set of coefficients which are defined by a
recurrence relation, from a set of initial coefficients Cinit and a
function which defines the recurrence rule.
Input:
Cinit (ndarray, 1D or 2D): A matrix; contains a set of initial coefficients which are
involved in the iterative process. The row index represents different
orders of the recursive process, and the column index represents the
different coeffcients of the same order in the recursive process.
Note: In order to avoid repeated memory allocation, the number of columns
should be the same as the final output matrix C.
Norders (int): The number of orders in the recursive process to be computed
f_rec (function object): A function of the form f_rec(previousC, prev_order);
computes a new order of coefficients from the previous orders.
   Input of f_rec:
   previousC (ndarray, 1D or 2D) : A matrix of the previous coefficients of the form of Cinit.
   prev_order (int): The order of previousC
   Output of f_rec: a row vector of the new coefficients.
order_init (int): The order of the coefficients in Cinit.
data_type (type): The data type of the output ndarray C.
Output:
C (2D ndarray): The set of all coeffcients, including Cinit; the row and
column index represent the same as in Cinit.
"""
    if data_type is None:
        # The default is the type of the input coefficients:
        data_type = Cinit.dtype.type
    if Cinit.ndim == 1:
        Ncoefs = Cinit.size
        Nrec = 1
    else:
        Nrec, Ncoefs = np.shape(Cinit)
    C = np.zeros((Norders + Nrec, Ncoefs), dtype=data_type)
    C[0:Nrec, :] = Cinit
    for Ci in range(Nrec, (Nrec + Norders)):
        C[Ci, :] = f_rec(C[(Ci - Nrec):Ci, :].squeeze(), order_init + Ci)
    return C
                

