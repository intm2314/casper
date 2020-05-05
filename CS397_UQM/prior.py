#!/usr/bin/env python
#
import numpy as np

#Model Parameters
def prior_q(x):
    if (x <= 0) | (x >= 1):
        return -1 * np.inf
    else:
        return 1.

def prior_beta(x):
    if (x <= 0) | (x >= 1):
        return -1 * np.inf
    else:
        return 1.
        
def prior_k(x):
    if (x <= 0) | (x >= 1):
        return -1 * np.inf
    else:
        return 1.

def prior_c1(x):
    if (x <= 0):
        return -1 * np.inf
    else:
        return 1.
        
def prior_c2(x):
    if (x <= 0):
        return -1 * np.inf
    else:
        return 1.
        
def prior_c3(x):
    if (x <= 0):
        return -1 * np.inf
    else:
        return 1.
        
def prior_deq(x):
    if (x <= 0) | (x >= 1):
        return -1 * np.inf
    else:
        return 1. 

def prior_deqq(x):
    if (x <= 0) | (x >= 1):
        return -1 * np.inf
    else:
        return 1.

def prior_diq(x):
    if (x <= 0) | (x >= 1):
        return -1 * np.inf
    else:
        return 1.

def prior_delta(x):
    if (x <= 0) | (x >= 1):
        return -1 * np.inf
    else:
        return 1.

def prior_gamma(x):
    if (x <= 0) | (x >= 1):
        return -1 * np.inf
    else:
        return 1.
        
#Initial Conditions
def prior_E0(x):
    if (x <= 0):
        return -1 * np.inf
    else:
        return 1.

def prior_I0(x):
    if (x <= 0):
        return -1 * np.inf
    else:
        return 1.

def prior(q,beta,k,c1,c2,c3,deq,deqq,diq,delta,gamma,E0,I0):
    return prior_q(q) + prior_beta(beta) + prior_k(k) + prior_c1(c1) \
    + prior_c2(c2) + prior_c3(c3) + prior_deq(deq) + prior_deqq(deqq) \
    + prior_diq(diq) + prior_delta(delta) + prior_gamma(gamma) \
    + prior_E0(E0) + prior_I0(I0)
