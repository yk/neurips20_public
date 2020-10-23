#!/usr/bin/env python3

import itertools as itt

import numpy as np
from absl import logging


def _eigenvalue(A, v):
    return v.dot(A.dot(v))


def _singular_value(A, u, v):
    return u.dot(A.dot(v))


def normalize(v):
    v /= np.linalg.norm(v)
    return v


def power_method_square_matrix(A, v0=None, max_iters=100000, tol=1e-9):
    assert A.shape[0] == A.shape[1]
    d = A.shape[0]

    if v0 is None:
        v0 = np.ones(d, A.dtype) / np.sqrt(d)

    ev = _eigenvalue(A, v0)

    def fn(v):
        v = A.dot(v)
        normalize(v)
        return v

    def stop_fn(v):
        nonlocal ev
        ev_new = _eigenvalue(A, v)
        if np.abs(ev - ev_new) < tol:
            return True
        ev = ev_new

    v1 = power_method_symmetric(fn, v0, stop_fn, max_iters=max_iters)
    ev = _eigenvalue(A, v1)

    return ev, v1


def power_method_symmetric(fn, v0, stop_fn, max_iters=100000):
    for i in range(max_iters):
        v0 = fn(v0)
        if stop_fn(v0):
            break
    else:
        logging.warning('Power method did not converge')
    logging.debug('Power method took {} iterations'.format(i))

    return v0


def power_method_asymmetric(forward_backward_fn, v0, stop_fn, max_iters=100000, log_max_iters_reached=True):
    u0 = v0
    for i in range(max_iters):
        u0, v0 = forward_backward_fn(v0)
        if stop_fn(u0, v0):
            break
    else:
        if log_max_iters_reached:
            logging.warning('Power method did not converge')
    logging.debug('Power method took {} iterations'.format(i))

    return u0, v0



def power_method_nonsquare_matrix(A, v0=None, max_iters=100000, tol=1e-9):
    m, n = A.shape
    u0 = np.ones(m, A.dtype) / np.sqrt(m)
    if v0 is None:
        v0 = np.ones(n, A.dtype) / np.sqrt(n)

    sv = _singular_value(A, u0, v0)

    def forward_fn(v):
        u = A.dot(v)
        normalize(u)
        return u

    def backward_fn(u):
        v = A.T.dot(u)
        normalize(v)
        return v

    def forward_backward_fn(v):
        u = forward_fn(v)
        v = backward_fn(u)
        return u, v

    def stop_fn(u, v):
        nonlocal sv
        sv_new = _singular_value(A, u, v)
        if np.abs(sv - sv_new) < tol:
            return True
        sv = sv_new

    u1, v1 = power_method_asymmetric(forward_backward_fn, v0, stop_fn)
    sv = _singular_value(A, u1, v1)

    return sv, u1, v1
