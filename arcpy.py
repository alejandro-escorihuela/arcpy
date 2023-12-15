#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# 13-12-2023
# alex
# arcpy.py

import numpy as np
from scipy import optimize
from scipy import linalg
from numpy import linalg as LA
import time as tm

def f(x):
    # return [x[0]**2+x[1]**2 - 4]
    return [x[0]**2+x[1]**2+x[2]**2 - 4, x[0]+x[1]-x[2]]

def g(x):
    return ((x[0]-1)**2 - 1)

def fhiperpla(x, *val):
    b0, dt, func, x0 = val[0], val[1], val[2], val[3]
    bpx = 0.0
    for i in range(len(b0)):
        bpx += b0[i]*(x[i] - x0[i])
    ret = []
    for i in range(len(x0) - 1):
        ret.append(func(x)[i])
    ret.append(bpx - dt)
    return ret

def calcbeta(f, x0, b0):
    F = lambda x: [f(x)[i] for i in range(len(x0) - 1)] + [0.0]
    fp = optimize.approx_fprime(x0, F, epsilon = 1e-10)
    fp[-1] = b0.copy()
    ti = [0.0]*(len(x0) - 1) + [1.0]
    bi = linalg.solve(fp, ti)
    bi = bi/LA.norm(bi, 2)
    return bi

def newtonsim(func, x0, args = (), full_output = 0, tolx = 1e-10, lam = 0.8, kmax = 5, told = 1e-6):
    jac = optimize.approx_fprime(x0, func, 1e-10, *args)
    stop, it = False, 0
    xj = x0.copy()
    while not stop:
        b = -np.array(func(xj, *args))
        dx = linalg.solve(jac, b)
        xj = xj + dx
        stop = LA.norm(dx, 1) < 1e-10
        it += 1
    print(it)
    return xj

def arcpy(f, g, x0, s, t0, tf, n):
    xi = x0.copy()
    bi = [s] + [0.0]*(len(x0) - 1)
    h = (tf - t0)/n
    while (t0 < tf):
        bi = calcbeta(f, xi, bi)
        t1 = t0 + h
        # res = optimize.fsolve(fhiperpla, xi, args = (bi, h, f, xi))
        res = newtonsim(fhiperpla, xi, args = (bi, h, f, xi))
        xi = res.copy()
        t0 = t1
        print(t0, xi, f(xi), g(xi))
    return xi

if __name__ == "__main__":
    # x0 = [-np.sqrt(2), np.sqrt(2)]
    tmp0 = tm.time()
    x0 = [-2*np.sqrt(2/3), np.sqrt(2/3), -np.sqrt(2/3)]
    x1 = arcpy(f, g, x0, 1.0, 0.0, 70.0, 100)
    print("t =", tm.time() - tmp0)
    print(x1)

