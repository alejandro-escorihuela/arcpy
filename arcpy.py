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

def jacobian(x0, f, *args):
    eps = 1e-5
    xi = x0.copy()
    dim = len(x0)
    jac = np.zeros((dim, dim))
    for i in range(dim):
        xi[i] -= eps
        fxminus = f(xi, *args)
        xi[i] += eps
        fxplus = f(xi, *args)
        for j in range(dim):
            jac[i][j] = (fxplus[j] - fxminus[j])/eps
    return np.transpose(jac)

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
    # fp = optimize.approx_fprime(x0, F)
    fp = jacobian(x0, F)
    fp[-1] = b0.copy()
    ti = [0.0]*(len(x0) - 1) + [1.0]
    bi = linalg.solve(fp, ti)
    bi = bi/LA.norm(bi, 2)
    return bi

def newtonsim(func, x0, told, args = (), tolx = 1e-10, lam = 0.8, kmax = 5):
    stop, it, iter_max, k = False, 0, 20, 0
    xj = x0.copy()
    nda = 0.0
    #jac = optimize.approx_fprime(x0, func, 1e-10, *args)
    jac = jacobian(x0, func, *args)
    while not stop:
        b = -np.array(func(xj, *args))
        dx = linalg.solve(jac, b)
        xj = xj + dx
        ndx = LA.norm(dx, 1)
        if it == 0:
            nd0 = ndx
        if ndx > lam*nda:
            k +=1
        stop = ndx < 1e-10 or ndx < told*nd0 or it > iter_max or k > kmax
        nda = ndx
        it += 1
    return xj, it

def nexth(h, param):
    it, conv, told, method = param
    ito_hybr, ito_new = 25, 7
    hamin, hamax = 1e-6, 0.05
    hnova = h
    if not conv:
        hnova = canviarh(h, -1, hamin, hamax)
    else:
        if method == "hybr":
            if it <= ito_hybr:
                hnova = canviarh(h, +1, hamin, hamax)
            else:
                hnova = canviarh(h, -1, hamin, hamax)
        elif method == "newtonsim":
            hnova = h*told**((1/ito_new) - (1/it))
            if abs(hnova) > hamax:
                hnova = np.sign(h)*hamax
    return hnova

def canviarh(h, aug, hamin, hamax):
    fac = 2.0
    hnova = h
    if aug > 0:
        hnova = h*fac
        if abs(hnova) > hamax:
            hnova = np.sign(h)*hamax        
    elif aug < 0:
        hnova = h/fac
        if abs(hnova) < hamin:
            hnova = np.sign(h)*hamin           
    return hnova

def arcstep(f, x0, b0, t0, dt, method):
    tolf, told = 1e-11, 1e-6
    if method == "hybr":
        x1, info, status, txt = optimize.fsolve(fhiperpla, x0, full_output = 1, args = (b0, dt, f, x0))
        it, conv = info['nfev'], LA.norm(f(x1), 1) < tolf
    elif method == "newtonsim":
        x1, it = newtonsim(fhiperpla, x0, told, args = (b0, dt, f, x0))
        conv = LA.norm(f(x1), 1) < tolf
    else:
        print(method + " is not a valid solver")
        exit(-1)
    b1 = calcbeta(f, x1, b0)
    return x1, b1, (it, conv, told)

def arcpy(f, g, x0, s, t0, tf, action, method = "hybr", piter = False):
    xi, ti = x0.copy(), t0
    h, ha = 1e-3, 0
    bi = [s] + [0.0]*(len(x0) - 1)
    bi = calcbeta(f, xi, bi)
    notf, sicv = True, True
    xa, ba, ga, ta = xi.copy(), bi.copy(), g(xi), ti
    decreixg = False
    it = 0
    while notf and sicv:
        xo, bo, info = arcstep(f, xi, bi, ti, h, method)
        it += info[0] + 2*len(x0)
        if method == "newtonsim":
            it += 2*len(x0)
        ha = h
        h = nexth(h, (info[0], info[1], info[2], method))        
        if info[1]:
            xa, ba, ga, ta = xi.copy(), bi.copy(), g(xi), ti
            xi, bi = xo.copy(), bo.copy()
            g0 = g(xi)
            ti = ti + h
            if action == 1:
                if not decreixg and g0 < ga:
                    decreixg = True
                elif decreixg and g0 > ga:
                    return xa, {'tf': ta, 'success': True, 'iter': it}
            if action == 2 and np.sign(g0) != np.sign(ga):
                xcan = xa - ga*(xi - xa)/(g0 - ga)
                F = lambda x: [f(x)[i] for i in range(len(x0) - 1)] + [g(x)]
                xres, info, status, txt = optimize.fsolve(F, xcan, full_output = 1)
                it += info["nfev"]
                return xres, {'tf': ta, 'success': LA.norm(info["fvec"], 1) < 1e-10, 'iter': it}
        notf, sicv = ti < tf, info[1] or h != ha
        if piter:
            print(ti, xi[:3], LA.norm(f(xi), 1), g(xi), (info[0], info[1]))
    if action == 1 and not decreixg :
        return x0, {'tf': t0, 'success': False, 'iter': it}
    return xi, {'tf': ti, 'success': action == 0, 'iter': it}
