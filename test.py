#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# 15-12-2023
# alex
# test.py

from arcpy import *
import numpy as np
import time as tm

def f(x, *val):
    return [x[0]**2+x[1]**2+x[2]**2 - val[0]**2, x[0]+x[1]-val[1]*x[2]]

def g(x):
    return ((x[0] - 0.5)**2 - 0.5)

if __name__ == "__main__":
    tmp0 = tm.time()
    x0 = [-2*np.sqrt(2/3), np.sqrt(2/3), -np.sqrt(2/3)]
    p = (2.0, 1.0)
    b0 = [1.0, 0.0, 0.0]
    print(f(x0, *p))
    x1, info = arcpy(f, g, x0, p, b0, 0.0, 100.0, action = 0, method = "newtonsim")
    print("t =", tm.time() - tmp0)
    print(x1)
    print(info)
