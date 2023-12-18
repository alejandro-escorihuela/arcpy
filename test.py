#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# 15-12-2023
# alex
# test.py

from arcpy import *
import numpy as np
import time as tm

def f(x):
    # return [x[0]**2+x[1]**2 - 4]
    return [x[0]**2+x[1]**2+x[2]**2 - 4, x[0]+x[1]-x[2]]

def g(x):
    return ((x[0] - 0.5)**2 - 0.5)

if __name__ == "__main__":
    ## x0 = [-np.sqrt(2), np.sqrt(2)]
    tmp0 = tm.time()
    x0 = [-2*np.sqrt(2/3), np.sqrt(2/3), -np.sqrt(2/3)]
    b0 = [1.0, 0.0, 0.0]
    x1, info = arcpy(f, g, x0, b0, 0.0, 10.0, action = 0, method = "hybr")
    print("t =", tm.time() - tmp0)
    print(x1)
    print(info)
