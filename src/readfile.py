#!/usr/bin/env python3

import numpy as np
import glob
import matplotlib.pyplot as plt

filenames = glob.glob('data/*')

#for filename in filenames:
#    with open(filename) as fs:
#        lines = fs.readlines()
#
#    print(lines)
#
#    break

#for filename in filenames:
#    lines = filename.read().split(' ')

from numpy import loadtxt

lines = loadtxt(filenames[0], comments="#", delimiter=' ')

"""
want to try variable 
    models = [basic/naive, double theta, double encoding]
        shots = [1k, 10k]
            learningrates = [0.1, 0.5, 1]
"""
