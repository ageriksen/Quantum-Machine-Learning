#!/usr/bin/env python3

import numpy as np

filename = "somestring"

with open("data/"+filename+".dat") as fs:
    lines = fs.readlines()

print(lines)

array = np.array(lines[1])
print(array)


"""
want to try variable 
    models = [basic/naive, double theta, double encoding]
        shots = [1k, 10k]
            learningrates = [0.1, 0.5, 1]
"""
