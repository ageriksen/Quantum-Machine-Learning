#!/usr/bin/env python3
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
x = iris.data
y = iris.target
y1 = y

print("*"*100)
print(x.shape)
print(y1.shape)
print(y1)

idx = np.where(y < 2) # we only take the first two targets.
print("*"*100)
print(idx)

x = x[idx,:]
y = y[idx]

print("*"*100)
print(x.shape)
print(y.shape)
print(y)



import matplotlib.pyplot as plt
import seaborn as sns
iris = sns.load_dataset("iris")

ratio = iris["sepal_length"]/iris["sepal_width"]

for name, group in iris.groupby("species"):
    plt.scatter(group.index, ratio[group.index], label=name)

plt.legend()
plt.grid()
plt.show()
