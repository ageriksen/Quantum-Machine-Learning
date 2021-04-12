#!/usr/bin/env python3

from sklearn.datasets import load_breast_cancer
import qiskit as qk
import numpy as np

np.random.seed(42)

data = load_breast_cancer()
#print(data)

h = data.feature_names # feature names
x = data.data # features
y = data.target # targets

print("Feature names, numbered {0}: ".format(h.shape))
print(h)
print("Features, numbered {0}: ".format(x.shape))
print(x)
print("Targest, numbered {0}: ".format(y.shape))
print(y)


#p = 2 # nr. features. 
p = int(len(h)*.3) # pick out first p features from x's 1st target
sample = x[0,0:p]
print("*"*100, "\n sample size: {0}\n".format(sample.shape), "*"*100)
target = y[0] #x's first target

data_register = qk.QuantumRegister(p)
classical_register = qk.ClassicalRegister(1)

circuit = qk.QuantumCircuit(data_register, classical_register)

for feature_idx in range(p):
    circuit.ry( 2*np.pi*sample[feature_idx], data_register[feature_idx])

print(circuit)


n_params = p
theta = 2*np.pi*np.random.uniform(size=n_params)

for i in range(n_params):
    circuit.rx(theta[i], data_register[i])


end = int((n_params+1)/2)
print("stop at i = {}".format(end))
for i in range(1, end):
    j = 2*i-2
    k = 2*i-1
    print(j, k)
    circuit.cx(data_register[j], data_register[k])

print(circuit)
