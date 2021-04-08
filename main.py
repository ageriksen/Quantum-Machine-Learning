#!/usr/bin/env python3

from sklearn.datasets import load_breast_cancer
import qiskit as qk
import numpy as np

np.random.seed(42)

data = load_breast_cancer()
#print(data)

x = data.data # features
y = data.target # targets
h = data.feature_names

p = 2 # nr. features. 

data_register = qk.QuantumRegister(p)
classical_register = qk.ClassicalRegister(1)

circuit = qk.QuantumCircuit(data_register, classical_register)

sample = np.random.uniform(size=p)
target = np.random.uniform(size=1)

for feature_idx in range(p):
    circuit.ry( 2*np.pi*sample[feature_idx], data_register[feature_idx])

print(circuit)
