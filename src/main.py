#!/usr/bin/env python3

from sklearn.datasets import load_breast_cancer
import qiskit as qk
import numpy as np

np.random.seed(42)

#data = load_breast_cancer()
##print(data)
#
#h = data.feature_names # feature names
#x = data.data # features
#y = data.target # targets

#print("Feature names, numbered {0}: ".format(h.shape))
#print(h)
#print("Features, numbered {0}: ".format(x.shape))
#print(x)
#print("Targest, numbered {0}: ".format(y.shape))
#print(y)

#====================================================================================================
#   a) Encoding Data Into a Quantum State
#Consider a simple way to encode randomly generated data set sample into a quantum state.
#Sample code shows random set of p=2 features encoded into quantum state on 2 qubits.
#Each feature on a respective qubit rotated on bloch sphere by an R_y(ϴ) gate. Features scaled with 
#2π to represent radians. Classical register used later to store measured value of circuit.

#Main task, get familiar with functionality & implement code for first p values of breast cancer data.
#====================================================================================================
#p = int(len(h)*.3) # pick out first p features from x's 1st target
#sample = x[0,0:p]
#target = y[0] #x's first target
#print("*"*100, "\n sample size: {0}\n".format(sample.shape), "*"*100)

p = 2 # nr. features. 
sample = np.random.uniform(size=p)
target = np.random.uniform(size=1)

data_register = qk.QuantumRegister(p)
classical_register = qk.ClassicalRegister(1)

circuit = qk.QuantumCircuit(data_register, classical_register)

for feature_idx in range(p):
    circuit.ry( 2*np.pi*sample[feature_idx], data_register[feature_idx])

print(circuit)

#====================================================================================================
#   b) Processing Encoded Data With Parameterized Gates
#After encoding a quantum state with the data, the circuit needs to be extended with operations that
#process the state in a way that allows infering target data. Can be done by quantum gates dependent on 
#learnable parameters ϴ. This is the 'anzats'

#Familiarize again with functionality and create own ansatz for p first breast cancer features. number of 
#learnable ϴ should be arbitrary. 
#====================================================================================================

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

#====================================================================================================
#   c) Measuring Quantum state & Making Inference
#Next, generate prediction from quantum ML model. Perform measurement on the quantum state. A 
#measurement operation on the final qubit in the circuit. Interpret prediction as this qubit being in
#the |1> state. 

#Implement own function for prediction by measuring a qubit.
#====================================================================================================

circuit.measure(data_register[-1], classical_register[0])
shots = 100

job = qk.execute(circuit,
        backend = qk.Aer.get_backend('qasm_simulator'),
        shots=shots,
        seed_simulations=42
        )
results = job.result()
results = results.get_counts(circuit)

prediction = 0
for key, value in results.items():
    if key == '1':
        prediction += value
prediction/=shots
print("Prediction: ", prediction, "| Target: ", target[0])
#print("Prediction: {} | Target: {}".format(prediction, target[0]))


#====================================================================================================
#   d) Putting it all together
#Put together the above steps. Ideally through a class/function which, given feature matrix w/ n 
#samples and arbitrary nr model parameters returns vector of n outputs. example input:
#====================================================================================================

n = 100 # nr samples
p = 10 # nr features

theta = np.random.uniform(size = 20) # array of model parameters

X = np.random.uniform(size=(n,p)) # design matrix

#'model' is the class/function for the circuit creation and implementation. 
#y_pred = model(X, theta) #prediction, shape (n)

