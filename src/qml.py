import numpy as np
import pandas as pd
import qiskit as qk
import matplotlib.pyplot as plt
from sklearn import datasets

class QML:
    def __init__(self, n_quantum, n_classic, seed):
        self.n_quantum = n_quantum
        self.n_classic = n_classic
        self.seed = seed
        self.makeCircuit(n_quantum,n_classic)

    def makeCircuit(self, n_quantum, n_classic):
        self.quantum_register = qk.QuantumRegister(n_quantum)
        self.classical_register = qk.ClassicalRegister(n_classic)
        self.circuit = qk.QuantumCircuit(self.quantum_register, self.classical_register)



    def encoder(self, feature_vector):
        self.feature_vector = feature_vector
        for i, feature in enumerate(feature_vector):
            self.circuit.ry(feature, self.quantum_register[i])



    def ansatz(self, n_model_parameters):
        self.n_model_parameters = n_model_parameters
        self.theta = np.random.randn(self.n_model_parameters)

        for i in range(len(self.feature_vector)):
            self.circuit.rx(self.theta[i], self.quantum_register[i])

        for qubit in range(self.n_quantum - 1):
            self.circuit.cx(self.quantum_register[qubit], self.quantum_register[qubit - 1])

        self.circuit.ry(self.theta[-1], self.quantum_register[-1])
        self.circuit.measure(self.quantum_register[-1], self.classical_register)



    def update_ansatz(self):
        for i in range(len(self.feature_vector)):
            self.circuit.rx(self.theta[i], self.quantum_register[i])


        for qubit in range(self.n_quantum - 1):
            self.circuit.cx(self.quantum_register[qubit], self.quantum_register[qubit - 1])

        self.circuit.ry(self.theta[-1], self.quantum_register[-1])
        self.circuit.measure(self.quantum_register[-1], self.classical_register)



    def run(self, backend='qasm_simulator', shots=1000):
        """
        runs the circuit for the specified number of shots.
        """

        job = qk.execute(self.circuit,
                        backend=qk.Aer.get_backend(backend),
                        shots=shots,
                        seed_simulator=self.seed
                        )
        results = job.result().get_counts(self.circuit)
        self.model_prediction = results['0'] / shots



    def train(self, target, epochs=100, learning_rate=0.1, debug=False):

        for epoch in range(epochs):
            self.run()
            mean_squared_error = (self.model_prediction - target)**2
            mse_derivative = 2 * (self.model_prediction - target)
            theta_gradient = np.zeros_like(self.theta)

            for i in range(self.n_model_parameters):

                """
                We are being stupid here, we need to update the thetas in the
                actual ansatz as well, because now we just run with the same values for the gates in the circuit.
                Trying to find some way to adjust one of the gate values without remaking the whole circuit
                """
                print("*"*20, "\n",self.circuit.depth(),"\n","*"*20)
                """
                We need to reset the circuit every time, now we are just adding to the end of the existing circuit
                this is prob because we're using self.circuit stuff
                """


                self.theta[i] += np.pi / 2
                self.makeCircuit(self.n_quantum,self.n_classic)
                self.update_ansatz()
                self.run()
                out_1 = self.model_prediction

                self.theta[i] -= np.pi
                self.update_ansatz()
                self.run()
                out_2 = self.model_prediction

                self.theta[i] += np.pi / 2
                theta_gradient[i] = (out_1 - out_2) / 2

                if debug:
                    print(f'output 1: {out_1}')
                    print(f'output 2: {out_2}')


            print(self.circuit)
            self.theta = self.theta - learning_rate * theta_gradient * mse_derivative
            print(mean_squared_error)



seed = 2021
np.random.seed(seed)

qml = QML(4, 1, seed)
qml.encoder([1.0, 1.5, 2.0, 0.3])
qml.ansatz(5)
print(qml.circuit)
qml.train(0.7, epochs=10)
