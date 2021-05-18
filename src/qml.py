import numpy as np
import pandas as pd
import qiskit as qk
import matplotlib.pyplot as plt
from sklearn import datasets

class QML:
    def __init__(self, n_quantum, n_classic, seed, ansatz="basic_ansatz", encoder="basic_encoder", lossfunc="BCE", delLoss="BCEderivative"):
        self.n_quantum = n_quantum
        self.n_classic = n_classic

        self.seed = seed

        self.ansatzes = {"basic_ansatz":self.basicAnsatz}
        self.encoders = {"basic_encoder":self.basicEncoder}
        self.ansatz = self.ansatzes[ansatz]
        self.encoder = self.encoders[encoder]

        self.loss = {"BCE":self.BCE}
        self.delLoss = {"BCEderivative": self.BCEderivative}
        self.lossFunction = self.loss[lossfunc]
        self.lossDerivative = self.delLoss[delLoss]

    def setModel(self, feature_vector, target, n_model_parameters):
        """
        defines the features, targets and model parameters to be input into the
        model before compiling to circuits and jobs.
        """
        self.feature_vector = feature_vector
        self.target = target
        self.n_model_parameters = n_model_parameters

    def setAnsatz(self, ansatz):
        try:
            self.ansatz = self.ansatzes[ansatz]
        except:
            print(f"ansatz '{ansatz}' not implemented")

    def setEncoder(self, encoder):
        try:
            self.encoder = self.encoders[encoder]
        except:
            print(f"encoder '{encoder}' not implemented")

    #====================================================================================================
    #   === encoders & ansatzes ===
    def basicEncoder(self):
#        for i, feature in enumerate(self.feature_vector):
#            self.circuit.ry(feature, self.quantum_register[i])
        for i, feature in enumerate(self.feature_vector):
            self.circuit.ry(2*np.pi*feature, self.quantum_register[i])


    def basicAnsatz(self):
        for i in range(len(self.feature_vector)):
            self.circuit.rx(self.theta[i], self.quantum_register[i])

        for qubit in range(self.n_quantum - 1):
            self.circuit.cx(self.quantum_register[qubit], self.quantum_register[qubit + 1])

        self.circuit.ry(self.theta[-1], self.quantum_register[-1])
        self.circuit.measure(self.quantum_register[-1], self.classical_register)

    #====================================================================================================

    def model(self, backend='qasm_simulator', shots=1000):
        """
        Set up and run the model with the predefined encoders and ansatzes for the circuit. 
        """

        self.quantum_register = qk.QuantumRegister(self.n_quantum)
        self.classical_register = qk.ClassicalRegister(self.n_classic)
        self.circuit = qk.QuantumCircuit(self.quantum_register, self.classical_register)

        self.theta = 2*np.pi*np.random.randn(self.n_model_parameters)

        self.encoder()
        self.ansatz()

        job = qk.execute(self.circuit,
                        backend=qk.Aer.get_backend(backend),
                        shots=shots,
                        seed_simulator=self.seed
                        )
        results = job.result().get_counts(self.circuit)
        self.model_prediction = results['0'] / shots
        return self.model_prediction

    def BCE(self,out, target):
        #TODO figure out what N stands for. And where the loop fits in. 
        N = 1. # I don't know what the N is supposed to be. 

        return (-1./N)*( target*np.log10(out) + (1-target)*np.log(1-out) )

    def BCEderivative(self, out, target):
        #TODO figure out what N stands for. And where the loop fits in. 
        N = 1. # Still don't know what N is.
        dydtheta = 1. # don't know where to put this. 
        return (-1./N) * ( (target/float(out)) - ((1-target)/(1-out)) )


    def train(self, target, epochs=100, learning_rate=0.1, debug=False):
        """
        Uses the initial quess for an ansatz for the model to train and optimise the model ansatz for
        the given cost/loss function. 
        """

        for epoch in range(epochs):

            out = self.model()

            #mean_squared_error = (out - target)**2
            #mse_derivative = 2 * (out - target)
            loss = self.lossFunction(out, target)
            lossDerivative = self.lossDerivative(out, target)

            theta_gradient = np.zeros_like(self.theta)

            for i in range(self.n_model_parameters):

                self.theta[i] += np.pi / 2
                out_1 = self.model()

                self.theta[i] -= np.pi
                out_2 = self.model()

                self.theta[i] += np.pi / 2
                theta_gradient[i] = (out_1 - out_2) / 2

                if debug:
                    print(f'output 1: {out_1}')
                    print(f'output 2: {out_2}')


            #print(self.circuit)
            #self.theta = self.theta - learning_rate * theta_gradient * mse_derivative
            self.theta = self.theta - learning_rate * theta_gradient * lossDerivative
            #print(mean_squared_error)
            print(loss)


if __name__ == "__main__":
    seed = 2021
    np.random.seed(seed)

    dic_data = datasets.load_iris(as_frame=True)
    df = dic_data['frame']
    features = dic_data['data']
    targets = dic_data['target']
    species = dic_data['target_names']

    index = np.where(targets < 2 )

    df = df.iloc[index]
    features = features.iloc[index]
    targets = targets.iloc[index]

    n_model_parameters = features.columns.shape[0] + 1

    n_quantum = features.columns.shape[0]
    n_classic = 1

    qml = QML(n_quantum, n_classic, seed)
    #qml.setModel(feature_vector,target, n_model_parameters)
    qml.setModel(features.iloc[0],targets[0], n_model_parameters)
    qml.model()
    print(qml.circuit)
    qml.train(targets[0], epochs=10)
