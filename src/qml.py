
class QML:
    def __init__(self,\
            n_quantum, n_classic, featureMatrix, \
            n_model_parameters, seed, \
            ansatz="basic_ansatz", encoder="basic_encoder", lossfunc="BCE", delLoss="BCEderivative"):

        self.n_quantum = n_quantum
        self.n_classic = n_classic
        self.featureMatrix = featureMatrix
        self.n_samples = featureMatrix.shape[0]
        self.n_features = featureMatrix.shape[1]
        #self.feature_vector = self.featureMatrix.iloc[0]
        self.feature_vector = self.featureMatrix[0,:]

        self.n_model_parameters = n_model_parameters
        self.seed = seed

        self.ansatzes = {"basic_ansatz":self.basicAnsatz, "doubleAnsatz":self.doubleAnsatz}
        self.encoders = {"basic_encoder":self.basicEncoder}
        self.ansatz = self.ansatzes[ansatz]
        self.encoder = self.encoders[encoder]

        self.loss = {"BCE":self.BCE}
        self.delLoss = {"BCEderivative": self.BCEderivative}
        self.lossFunction = self.loss[lossfunc]
        self.lossDerivative = self.delLoss[delLoss]

        self.theta = 2*np.pi*np.random.uniform(0,1, size=(self.n_model_parameters))
        print(self.theta.shape)

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
        """
        scaling with pi to avoid mapping 0 and 1 to the same rotation.
        """
        for i, feature in enumerate(self.feature_vector):
            #self.circuit.ry(np.pi*feature, self.quantum_register[i])
            self.circuit.rx(np.pi*feature, self.quantum_register[i])


    def basicAnsatz(self):
        for i in range(len(self.feature_vector)):
            self.circuit.ry(self.theta[i], self.quantum_register[i])

        for qubit in range(self.n_quantum - 1):
            self.circuit.cx(self.quantum_register[qubit], self.quantum_register[qubit + 1])

        self.circuit.ry(self.theta[-1], self.quantum_register[-1])
        self.circuit.measure(self.quantum_register[-1], self.classical_register)

    def doubleAnsatz(self):

        for i in range(len(self.feature_vector)):
            self.circuit.ry(self.theta[i], self.quantum_register[i])

        for qubit in range(self.n_quantum - 1):
            self.circuit.cx(self.quantum_register[qubit], self.quantum_register[qubit + 1])

        for i in range(len(self.feature_vector)):
            self.circuit.ry(self.theta[self.n_quantum + i], self.quantum_register[i])

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
        return -( target*np.log10(out) + (1-target)*np.log(1-out) )

    def BCEderivative(self, out, target):
        return -( (target/float(out)) - ((1-target)/(1-out)) )


    def train(self, target, epochs=100, learning_rate=.1, debug=False):
        """
        Uses the initial quess for an ansatz for the model to train and optimise the model ansatz for
        the given cost/loss function. 
        """
        from tqdm import tqdm

        for epoch in range(epochs):

            thetaShift = np.zeros([self.n_samples,len(self.theta)])
            loss = np.ones(self.n_samples)
            lossDerivative = np.zeros_like(loss)

            for sample in tqdm(range(self.n_samples)):

                #self.feature_vector = self.featureMatrix.iloc[sample]
                self.feature_vector = self.featureMatrix[sample, :]

                out = self.model()

                loss[sample] = self.lossFunction(out, target[sample])
                lossDerivative[sample] = self.lossDerivative(out, target[sample])

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

                    #thetaShift[sample, i] = - learning_rate * theta_gradient * np.mean(lossDerivative) # theta gradient for vairable shift.
                    thetaShift[sample, i] = theta_gradient[i]

            print(np.mean(loss))


            #self.theta = self.theta + np.average(thetaShift, axis=0)
            self.theta -= learning_rate * np.mean((thetaShift *  lossDerivative.reshape(-1,1)), axis=0)
        #return self.theta, loss


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import qiskit as qk
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.preprocessing import MinMaxScaler

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

    n_model_parameters = 2*features.columns.shape[0] + 1

    n_quantum = features.columns.shape[0]
    n_classic = 1

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)


    qml = QML(n_quantum, n_classic, features, n_model_parameters, seed, ansatz="doubleAnsatz")

    #printing circuit layout
    qml.model()
    print(qml.circuit)

    qml.train(targets, epochs=10)
