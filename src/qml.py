import numpy as np
import pandas as pd
import qiskit as qk
import matplotlib.pyplot as plt
from sklearn import datasets





def sklearn_iris_to_df(sklearn_dataset):
    """
    processes the dataset from sklearn into a pandas dataframe with an
    additional column with string definition of the three classes.

    Input: (sklearn dataset)
    Output: (Pandas DataFrame Object)
    """
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df



def look_at_data(dataframe):
    """
    Simple function to generate scatter plot from a pandas dataframe
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['red', 'green', 'blue']

    for i, s in enumerate(dataframe.species.unique()):
        dataframe[dataframe.species==s].plot.scatter(x = 'sepal length (cm)',
                                            y = 'sepal width (cm)',
                                            label = s,
                                            color = colors[i],
                                            ax = ax)
    plt.show()



class QML:
    def __init__(self, n_quantum, n_classic, seed):
        self.n_quantum = n_quantum
        self.n_classic = n_classic
        self.seed = seed
        self.quantum_register = qk.QuantumRegister(n_quantum)
        self.classical_register = qk.ClassicalRegister(n_classic)
        self.circuit = qk.QuantumCircuit(self.quantum_register, self.classical_register)



    def encoder(self, feature_vector):
        self.feature_vector = feature_vector
        for i, feature in enumerate(feature_vector):
            self.circuit.ry(feature, self.quantum_register[i])



    def ansatz(self, n_ansatz_parameters):
        self.n_ansatz_parameters = n_ansatz_parameters
        self.theta = np.random.randn(self.n_ansatz_parameters)

        for i in range(len(self.feature_vector)):
            self.circuit.rx(self.theta[i], self.quantum_register[i])

        for qubit in range(self.n_quantum - 1):
            self.circuit.cx(self.quantum_register[qubit], self.quantum_register[qubit - 1])



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



    def train(self, target, epochs=100, learning_rate=0.1):
        theta_gradient = np.zeros_like(self.theta)

        for epoch in range(epochs):
            self.run()
            mean_squared_error = (self.model_prediction - target)**2
            mse_derivative = 2 * (self.model_prediction - target)

            for i in range(self.n_ansatz_parameters):

                self.theta[i] += np.pi / 2
                self.run()
                out_1 = self.model_prediction

                self.theta[i] -= np.pi
                self.run()
                out_2 = self.model_prediction

                self.theta[i] += np.pi / 2
                theta_gradient[i] = (out_1 - out_2) / 2


            self.theta = self.theta - learning_rate * theta_gradient * mse_derivative
            print(mean_squared_error)





qml = QML(4, 1, 2021)
qml.encoder([1.0, 1.5, 2.0, 0.3])
qml.ansatz(5)
qml.train(0.7)