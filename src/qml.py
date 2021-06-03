
class QML:
    def __init__(self,\
            n_quantum, n_classic, featureMatrix, \
            n_model_parameters, seed, shots=1000, \
            backend='qasm_simulator',\
            model="basicModel", lossfunc="BCE", delLoss="BCEderivative"):# ansatz="basic_ansatz", encoder="basic_encoder",

        #Setting class variables:
        self.n_model_parameters = n_model_parameters
        self.n_quantum = n_quantum
        self.n_classic = n_classic
        self.backend = backend
        self.shots = shots
        self.seed = seed

        #Setting data variables:
        self.featureMatrix = featureMatrix
        self.n_samples = featureMatrix.shape[0]
        self.n_features = featureMatrix.shape[1]
        self.feature_vector = self.featureMatrix[0,:]

        #Defining model version:
        self.models = {"basicModel":self.basicModel, \
                "doubleAnsatz": self.doubleAnsatz, \
                "doubleEncoding": self.doubleEncoding,\
                "doubleAnsatzdoubleEncoding": self.doubleAnsatzdoubleEncoding}
        self.modelName = model
        self.model = self.models[self.modelName]


        #Defining loss function:
        self.loss = {"BCE":self.BCE}
        self.delLoss = {"BCEderivative": self.BCEderivative}
        self.lossFunction = self.loss[lossfunc]
        self.lossDerivative = self.delLoss[delLoss]

        self.theta = 2*np.pi*np.random.uniform(0,1, size=(self.n_model_parameters))

    #====================================================================================================
    #   === encoders & ansatzes ===

    def encoder(self):
        """
        mapping features scaled to 1 onto bloch sphere.
        scaling with pi to avoid mapping 0 and 1 to the same rotation.
        """
        for i, feature in enumerate(self.feature_vector):
            self.circuit.rx(np.pi*feature, i)

        for qubit in range(self.n_quantum - 1):
            self.circuit.cx(qubit, qubit + 1)

    def ansatz(self, iteration=0):
        """
        Rotating qubit states by parameters theta around bloch sphere
        to adjust model for prediction of encoded features
        """
        for i in range(self.n_quantum):
            self.circuit.ry(self.theta[iteration*self.n_quantum +i], i)

        for qubit in range(self.n_quantum - 1):
            self.circuit.cx(qubit, qubit + 1)

    def measure(self):
        """
        measuring the final qubit after applying the final model 
        parameter as a bias
        """
        self.circuit.ry(self.theta[-1], -1)
        self.circuit.measure(-1, 0)


    def basicModel(self):
        self.encoder()
        self.ansatz()
        self.measure()

    def doubleAnsatz(self):
        self.encoder()
        self.ansatz()
        self.ansatz(iteration=1)
        self.measure()

    def doubleEncoding(self):
        self.encoder()
        self.ansatz()
        self.encoder()
        self.measure()

    def doubleAnsatzdoubleEncoding(self):
        self.encoder()
        self.ansatz(iteration=0)
        self.encoder()
        self.ansatz(iteration=1)
        self.measure()

    #====================================================================================================
    #circuit
    def modelCircuit(self, printC=False):#, backend='qasm_simulator', shots=1000):
        """
        Set up and run the model with the predefined encoders and ansatzes for the circuit.
        """

        self.circuit = qk.QuantumCircuit(self.n_quantum, self.n_classic)

        self.model()

        if printC:
            print(self.circuit)

        job = qk.execute(self.circuit,
                        backend=qk.Aer.get_backend(self.backend),
                        shots=self.shots,
                        seed_simulator=self.seed
                        )

        results = job.result().get_counts(self.circuit)

        counts = 0
        for key, value in results.items():
            if key=='1':
                counts += value

        self.model_prediction = counts / float(self.shots)
        return self.model_prediction

    #====================================================================================================
    #loss
    def BCE(self,out, target):
        #fixing overflow errors in log10
        eps = 1e-8
        return -( target*np.log(out + eps) + (1-target)*np.log(1-out + eps))

    def BCEderivative(self, out, target):
        #fixing divide by zero error in division
        eps = 1e-8
        return -( (target/float(out + eps)) - ((1-target)/(1-out + eps)) )
    #====================================================================================================


    def train(self, target, epochs=100, learning_rate=.1, debug=False):
        """
        Uses the initial quess for an ansatz for the model to train and optimise the model ansatz for
        the given cost/loss function.
        """
        from tqdm import tqdm

        mean_loss = np.zeros(epochs)
        accuracy = np.zeros_like(mean_loss)

        for epoch in range(epochs):

            #   setup of storage arrays
            thetaShift = np.zeros([self.n_samples,len(self.theta)])
            loss = np.ones(self.n_samples)
            lossDerivative = np.zeros_like(loss)
            acc = 0

            for sample in tqdm(range(self.n_samples)):

                #self.feature_vector = self.featureMatrix.iloc[sample]
                self.feature_vector = self.featureMatrix[sample, :]

                out = self.modelCircuit()
                acc += np.round(out)==target[sample]


                loss[sample] = self.lossFunction(out, target[sample])
                lossDerivative[sample] = self.lossDerivative(out, target[sample])

                theta_gradient = np.zeros_like(self.theta)

                for i in range(self.n_model_parameters):

                    self.theta[i] += np.pi / 2
                    out_1 = self.modelCircuit()

                    self.theta[i] -= np.pi
                    out_2 = self.modelCircuit()

                    self.theta[i] += np.pi / 2
                    theta_gradient[i] = (out_1 - out_2) / 2

                    if debug:
                        print(f'output 1: {out_1}')
                        print(f'output 2: {out_2}')

                    #thetaShift[sample, i] = - learning_rate * theta_gradient * np.mean(lossDerivative) # theta gradient for vairable shift.
                    thetaShift[sample, i] = theta_gradient[i]

            accuracy[epoch] = float(acc)/self.n_samples
            mean_loss[epoch] = np.mean(loss)

            self.theta -= learning_rate * np.mean((thetaShift *  lossDerivative.reshape(-1,1)), axis=0)

            print("mean loss per epoch: ", mean_loss[epoch])
            print("accuracy per epoch: ", accuracy[epoch])
        return self.theta, mean_loss, accuracy


if __name__ == "__main__":
    #imports==================================================
    import numpy as np
    import pandas as pd
    import qiskit as qk
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.preprocessing import MinMaxScaler
    from qiskit.visualization import circuit_drawer
    import sys

    #seed=======================================================
    seed = 42
    np.random.seed(seed)

    #import and preprocess data=================================
    dic_data = datasets.load_iris(as_frame=True)
    df = dic_data['frame']
    features = dic_data['data']
    targets = dic_data['target']
    species = dic_data['target_names']

    index = np.where(targets < 2 )

    df = df.iloc[index]
    features = features.iloc[index]
    targets = targets.iloc[index]

    n_model_parameters = 2*features.columns.shape[0] + 1 #most of these could be unused if not controlled.TODO?

    n_quantum = features.columns.shape[0]
    n_classic = 1

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    #model==================================================
    #modelName = "doubleAnsatz"

    #shots = 1000

    #learningRate = 0.1
    epochs = 10
    parameterList = [9]#[5, 9, 5, 5, 9]
    modelList = ["basicModel"]#["basicModel", "doubleAnsatz", "lessEntangled", "doubleEncoding", "doubleAnsatzdoubleEncoding"]
    shotList = [1000, 10000]
    learnList = [0.1, 0.5, 1]

    for i, modelName in enumerate(modelList):
        print(modelName)
        for nshots in shotList:
            for learn in learnList:
                qml = QML(n_quantum, n_classic, features, parameterList[i], seed, shots=nshots, model=modelName)#, ansatz="doubleAnsatz")

                qml.modelCircuit(printC=True)

                """(comment here to uncomment :P) to make circuit diagrams for article
                circuit_drawer(qml.circuit, output='mpl')
                plt.show()
                """

                model, loss, accuracy = qml.train(targets, epochs=epochs, learning_rate=learn)
                """
                store into file:
                name.dat [or something]
                #first line with meta data, nr. shots, learning rate, ...
                model theta values[..]
                mean loss per epoch[..]
                accuracy per epoch[..]
                """
                #filename = "model"+modelName+"_lrn"+str(learn)+"_shots"+str(nshots)+"_epochs"+str(epochs)
                #metaline = "model:" + modelName \
                #        + ", seed:" + str(seed) \
                #        + ", epochs:" + str(epochs) \
                #        + ", learningRate:" + str(learn) \
                #        + ", shots:" + str(nshots) \
                #        +" 1st:model, 2nd:loss, 3rd:accuracy"


                #if sys.platform == "linux":
                #    fs = open("data/"+filename+".dat", "w")

                #elif sys.platform == "win32":
                #    fs = open("data\\"+filename+".dat","w")

                #fs.write("#"+metaline); fs.write("\n")
                #fs.write(str(model)); fs.write("\n")
                #fs.write(str(loss)); fs.write("\n")
                #fs.write(str(accuracy));
                #fs.close()
