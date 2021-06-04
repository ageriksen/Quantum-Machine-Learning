# Quantum-Machine-Learning
Repo for project 2, Comp. phys II 2021

### Report in report/FYS4411_Project_2.pdf

### Code in src/qml.py
There are no additional call arguments to run the code.
The models are chosen through string names corresponding
to implemented functions in the class. The class 
initialization is at the bottom and the different models,
learning rates and shots can be chosen there before running
the script. 

### Run results in src/data/
the results after runs are stored in files with names corresponding
to the run. The structure of the resuts are made as
1. \# A meta line with a pound to comment
2. The model theta values
3. The loss averaged over samples per epoch
4. the accuracy of prediction per epoch

### Plot of results in src/plots/
The plots are named according to the runs and titled regarding 
cross-entropy or accuracy. There is no direct differentiation within 
a plot of the number of shots in the model, but this is in the name.
