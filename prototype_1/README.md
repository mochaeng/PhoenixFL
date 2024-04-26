# Prototype 1

Here is the code used for experiments comparing federated learning with the centralized approach. The goal of this prototype is to demonstrate the feasibility of federated learning in training a more robust model for network packet classification in heterogeneous networks, while preserving user data privacy.

* `centralized/`: Contains code for training a **multilayer perceptron (MLP)** using the centralized dataset. The final model is evaluated on the test set of each of the four clients. 

* `federated/`: Contains code for training an MLP using the federated learning (FL) approach. The trained model is also evaluated at each client using its own test set.

* `neural_helper/`: Contains code for the MLP implementation.


- `pre_process.py`: Contains general-purpose code for reading datasets and data standardization.
