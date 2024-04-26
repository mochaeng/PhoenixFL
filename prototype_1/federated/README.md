# Federated Learning

## Overview

This part deals with the process of training a **Multilayer Perceptron (MLP)** using the federated learning approach. A simulation is conducted using the Flower simulation module. In this approach, each client trains a local model on its own data. The server then collects these models and aggregates them into a single global model. This global model is then used for federated evaluation on each client's test set.

The code supports two aggregation algorithms: the classic `FedAvg` and `FedProx`.

## Simulation

The [`main.py`](main.py) script loads data from each client to conduct the simulation process. In each round, clients train a local model for `E` epochs. The server then aggregates the models using a chosen strategy (e.g., `FedAvg` or `FedProx`). Finally, a federated evaluation is performed at the end of the round.

> Run from the project root directory !

The program can be executed with various options to control the federated learning simulation:

- To train a model with 10 rounds of federated learning:

    ```sh
    python -m prototype_1.federated.main --num-rounds 10
    ```
- To save the federated evaluation metrics from each client in a `JSON` file:

    ```sh
    python -m prototype_1.federated.main --num-rounds 10 --save-results true
    ```
- To use the `FedProx` strategy for aggregation:

    ```sh
    python -m prototype_1.federated.main --algo fedprox --mu 1.0
    ```

<!-- ## Running

To simulate the training of 2 models with 3 rounds each, run:

```sh
bash fl_pipeline.sh -m 2 -r 3
``` -->
