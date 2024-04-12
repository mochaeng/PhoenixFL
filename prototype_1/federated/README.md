# Federated Learning

## Overview

The goal is to evaluate federated learning (FL) technique to train a more robust model for heterogeneous networks.

> The file [simulation.py](simulation.py) contains the code for simulating the training of a machine learning (ML) model using FL. At each round, a federated evaluation is conducted amoung the clients. The metrics evaluated for a given client in a particular round are saved to a txt file called _temp_metrics.txt_.

> The file [aggregating_metrics.py](aggregating_metrics.py) is responsible for reading the _temp_metrics.txt_ file and aggregated the metrics in a more readable format (e.g. json). That file is _metrics.json_. 

## Running

To simulate the training of 2 models with 3 rounds each, run:

```sh
bash fl_pipeline.sh -m 2 -r 3
```
