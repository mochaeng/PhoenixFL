import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Dict
import numpy as np
from collections import OrderedDict
import os

from ..neural_helper.mlp import get_train_and_test_loaders
from ..pre_process import (
    get_standardized_train_test_data,
    CLIENTS_PATH,
    get_train_and_test_dfs,
    get_prepared_data_for_loader,
)

NUM_CLIENTS = 4
NUM_ROUNDS = 3

PATH_TO_METRICS_FOLDER = "./prototype_1/federated/metrics"
METRICS_FILE_PATH = os.path.join(PATH_TO_METRICS_FOLDER, "metrics.json")
WEIGHTED_METRICS_FILE_PATH = os.path.join(PATH_TO_METRICS_FOLDER, "weighted.json")


class FederatedMetricsRecord:
    def __init__(self) -> None:
        self.__metrics = {}

    def __repr__(self) -> str:
        return str(self.__metrics)

    def add(self, server_round, name, values):
        round_key = f"round_{server_round}"
        if round_key not in self.__metrics:
            self.__metrics[round_key] = {}
        self.__metrics[round_key][name] = values

    def add_weighted_values(self, values):
        self.__metrics["weighted"] = values

    def get(self):
        return self.__metrics


class AggregatedFederatedMetricsRecorder:
    def __init__(
        self,
        num_models: int,
        num_rounds: int,
        client_names: List[str],
        metrics_names: List[str],
    ) -> None:
        self.client_names = client_names
        self.metrics_names = metrics_names
        self.models = list(range(1, num_models + 1))
        self.rounds = list(range(1, num_rounds + 1))
        self._metrics = {
            f"model_{num_model}": {
                f"round_{num_round}": {
                    client_name: {metric_name: [] for metric_name in self.metrics_names}
                    for client_name in self.client_names
                }
                for num_round in self.rounds
            }
            for num_model in self.models
        }

    def add(self, num_model, num_round, name, values):
        self._metrics[num_model][num_round][name] = values

    def get(self):
        return self._metrics


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_all_federated_loaders(
    batch_size,
) -> Dict[int, Tuple[Tuple, Tuple[DataLoader, DataLoader]]]:
    def get_all_federated_client_data():
        all_data: Dict[int, Tuple[str, Dict]] = {}
        for idx, (dataset_name, path) in enumerate(CLIENTS_PATH):
            train_df, test_df = get_train_and_test_dfs(path)
            (train_data, test_data), _ = get_standardized_train_test_data(
                train_df, test_df
            )
            data = get_prepared_data_for_loader(
                train_data=train_data, test_data=test_data
            )
            all_data[idx] = (dataset_name, data)
        return all_data

    all_data = get_all_federated_client_data()

    loaders: Dict[int, Tuple[Tuple, Tuple[DataLoader, DataLoader]]] = {}
    for cid, data in all_data.items():
        dataset_name, data = data
        train_loader, test_loader = get_train_and_test_loaders(data, batch_size)
        loaders[cid] = ((cid, dataset_name), (train_loader, test_loader))

    return loaders
