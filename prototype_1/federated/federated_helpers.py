import torch
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict
import numpy as np
from collections import OrderedDict
import os
from flwr.common import Scalar

from neural_helper.mlp import get_train_and_test_loaders, TRAIN_CONFIG
from pre_process.pre_process import (
    get_standardized_train_test_data,
    CLIENTS_PATH,
    get_train_and_test_dfs,
    get_prepared_data_for_loader,
)

TOTAL_NUMBER_OF_CLIENTS = 4
NUM_ROUNDS = 3

PATH_TO_METRICS_FOLDER = "federated/metrics"
METRICS_FILE_PATH = os.path.join(PATH_TO_METRICS_FOLDER, "metrics.json")
WEIGHTED_METRICS_FILE_PATH = os.path.join(PATH_TO_METRICS_FOLDER, "weighted.json")

MetricType = dict[str, list[list[float]]]


class FederatedMetrics:
    def __init__(self) -> None:
        self.__metrics: dict[str, MetricType] = {}
        self.has_new_model_started = False

    def add_client_evaluated_results(self, client_name: str, results: dict[str, float]):
        if client_name in self.__metrics:
            for metric_name, value in results.items():
                self.__metrics[client_name][metric_name][-1].append(value)

        else:
            self.__metrics[client_name] = {
                metric_name: [[value]] for metric_name, value in results.items()
            }

    def add_new_round(self):
        if len(self.__metrics.keys()) == 0:
            return ValueError("no client data")

        for client_name in self.__metrics.keys():
            for metric_name in self.__metrics[client_name]:
                self.__metrics[client_name][metric_name].append([])

    def add_weighteds_metrics(self, values):
        self.__metrics["weighted_metrics"] = values

    def get_metrics(self):
        return self.__metrics


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def fit_config(server_round: int) -> Dict[str, Scalar]:
    merge_config = TRAIN_CONFIG.copy()
    merge_config.update(
        {
            "server_round": server_round,
        }
    )
    return merge_config


def eval_config(server_round: int) -> Dict[str, Scalar]:
    config: Dict[str, Scalar] = {
        "server_round": server_round,
    }
    return config


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
