from flwr.common import Scalar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from typing import Tuple, List, Dict
from collections import OrderedDict

from neural.helpers import get_train_and_test_loaders, TRAIN_CONFIG, get_test_loader
from pre_process.pre_process import (
    get_standardized_train_test_data,
    CLIENTS_PATH,
    get_train_and_test_dfs,
    get_prepared_data_for_loader,
    DATASETS_PATHS,
    ScalerType,
    get_df,
    get_standardized_df,
)

TOTAL_NUMBER_OF_CLIENTS = 3
NUM_ROUNDS = 3

PATH_TO_METRICS_FOLDER = "federated/metrics"
METRICS_FILE_PATH = os.path.join(PATH_TO_METRICS_FOLDER, "metrics.json")
WEIGHTED_METRICS_FILE_PATH = os.path.join(PATH_TO_METRICS_FOLDER, "weighted.json")

FederatedLoadersFnReturnType = Dict[
    int, Tuple[Tuple[int, str, ScalerType], Tuple[DataLoader, DataLoader]]
]
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


def set_parameters(net: nn.Module, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    state_dict = OrderedDict()
    for k, v in params_dict:
        state_dict[k] = torch.tensor(v, dtype=net.state_dict()[k].dtype)
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


def get_all_federated_loaders(batch_size) -> FederatedLoadersFnReturnType:
    def get_all_federated_client_data():
        all_data: Dict[int, Tuple[str, Dict, ScalerType]] = {}
        for idx, (dataset_name, path) in enumerate(CLIENTS_PATH):
            train_df, test_df = get_train_and_test_dfs(path)
            (train_df_scaled, test_df_scaled), scaler = (
                get_standardized_train_test_data(train_df, test_df)
            )
            data = get_prepared_data_for_loader(
                train_df=train_df_scaled, test_df=test_df_scaled
            )
            all_data[idx] = (dataset_name, data, scaler)
        return all_data

    all_data = get_all_federated_client_data()

    loaders: FederatedLoadersFnReturnType = {}
    for cid, data in all_data.items():
        dataset_name, data, scaler = data
        train_loader, test_loader = get_train_and_test_loaders(data, batch_size)
        loaders[cid] = ((cid, dataset_name, scaler), (train_loader, test_loader))

    return loaders


def get_centralized_test_loader(batch_size, scaler: ScalerType):
    train_test_path = DATASETS_PATHS["CENTRALIZED"]["TEST"]
    test_df = get_df(train_test_path)

    test_df_scaled = get_standardized_df(test_df, scaler)
    test_data = get_prepared_data_for_loader(test_df=test_df_scaled)
    test_loader = get_test_loader(test_data, batch_size)

    return test_loader
