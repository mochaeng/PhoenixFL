import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Dict
import numpy as np
from collections import OrderedDict
import os

from ..pre_process import (
    COLUMN_TO_REMOVE,
    get_standardized_data_from_train_test_dataframes,
    CLIENTS_PATH,
    get_train_test_dataframes_and_drop_column,
)

NUM_CLIENTS = 4
NUM_ROUNDS = 3

PATH_TO_METRICS_FOLDER = "./prototype_1/federated/metrics"
# TEMP_METRICS_PATH = os.path.join(PATH_TO_METRICS_FOLDER, "temp_metrics.txt")
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
    batch_size=32,
) -> dict[int, Tuple[Tuple, Tuple[DataLoader, DataLoader]]]:
    def _get_all_federated_client_data():
        all_data: Dict[int, Tuple[str, Dict]] = {}
        for idx, (dataset_name, path) in enumerate(CLIENTS_PATH):
            train_df, test_df = get_train_test_dataframes_and_drop_column(
                path, COLUMN_TO_REMOVE
            )
            data_std, _ = get_standardized_data_from_train_test_dataframes(
                train_df, test_df
            )
            all_data[idx] = (dataset_name, data_std)
        return all_data

    all_data = _get_all_federated_client_data()

    loaders: dict[int, Tuple[Tuple, Tuple[DataLoader, DataLoader]]] = {}
    for cid, data in all_data.items():
        name, data_std = data

        x_train_tensor = torch.tensor(data_std["x_train"], dtype=torch.float32)
        y_train_tensor = torch.tensor(data_std["y_train"], dtype=torch.float32).view(
            -1, 1
        )
        x_eval_tensor = torch.tensor(data_std["x_test"], dtype=torch.float32)
        y_eval_tensor = torch.tensor(data_std["y_test"], dtype=torch.float32).view(
            -1, 1
        )
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        eval_dataset = TensorDataset(x_eval_tensor, y_eval_tensor)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size, shuffle=False)

        loaders[cid] = ((cid, name), (train_loader, eval_loader))

    return loaders
