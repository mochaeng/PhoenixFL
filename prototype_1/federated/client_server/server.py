import flwr as fl
from flwr.common import Metrics, Scalar
from typing import List, Tuple

from federated.federated_helpers import (
    get_parameters,
)
from neural.helpers import TRAIN_CONFIG, DEVICE
from neural.architectures import MLP
from federated.strategies.factory import create_federated_strategy
from federated.client_server.client import federated_evaluation_results


def fit_config(server_round: int) -> dict[str, Scalar]:
    merge_config = TRAIN_CONFIG.copy()
    merge_config.update(
        {
            "server_round": server_round,
        }
    )
    return merge_config


def eval_config(server_round: int) -> dict[str, Scalar]:
    config: dict[str, Scalar] = {
        "server_round": server_round,
    }
    return config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    starting_params = get_parameters(MLP().to(DEVICE))

    num_clients = 4
    strategy_config = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": num_clients,
        "min_evaluate_clients": num_clients,
        "min_available_clients": num_clients,
        "evaluate_metrics_aggregation_fn": weighted_average,
        "initial_parameters": fl.common.ndarrays_to_parameters(starting_params),
        "on_fit_config_fn": fit_config,
        "on_evaluate_config_fn": eval_config,
        "lambda_value": 0,
    }

    strategy_name = "fedavgplus"
    strategy = create_federated_strategy(
        strategy_name, **strategy_config
    ).create_strategy(on_federated_evaluation_results=federated_evaluation_results)

    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,  # type: ignore
    )  # type: ignore
