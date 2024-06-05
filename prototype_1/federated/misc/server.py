import flwr as fl
from flwr.common import Metrics, Scalar
from typing import List, Tuple

from federated.federated_helpers import get_parameters
from neural_helper.mlp import TRAIN_CONFIG, MLP, DEVICE


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

    fl.server.strategy.QFedAvg

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(starting_params),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
    )

    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )
