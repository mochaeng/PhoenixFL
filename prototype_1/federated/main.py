import flwr as fl
from flwr.common import Metrics, Scalar
from flwr.server import History
from typing import List, Tuple, Dict
import json
import argparse
import uuid
import datetime

from ..pre_process import BATCH_SIZE
from ..neural_helper.mlp import MLP, train, evaluate_model, DEVICE, TRAIN_CONFIG
from .federated_helpers import (
    get_all_federated_loaders,
    get_parameters,
    set_parameters,
    NUM_CLIENTS,
    PATH_TO_METRICS_FOLDER,
    FederatedMetricsRecord,
)
from .strategies import create_federated_strategy


class FlowerNumPyClient(fl.client.NumPyClient):
    def __init__(self, cid, name, net, train_loader, eval_loader):
        self.cid = cid
        self.name = name
        self.net = net
        self.train_loader = train_loader
        self.eval_loader = eval_loader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        server_round = config["server_round"]
        del config["server_round"]

        # possible_hyperparameters = [
        #     "proximal_mu",
        #     "momentum",
        #     "weight_decay",
        #     "optimizer",
        # ]
        # train_config = {
        #     "epochs": config["epochs"],
        #     "lr": config["lr"],
        #     **{key: config[key] for key in possible_hyperparameters if key in config},
        # }

        print(f"\n[Client {self.cid}], round {server_round} fit, config: {config}")

        set_parameters(self.net, parameters)
        train(self.net, self.train_loader, train_config=config)
        return get_parameters(self.net), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        print(f"\n[Client {self.cid}] evaluate, config: {config}")

        set_parameters(self.net, parameters)
        metrics = evaluate_model(self.net, self.eval_loader)

        print(f"{self.name}, accuracy: {float(metrics['accuracy'])})")

        return (
            metrics["final_loss"],
            len(self.eval_loader),
            metrics,
        )


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


def client_fn(cid: str):
    idx = int(cid) % len(LOADERS)
    (cid_, name), (train_loader, eval_loader) = LOADERS[idx]

    model = MLP().to(DEVICE)

    return FlowerNumPyClient(
        cid_,
        name,
        model,
        train_loader,
        eval_loader,
    ).to_client()


def federated_evaluation_results(
    server_round, metrics: List[Tuple[int, Metrics]]
) -> None:
    print("\nFederated evaluation results\n")
    global metrics_record

    for cid, results in metrics:
        idx = cid % len(LOADERS)
        (cid_, name), _ = LOADERS[idx]
        assert cid_ == cid
        metrics_record.set_values(server_round, name, results)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print("Weighted average")
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulating the training of an ML model with Federated Learning"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        help="The number of rounds of FL",
        default=2,
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        help="The number of clients that will be participating in the training process",
        default=NUM_CLIENTS,
    )
    parser.add_argument(
        "--save-results",
        type=bool,
        help="If you want to save the results of a federated evaluation to a file",
        default=True,
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["fedprox", "fedavg"],
        help="Federeated algorithm for aggregation",
        default="fedavg",
    )
    parser.add_argument(
        "--mu",
        type=float,
        help="Value for proximal_mu",
        default=1.0,
    )

    args = parser.parse_args()
    num_rounds = args.num_rounds
    num_clients = args.num_clients
    is_save_results = args.save_results
    strategy_name = args.algo
    proximal_mu = args.mu

    metrics_record = FederatedMetricsRecord(strategy_name)
    LOADERS = get_all_federated_loaders(BATCH_SIZE)
    starting_params = get_parameters(MLP().to(DEVICE))

    if is_save_results:
        time_str = datetime.datetime.now().strftime("%I:%M%p_%B%d%Y")
        uuid_hex = uuid.uuid4().hex
        TEMP_FILE_NAME = f"{PATH_TO_METRICS_FOLDER}/TEMP_results_{strategy_name}_{time_str}_{uuid_hex}.json"

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
    }

    if strategy_name == "fedprox" and proximal_mu is not None:
        strategy_config["proximal_mu"] = proximal_mu

    strategy = create_federated_strategy(
        strategy_name, **strategy_config
    ).create_strategy(on_federated_evaluation_results=federated_evaluation_results)

    clients_resources = {"num_cpus": 3, "num_gpus": 1}

    history: History = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=clients_resources,
    )

    # # self.losses_distributed: List[Tuple[int, float]] = []
    # # self.losses_centralized: List[Tuple[int, float]] = []
    # # self.metrics_distributed_fit: Dict[str, List[Tuple[int, Scalar]]] = {}
    # # self.metrics_distributed: Dict[str, List[Tuple[int, Scalar]]] = {}
    # # self.metrics_centralized: Dict[str, List[Tuple[int, Scalar]]] = {}

    weighted_metrics = {
        "metrics_distributed": history.metrics_distributed,
        "losses_distributed": history.losses_distributed,
    }
    metrics_record.set_weighted_values(weighted_metrics)

    if is_save_results:
        with open(TEMP_FILE_NAME, "w+") as f:
            json.dump(metrics_record.get(), f, indent=4)

    print(f"FINAL: {metrics_record}")
