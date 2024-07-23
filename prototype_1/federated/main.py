import os
import flwr as fl
from flwr.common import Metrics
from flwr.server import History
from typing import List, Tuple
import json
import argparse

from pre_process.pre_process import BATCH_SIZE
from neural.architectures import MLP
from neural.train_eval import train, evaluate_model
from neural.helpers import DEVICE, TRAIN_CONFIG, zeroing_parameters
from federated.federated_helpers import (
    get_all_federated_loaders,
    get_parameters,
    set_parameters,
    TOTAL_NUMBER_OF_CLIENTS,
    FederatedMetrics,
    PATH_TO_METRICS_FOLDER,
    fit_config,
    eval_config,
)
from federated.strategies.factory import create_federated_strategy


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
        server_round = int(config["server_round"])
        del config["server_round"]

        set_parameters(self.net, parameters)

        # if server_round >= 1 and server_round <= 5:
        #     config["epochs"] = 3
        # else:
        #     config["epochs"] = 5

        if "eta_l" in config:
            config["lr"] = config["eta_l"]

        if "proximal_mu" in config:
            training_style = "fedprox"
        else:
            training_style = "standard"

        print(f"\n[Client {self.cid}], round {server_round} fit, config: {config}")

        train(
            net=self.net,
            trainloader=self.train_loader,
            training_style=training_style,
            train_config=config,
        )

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


def client_fn(cid: str):
    idx = int(cid) % len(LOADERS)
    (cid_, name, _), (train_loader, eval_loader) = LOADERS[idx]

    model = MLP().to(DEVICE)

    return FlowerNumPyClient(
        cid_,
        name,
        model,
        train_loader=train_loader,
        eval_loader=eval_loader,
    ).to_client()


def federated_evaluation_results(server_round, metrics) -> None:
    print("\nFederated evaluation results\n")
    global federated_metrics

    for cid, results in metrics:
        idx = cid % len(LOADERS)
        (cid_, name, _), _ = LOADERS[idx]
        assert cid_ == cid

        # del results["final_loss"]
        federated_metrics.add_client_evaluated_results(name, results)


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
        "--num-models",
        type=int,
        help="The total number of models you want to simulate",
        default=1,
    )
    parser.add_argument(
        "--fit-clients",
        type=int,
        help="The number of clients who will train in a round",
        default=TOTAL_NUMBER_OF_CLIENTS,
    )
    parser.add_argument(
        "--eval-clients",
        type=int,
        help="The number of clients who will evaluate in a round",
        default=TOTAL_NUMBER_OF_CLIENTS,
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
        choices=[
            "fedprox",
            "fedavg",
            "fedadagrad",
            "fedadam",
            "fedyogi",
            "fedmedian",
            "qfedavg",
            "fedtrimmed",
        ],
        help="Federeated algorithm for aggregation",
        default="fedavg",
    )
    parser.add_argument(
        "--mu",
        type=float,
        help="Value for proximal_mu",
        default=1.0,
    )
    parser.add_argument(
        "--tau",
        type=float,
        help="Value for tau (fedadam, fedadagrad and fedyogi)",
        default=1e-9,
    )
    parser.add_argument(
        "--eta",
        type=float,
        help="Value for eta (fedadam, fedadagrad and fedyogi)",
        default=1e-2,
    )
    parser.add_argument(
        "--eta_l",
        type=float,
        help="Value for eta_l (fedadam, fedadagrad and fedyogi)",
        default=1e-2,
    )

    args = parser.parse_args()
    num_rounds = args.num_rounds
    fit_clients = args.fit_clients
    eval_clients = args.eval_clients
    is_save_results = args.save_results
    strategy_name = args.algo
    proximal_mu = args.mu
    num_models = args.num_models
    tau = args.tau
    eta = args.eta
    eta_l = args.eta_l

    if num_models <= 0:
        raise ValueError(
            "you should train at least one model :(  Come on you can do it!"
        )

    if fit_clients <= 0 or fit_clients > TOTAL_NUMBER_OF_CLIENTS:
        raise ValueError(
            f"Number of clients to train should be between: 1 and {TOTAL_NUMBER_OF_CLIENTS}"
        )
    if eval_clients <= 0 or eval_clients > TOTAL_NUMBER_OF_CLIENTS:
        raise ValueError(
            f"Number of clients to test should be between: 1 and {TOTAL_NUMBER_OF_CLIENTS}"
        )

    LOADERS = get_all_federated_loaders(BATCH_SIZE)
    initial_model = MLP().to(DEVICE)
    # zeroing_parameters(initial_model)
    starting_params = get_parameters(initial_model)

    strategy_config = {
        "fraction_fit": fit_clients / TOTAL_NUMBER_OF_CLIENTS,
        "fraction_evaluate": eval_clients / TOTAL_NUMBER_OF_CLIENTS,
        "min_fit_clients": fit_clients,
        "min_evaluate_clients": eval_clients,
        "min_available_clients": TOTAL_NUMBER_OF_CLIENTS,
        "evaluate_metrics_aggregation_fn": weighted_average,
        "initial_parameters": fl.common.ndarrays_to_parameters(starting_params),
        "on_fit_config_fn": fit_config,
        "on_evaluate_config_fn": eval_config,
    }

    if strategy_name == "fedprox" and proximal_mu is not None:
        strategy_config["proximal_mu"] = proximal_mu

    if (
        strategy_name == "fedadam"
        or strategy_name == "fedyogi"
        or strategy_name == "fedadagrad"
    ):
        strategy_config["tau"] = tau
        strategy_config["eta"] = eta
        strategy_config["eta_l"] = eta_l

    strategy = create_federated_strategy(
        strategy_name, **strategy_config
    ).create_strategy(on_federated_evaluation_results=federated_evaluation_results)

    clients_resources = {"num_cpus": 3, "num_gpus": 1}

    federated_metrics = FederatedMetrics()
    weighteds_metrics = []

    for num_model in range(num_models):
        history: History = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=TOTAL_NUMBER_OF_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources=clients_resources,
        )

        weighted_metrics = {
            "metrics_distributed": history.metrics_distributed,
            "losses_distributed": history.losses_distributed,
        }
        weighteds_metrics.append({f"model_{num_model + 1}": weighted_metrics})

        if num_model < num_models - 1:
            federated_metrics.add_new_round()

    federated_metrics.add_weighteds_metrics(weighteds_metrics)

    if is_save_results:
        file_path = os.path.join(
            PATH_TO_METRICS_FOLDER,
            f"metrics_{strategy_name}.json",
        )
        with open(file_path, "w+") as f:
            json.dump(federated_metrics.get_metrics(), f, indent=4)
