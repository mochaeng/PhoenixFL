import torch.nn as nn
import flwr as fl
import argparse
from federated.federated_helpers import FederatedMetrics
import os

from pre_process.pre_process import BATCH_SIZE
from prototype_1.neural.train_eval import train, evaluate_model
from neural.helpers import DEVICE
from neural.architectures import MLP
from federated.federated_helpers import (
    get_all_federated_loaders,
    set_parameters,
    get_parameters,
    get_centralized_test_loader,
)

RESULTS_FOLDER_PATH = "federated/client_server/results"


def write_results_to_file(results: str, name: str):
    file_path = os.path.join(RESULTS_FOLDER_PATH, f"{name}.txt")
    with open(file_path, "a+") as f:
        f.write(results)


def federated_evaluation_results(server_round, metrics) -> None:
    print("\nFederated evaluation results\n")
    global federated_metrics, name

    # for cid, results in metrics:
    #     # idx = cid % len(LOADERS)
    #     # (cid_, name), _ = LOADERS[idx]
    #     # assert cid_ == cid
    #     del results["final_loss"]
    #     federated_metrics.add_client_evaluated_results(name, results)


class Client(fl.client.NumPyClient):
    def __init__(self, cid, name, train_loader, eval_loader, centralized_eval_loader):
        self.cid = cid
        self.name = name
        self.net: nn.Module
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.centralized_eval_loader = centralized_eval_loader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"Configs are {config}")
        server_round = config["server_round"]
        train_config: dict = config

        if server_round == 1:
            self.net = MLP().to(DEVICE)
            set_parameters(self.net, parameters)

        print(f"\n[Client {self.cid}], round {server_round} fit, config: {config}")

        aggregated_model = MLP().to(DEVICE)
        set_parameters(aggregated_model, parameters)
        train_config["aggregated_model"] = aggregated_model

        if "smooth_delta" in config:
            training_style = "fed+"
        elif "proximal_mu" in config:
            training_style = "fedprox"
        else:
            training_style = "standard"

        train(
            net=self.net,
            trainloader=self.train_loader,
            training_style=training_style,
            train_config=config,
        )

        return get_parameters(self.net), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        global logger

        print(f"\n[Client {self.cid}] evaluate, config: {config}")

        # model_to_evaluate = MLP().to(DEVICE)
        # set_parameters(model_to_evaluate, parameters)

        results_str = f"\nServer round: {config['server_round']}\n"
        results_str += "Evaluating client's model on its own testset\n"
        own_metrics = evaluate_model(self.net, self.eval_loader)
        results_str += f"\taccuracy: {float(own_metrics['accuracy'])})\n"
        results_str += "Evaluating client's model on centralized testset\n"
        centralized_metrics = evaluate_model(self.net, self.centralized_eval_loader)
        results_str += f"\taccuracy: {float(centralized_metrics['accuracy'])})\n"
        print(results_str)
        write_results_to_file(results_str, self.name)

        return (
            own_metrics["final_loss"],
            len(self.eval_loader),
            own_metrics,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulating the training of an ML model with Federated Learning"
    )
    parser.add_argument(
        "--cid",
        type=int,
        help="Client id",
    )

    args = parser.parse_args()
    cid = args.cid

    if cid < 0 or cid > 3:
        raise ValueError("the client id should be between 0 and 3")

    federated_metrics = FederatedMetrics()

    client_loaders = get_all_federated_loaders(BATCH_SIZE)
    (cid, name, scaler), (train_loader, eval_loader) = client_loaders[cid]
    centralized_test_loader = get_centralized_test_loader(BATCH_SIZE, scaler)

    fl.client.start_client(
        server_address="[::]:8080",
        client=Client(
            cid=cid,
            name=name,
            train_loader=train_loader,
            eval_loader=eval_loader,
            centralized_eval_loader=centralized_test_loader,
        ).to_client(),
    )
