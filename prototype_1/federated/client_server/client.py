import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import flwr as fl
import argparse
from typing import Iterator
from federated.federated_helpers import FederatedMetrics

from neural.train_test import train, evaluate_model
from neural.helpers import DEVICE
from neural.architectures import MLP
from federated.federated_helpers import (
    get_all_federated_loaders,
    set_parameters,
    get_parameters,
)


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
    def __init__(self, cid, name, train_loader, eval_loader):
        self.cid = cid
        self.name = name
        self.net: nn.Module
        self.train_loader = train_loader
        self.eval_loader = eval_loader

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
        print(f"\n[Client {self.cid}] evaluate, config: {config}")

        # set_parameters(self.net, parameters)
        metrics = evaluate_model(self.net, self.eval_loader)

        print(f"{self.name}, accuracy: {float(metrics['accuracy'])})")

        return (
            metrics["final_loss"],
            len(self.eval_loader),
            metrics,
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

    LOADERS = get_all_federated_loaders(batch_size=512)
    (_, name), (train_loader, eval_loader) = LOADERS[cid]

    # centralized_eval_loader =

    federated_metrics = FederatedMetrics()

    fl.client.start_client(
        server_address="[::]:8080",
        client=Client(
            cid=cid,
            name=name,
            train_loader=train_loader,
            eval_loader=eval_loader,
        ).to_client(),
    )
