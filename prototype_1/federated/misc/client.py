import torch.nn as nn
import flwr as fl
import argparse

from neural_helper.mlp import (
    MLP,
    train,
    evaluate_model,
    DEVICE,
)
from federated.federated_helpers import (
    get_all_federated_loaders,
    set_parameters,
    get_parameters,
)


class Client(fl.client.NumPyClient):
    def __init__(self, cid, name, net: nn.Module, train_loader, eval_loader):
        self.cid = cid
        self.name = name
        self.net = net
        self.train_loader = train_loader
        self.eval_loader = eval_loader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"Configs are {config}")
        server_round = config["server_round"]
        del config["server_round"]

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
    model = MLP().to(DEVICE)
    (_, name), (train_loader, eval_loader) = LOADERS[cid]

    fl.client.start_client(
        server_address="[::]:8080",
        client=Client(
            cid=cid,
            name=name,
            net=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
        ).to_client(),
    )
