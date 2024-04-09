import flwr as fl
from flwr.common import Metrics, Scalar
from torch.utils.data import DataLoader, TensorDataset

from typing import List, Tuple, Dict, Optional

from ..neural_helper.mlp import MLP, train, collect_metrics
from .federated_helpers import get_all_federated_loaders, get_parameters, set_parameters
from .custom_strategy import FedCustom


NUM_CLIENTS = 3


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
        local_epochs = int(config["local_epochs"])
        lr = float(config["lr"])
        momentum = float(config["momentum"])
        print(f"\n[Client {self.cid}], round {server_round} fit, config: {config}")

        set_parameters(self.net, parameters)
        train(
            self.net,
            self.train_loader,
            epochs=local_epochs,
            lr=lr,
            momentum=momentum,
        )
        return get_parameters(self.net), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        print(f"\n[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        metrics = collect_metrics(self.net, self.eval_loader)
        print(f"{self.name}, accuracy: {float(metrics['accuracy'])})")

        return (
            metrics["final_loss"],
            len(self.eval_loader),
            metrics,
        )


def fit_config(server_round: int) -> Dict[str, Scalar]:
    config: Dict[str, Scalar] = {
        "server_round": server_round,
        "local_epochs": 1,
        "lr": 0.001,
        "momentum": 0.9,
    }
    return config


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


def centralized_evaluation(
    server_round: int, params: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    _, eval_loader = LOADERS[0]
    net = MLP().to(DEVICE)
    set_parameters(net, params)
    metrics = collect_metrics(net, eval_loader)
    loss, acc = metrics["final_loss"], metrics["accuracy"]
    print(f"Server-side evaluation loss {loss} / accuracy {acc}")

    return loss, {"accuracy": acc}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    LOADERS = get_all_federated_loaders(batch_size=512)
    DEVICE = "cuda"

    # train_loader: DataLoader[TensorDataset]
    # train_loader, eval_loader = LOADERS[0]

    global_params = get_parameters(MLP().to(DEVICE))

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(global_params),
        # evaluate_fn=centralized_evaluation,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        # fit_metrics_aggregation_fn=
    )

    clients_resources = {"num_cpus": 1, "num_gpus": 1}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources=clients_resources,
    )
