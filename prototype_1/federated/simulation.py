import flwr as fl
from flwr.common import Metrics, NDArrays
from flwr_datasets import FederatedDataset
import torch
import numpy as np
from collections import OrderedDict

from typing import List, Tuple, Any

from ..neural_helper.mlp import PopoolaMLP, train, test_metrics
from .utils_fl import get_all_federated_loaders


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    

class FlowerNumPyClient(fl.client.NumPyClient):
    def __init__(self, cid, net, train_loader, eval_loader, lr, momentum):
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.lr = lr
        self.momentum = momentum

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.train_loader, lr=self.lr, momentum=self.momentum)
        return get_parameters(self.net), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test_metrics(self.net, self.eval_loader)
        return float(loss), len(self.eval_loader), {"accuracy": float(accuracy)}

    
def client_fn(cid: str):
    idx = int(cid) % len(LOADERS)
    train_loader, eval_loader = LOADERS[idx]
    
    print(f"\nEstou chamada {cid} | {idx}")
    
    lr = 0.001
    momentum = 0.9
    
    model = PopoolaMLP().to(DEVICE)
    return FlowerNumPyClient(cid, model, train_loader, eval_loader, lr, momentum).to_client()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    LOADERS = get_all_federated_loaders(batch_size=512)
    DEVICE = "cuda"
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    clients_resources = {"num_cpus": 1, "num_gpus": 1}
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources=clients_resources,
    )
