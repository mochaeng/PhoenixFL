from typing import Dict
from flwr.common import NDArrays
from ..neural_helper.mlp import PopoolaMLP, FnidsMLP, load_data, train, test_metrics, DEVICE
from .utils_fl import get_all_federated_loaders

from collections import OrderedDict
import torch
import torch.nn as nn
import flwr as fl


class Client(fl.client.NumPyClient):
    def __init__(self, model: nn.Module, train_loader, eval_loader):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, epochs=1)
        return self.get_parameters(config={}), len(self.train_loader), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test_metrics(self.model, self.eval_loader)
        return float(loss), len(self.eval_loader), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    LOADERS = get_all_federated_loaders()
    
    model = PopoolaMLP().to(DEVICE)
    train_loader, eval_loader = LOADERS[0]
    
    fl.client.start_client(
        server_address="[::]:8080",
        client=Client(model, train_loader=train_loader, eval_loader=eval_loader).to_client(),
    )
    