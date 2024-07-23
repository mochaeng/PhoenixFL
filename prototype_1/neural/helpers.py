import torch
import torch.amp
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from typing import Dict, Iterator, Tuple, Optional, Any, OrderedDict, Union, Literal

from neural.architectures import MLP


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CRITERION = torch.nn.BCEWithLogitsLoss


TRAIN_CONFIG = {
    "epochs": 5,
    "lr": 0.0001,
    "momentum": 0.9,
    "weight_decay": 0.1,
    "optimizer": "adam",
    "is_verbose": True,
    "is_epochs_logs": True,
}


def get_train_and_test_loaders(data: Dict, batch_size) -> Tuple[DataLoader, DataLoader]:
    x_train_tensor = torch.tensor(data["x_train"], dtype=torch.float32)
    y_train_tensor = torch.tensor(data["y_train"], dtype=torch.float32).view(-1, 1)
    x_test_tensor = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test_tensor = torch.tensor(data["y_test"], dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, test_loader


def get_test_loader(data: dict, batch_size):
    x_test_tensor = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test_tensor = torch.tensor(data["y_test"], dtype=torch.float32).view(-1, 1)

    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return test_loader


def calculate_proximal_term(
    local_model: nn.Module, global_params: Iterator[nn.Parameter]
):
    proximal_term = 0
    for local_weights, global_weights in zip(local_model.parameters(), global_params):
        proximal_term += (local_weights - global_weights).norm(2)
    return proximal_term


def calculate_regularization_term(
    local_model: nn.Module, global_model: nn.Module, delta: float
):
    theta = MLP().to(DEVICE)
    scaling_factor = (1 + delta) ** -1
    for theta_weights, local_weights, global_weights in zip(
        theta.parameters(), local_model.parameters(), global_model.parameters()
    ):
        with torch.no_grad():
            diff = scaling_factor * (local_weights - global_weights)
            theta_weights.copy_(diff)
    return theta


def check_models_equality(model1: nn.Module, model2: nn.Module) -> bool:
    for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
        if model1_param.data.ne(model2_param.data).sum() > 0:
            return False
    return True


def initialize_local_model_for_fedplus(
    local_params: Iterator[nn.Parameter],
    global_params: Iterator[nn.Parameter],
    lambda_value: float,
):
    with torch.no_grad():
        for local_param, global_param in zip(local_params, global_params):
            updated_param = (1 - lambda_value) * local_param + (
                lambda_value * global_param
            )
            local_param.copy_(updated_param)


def zeroing_parameters(model: nn.Module):
    for param in model.parameters():
        nn.init.constant_(param, 0)


def calculate_regularization_degree(sigma, learning_rate):
    return 1 / (1 + learning_rate * sigma)


def calculate_sigma_penalty(learning_rate, constraint_value=0.9):
    return (1 - constraint_value) / (learning_rate * constraint_value)


def get_parameters_as_tensor(model: nn.Module) -> torch.Tensor:
    params = [param.data.view(-1) for param in model.parameters()]
    tensor = torch.cat(params)
    return tensor


def get_optimizer(name, net, train_config):
    if name == "adam":
        return torch.optim.AdamW(
            net.parameters(),
            lr=train_config["lr"],
            weight_decay=train_config["weight_decay"],
        )
    else:
        return torch.optim.SGD(
            net.parameters(),
            lr=train_config["lr"],
            momentum=train_config["momentum"],
            weight_decay=train_config["weight_decay"],
        )
