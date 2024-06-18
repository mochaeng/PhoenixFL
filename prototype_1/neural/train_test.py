import torch
import torch.amp
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)
import copy
from typing import Dict, Iterator, Tuple, Optional, Any, OrderedDict, Union, Literal
from neural.architectures import MLP
from neural.helpers import (
    DEVICE,
    CRITERION,
    get_optimizer,
    calculate_proximal_term,
    calculate_regularization_term,
    calculate_sigma_penalty,
    calculate_regularization_degree,
    initialize_local_model_for_fedplus,
    check_models_equality,
)

TrainingStyle = Literal["standard"] | Literal["fedprox"] | Literal["fed+"]


def train(
    net: nn.Module,
    trainloader: DataLoader,
    training_style: TrainingStyle = "standard",
    train_config: dict = {
        "epochs": 10,
        "lr": 1e-4,
        "momentum": 0.9,
        "weight_decay": 1e-5,
    },
) -> Optional[Dict]:
    match training_style:
        case "fedprox":
            train_func = train_with_proximal_term
        case "fed+":
            train_func = train_with_smooth_delta
        case _:
            train_func = train_standard

    return train_func(net, trainloader, train_config)


def train_standard(
    net: nn.Module,
    trainloader: DataLoader,
    train_config={"epochs": 10, "lr": 1e-4, "momentum": 0.9, "weight_decay": 1e-5},
):
    criterion = CRITERION()

    optimizer = get_optimizer(train_config.get("optimizer"), net, train_config)
    epochs_logs = {}

    net.train()
    for epoch in range(train_config["epochs"]):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            round_outputs = torch.round(torch.sigmoid(outputs))
            correct += (round_outputs == labels).sum().item()

        epoch_loss /= len(trainloader.dataset)  # type: ignore
        epoch_acc = correct / total
        epochs_logs[f"epoch_{epoch}"] = {
            "loss": float(epoch_loss),
            # "acc": float(epoch_acc),
        }

        if train_config["is_verbose"]:
            print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    if train_config["is_epochs_logs"]:
        return epochs_logs


def train_with_proximal_term(
    net: nn.Module,
    trainloader: DataLoader,
    train_config={"epochs": 10, "lr": 1e-4, "momentum": 0.9, "weight_decay": 1e-5},
):
    criterion = CRITERION()
    optimizer = get_optimizer(train_config.get("optimizer"), net, train_config)
    global_params = copy.deepcopy(net).parameters()

    epochs_logs = {}

    net.train()
    for epoch in range(train_config["epochs"]):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(inputs)

            proximal_term = calculate_proximal_term(net, global_params)
            loss = (
                criterion(outputs, labels)
                + (train_config["proximal_mu"] / 2) * proximal_term
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            round_outputs = torch.round(torch.sigmoid(outputs))
            correct += (round_outputs == labels).sum().item()

        epoch_loss /= len(trainloader.dataset)  # type: ignore
        epoch_acc = correct / total
        epochs_logs[f"epoch_{epoch}"] = {
            "loss": float(epoch_loss),
            # "acc": float(epoch_acc),
        }

        if train_config["is_verbose"]:
            print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    if train_config["is_epochs_logs"]:
        return epochs_logs


def train_with_smooth_delta(
    net: nn.Module,
    trainloader: DataLoader,
    train_config={"epochs": 10, "lr": 1e-4, "momentum": 0.9, "weight_decay": 1e-5},
):
    if (
        "aggregated_model" not in train_config
        or "smooth_delta" not in train_config
        or "lambda" not in train_config
    ):
        raise ValueError("Necessary values missing for training with smooth_delta")

    criterion = CRITERION()
    optimizer = get_optimizer(train_config.get("optimizer"), net, train_config)

    global_model: MLP = train_config["aggregated_model"]
    learning_rate = train_config["lr"]
    delta = train_config["smooth_delta"]
    lambda_constant = train_config["lambda"]

    sigma = calculate_sigma_penalty(learning_rate)
    k = calculate_regularization_degree(sigma, train_config["lr"])
    theta = calculate_regularization_term(net, global_model, delta)
    initialize_local_model_for_fedplus(
        net.parameters(), global_model.parameters(), lambda_constant
    )

    epochs_logs = {}

    net.train()
    for epoch in range(train_config["epochs"]):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            # net.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            with torch.no_grad():
                for local_param, global_param, theta_param in zip(
                    net.parameters(), global_model.parameters(), theta.parameters()
                ):
                    update = k * local_param + (1 - k) * (global_param + theta_param)
                    local_param.copy_(update)

            # with torch.no_grad():
            #     for local_param, global_param, theta_param in zip(
            #         net.parameters(),
            #         global_model.parameters(),
            #         theta.parameters(),
            #     ):
            #         # optim = local_param.data - train_config["lr"] * local_param.grad
            #         # updated_param = (k * optim) + (1 - k) * (global_param + theta_param)
            #         # local_param.copy_(updated_param)

            epoch_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            round_outputs = torch.round(torch.sigmoid(outputs))
            correct += (round_outputs == labels).sum().item()

        epoch_loss /= len(trainloader.dataset)  # type: ignore
        epoch_acc = correct / total
        epochs_logs[f"epoch_{epoch}"] = {
            "loss": float(epoch_loss),
            # "acc": float(epoch_acc),
        }

        if train_config["is_verbose"]:
            print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    if train_config["is_epochs_logs"]:
        return epochs_logs


def evaluate_model(net: nn.Module, testloader) -> Dict[str, float]:
    criterion = CRITERION()

    accuracy = BinaryAccuracy().to(DEVICE)
    precision = BinaryPrecision().to(DEVICE)
    recall = BinaryRecall().to(DEVICE)
    f1_score = BinaryF1Score().to(DEVICE)

    val_loss = 0
    metrics: Dict[str, float] = {}

    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            # if not torch.isnan(loss):
            val_loss += loss.item() * inputs.size(0)
            # else:
            #     print(f"Inputs range: {inputs.min().item()} to {inputs.max().item()}")
            #     print(f"Labels range: {labels.min().item()} to {labels.max().item()}")
            #     print(
            #         f"Outputs range: {outputs.min().item()} to {outputs.max().item()}"
            #     )
            #     print(f"loss: {loss}")

            # print(f"Inputs range: {inputs.min().item()} to {inputs.max().item()}")
            # print(f"Labels range: {labels.min().item()} to {labels.max().item()}")
            # print(f"Outputs range: {outputs.min().item()} to {outputs.max().item()}")

            predicted = torch.round(torch.sigmoid(outputs.data))
            # print(f"loss: {val_loss}")

            labels = labels.bool()
            predicted = predicted.view(-1)
            labels = labels.view(-1)

            accuracy.update(predicted, labels)
            precision.update(predicted, labels)
            recall.update(predicted, labels)
            f1_score.update(predicted, labels)

    val_loss /= len(testloader.dataset)

    metrics = {
        "accuracy": accuracy.compute().item(),
        "precision": precision.compute().item(),
        "recall": recall.compute().item(),
        "f1_score": f1_score.compute().item(),
        "final_loss": val_loss,
    }

    return metrics
