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

    train_func(net, trainloader, train_config)

    # criterion = CRITERION()

    # optimizer = get_optimizer(train_config.get("optimizer"), net, train_config)

    # if "proximal_mu" in train_config:
    #     global_params = copy.deepcopy(net).parameters()
    # if "smooth_delta" in train_config and "aggregated_model" in train_config:
    #     global_model: MLP = train_config["aggregated_model"]
    #     theta_term = calculate_regularization_term(
    #         net, global_model, train_config["smooth_delta"]
    #     )
    #     sigma_penalty = calculate_sigma_penalty(
    #         constraint_value=0.9, learning_rate=train_config["lr"]
    #     )
    #     k_regularization_degree = calculate_regularization_degree(
    #         sigma_penalty, train_config["lr"]
    #     )

    # epochs_logs = {}

    # net.train()
    # for epoch in range(train_config["epochs"]):
    #     correct, total, epoch_loss = 0, 0, 0.0
    #     for batch_idx, (inputs, labels) in enumerate(trainloader):
    #         inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

    #         if "smooth_delta" in train_config:
    #             net.zero_grad()
    #         else:
    #             optimizer.zero_grad()

    #         outputs = net(inputs)

    #         if "proximal_mu" in train_config:
    #             proximal_term = calculate_proximal_term(net, global_params)
    #             loss = (
    #                 criterion(outputs, labels)
    #                 + (train_config["proximal_mu"] / 2) * proximal_term
    #             )
    #         else:
    #             loss = criterion(outputs, labels)

    #         loss.backward()

    #         if "smooth_delta" in train_config:
    #             # optimizer.step()
    #             with torch.no_grad():
    #                 for local_param, global_param, theta_param in zip(
    #                     net.parameters(),
    #                     global_model.parameters(),
    #                     theta_term.parameters(),
    #                 ):
    #                     optim = local_param.data - train_config["lr"] * local_param.grad
    #                     # updated_param = (k_regularization_degree * optim) + (
    #                     #     1 - k_regularization_degree
    #                     # ) * (global_param + theta_param)
    #                     updated_param = optim
    #                     local_param.copy_(updated_param)
    #         else:
    #             optimizer.step()

    #         # for (
    #         #     v1,
    #         #     v2,
    #         # ) in zip(updated_param, local_param):
    #         #     updated_param.equal(local_param)

    #         epoch_loss += loss.item() * inputs.size(0)
    #         total += labels.size(0)
    #         round_outputs = torch.round(torch.sigmoid(outputs))
    #         correct += (round_outputs == labels).sum().item()

    #     epoch_loss /= len(trainloader.dataset)  # type: ignore
    #     epoch_acc = correct / total
    #     epochs_logs[f"epoch_{epoch}"] = {
    #         "loss": float(epoch_loss),
    #         # "acc": float(epoch_acc),
    #     }

    #     if is_verbose:
    #         print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    # if is_epochs_logs:
    #     return epochs_logs


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

    theta = calculate_regularization_term(net, global_model, delta)
    sigma = calculate_sigma_penalty(learning_rate)
    k = calculate_regularization_degree(sigma, train_config["lr"])

    model_before = copy.deepcopy(net)
    initialize_local_model_for_fedplus(net, global_model, lambda_constant)
    equality = check_models_equality(model_before, net)
    print(f"Testing models equality: {equality}")

    epochs_logs = {}

    net.train()
    for epoch in range(train_config["epochs"]):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            # with torch.no_grad():
            #     # for local_param in net.parameters():
            #     #     update = local_param * k
            #     #     local_param.copy_(update)
            #     for local_param, global_param, theta_param in zip(
            #         net.parameters(), global_model.parameters(), theta.parameters()
            #     ):
            #         update = (local_param * k) + (1 - k) * (global_param + theta_param)
            #         local_param.copy_(update)

            # with torch.no_grad():
            #     for local_param, global_param, theta_param in zip(
            #         net.parameters(),
            #         global_model.parameters(),
            #         theta_term.parameters(),
            #     ):
            #         optim = local_param.data - train_config["lr"] * local_param.grad
            #         updated_param = k_regularization_degree * optim + (
            #             1 - k_regularization_degree
            #         ) * (global_param + theta_param)
            #         local_param.copy_(updated_param)

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
            predicted = torch.round(torch.sigmoid(outputs.data))

            val_loss += criterion(outputs, labels).item() * labels.size(0)

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
