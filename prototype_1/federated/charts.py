import os
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from result import Ok, Err
import argparse
import re

from pre_process.pre_process import read_file_as_dict


METRICS_PATH = "federated/metrics/"
CHARTS_PATH = "federated/charts/"


def get_model_idx_with_smallest_distributed_loss(models_metrics: dict):
    models_metrics = models_metrics[approach]["weighted_metrics"]
    smallest = None
    best_idx = 0
    for idx, model in enumerate(models_metrics):
        for _, values in model.items():
            losses = values["losses_distributed"]
            loss = losses[-1][1]
            if smallest is None:
                smallest = loss
                best_idx = idx
            elif loss < smallest:
                smallest = loss
                best_idx = idx
    return smallest, best_idx


def get_metrics(approach: str, is_weighted_metrics: bool = False):
    metrics = {}
    path = os.path.join(METRICS_PATH, approach)
    if not os.path.exists(path):
        raise ValueError(f"No folder found for the approach {approach}")

    for file_name in os.listdir(path):
        keys = re.split(r"[_.]", file_name)
        key = "_".join(keys[1:-1])

        match read_file_as_dict(path, file_name):
            case Ok(metrics_values):
                if not is_weighted_metrics:
                    del metrics_values["weighted_metrics"]
                metrics[key] = metrics_values
            case Err(_):
                raise ValueError("Could not load file")
    return metrics


def get_approach_metric_by_rounds(approach: str, metric_name: str):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
    axs = axs.flatten()  # type: ignore

    models_metrics = get_metrics(approach, True)
    _, best_model_idx = get_model_idx_with_smallest_distributed_loss(models_metrics)
    client_idx = 0
    del models_metrics[approach]["weighted_metrics"]

    for client_name, metrics in models_metrics[approach].items():
        print(client_name)

        client_idx += 1
        values = metrics[metric_name]

        x = range(1, len(values[0]) + 1)
        y = values[best_model_idx]
        print(x)
        print(y)

        client_idx = int(client_name.split(":")[0].split("-")[1]) - 1
        axs[client_idx].plot(x, y, label=client_name)
        axs[client_idx].set(title=client_name)
        axs[client_idx].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        axs[client_idx].set_xlabel("Communication rounds", linespacing=1.5)
        axs[client_idx].set_ylabel(metric_name)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines = [lines[0], lines[1], lines[2]]
    fig.legend(
        lines,
        labels,
        loc="upper right",
        bbox_to_anchor=(1, 1.00),
        ncol=4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    file_name_to_return = f"{approach}_metrics.png"
    return fig, file_name_to_return


def get_approach_loss_by_rounds(approach: str) -> Tuple[Figure, str]:
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
    axs = axs.flatten()  # type: ignore

    metrics = get_metrics(approach)
    for key, metrics_data in metrics.items():
        for client_name, client_data in metrics_data.items():
            for metric_name, values in client_data.items():
                if metric_name != "final_loss":
                    continue
                print(client_name)

                losses = values[0]
                rounds = len(losses)

                x = range(1, rounds + 1)
                y = losses

                match approach:
                    case "fedprox":
                        epochs, mu = key.split("_")
                        label = f"epochs = {epochs}, mu = {mu}"
                    case "fedadam":
                        epochs, tau, eta, eta_l = key.split("_")
                        label = (
                            f"epochs = {epochs}, tau = {tau}, eta = {eta}, etal={eta_l}"
                        )
                    case _:
                        epochs = key.split("_")
                        label = f"epochs = {epochs[0]}"

                client_idx = int(client_name.split(":")[0].split("-")[1]) - 1
                axs[client_idx].plot(x, y, label=label)
                axs[client_idx].set(title=client_name)
                axs[client_idx].xaxis.set_major_locator(
                    mticker.MaxNLocator(integer=True)
                )
                axs[client_idx].set_xlabel("Communication rounds", linespacing=1.5)
                axs[client_idx].set_ylabel("Epoch loss")

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines = [lines[0], lines[1], lines[2]]
    fig.legend(
        lines,
        labels,
        loc="upper right",
        bbox_to_anchor=(1, 1.00),
        ncol=4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    file_name_to_return = f"{approach}_losses.png"
    return fig, file_name_to_return


strategies_names = [
    "fedavg",
    "fedprox",
    "fedadagrad",
    "fedadam",
    "fedyogi",
    "fedmedian",
    "fedtrimmed",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creating charts for federated learning approach"
    )
    parser.add_argument(
        "--style",
        type=str,
        help="The style of charts method you want to make",
        choices=["loss", "metrics"],
        required=True,
    )
    parser.add_argument(
        "--approach",
        type=str,
        help="The name the approach, e.g., fedavg, fedprox",
        # choices=strategies_names.copy().append("all"),
        choices=strategies_names,
        required=True,
    )

    args = parser.parse_args()
    style = args.style
    approach = args.approach

    match style:
        case "loss":
            if approach == "all":
                raise ValueError("the loss graphs should be for a single approach")
            figure, file_name = get_approach_loss_by_rounds(approach)
        case "metrics":
            figure, file_name = get_approach_metric_by_rounds(approach, "f1_score")

    path = os.path.join(CHARTS_PATH, file_name)
    figure.savefig(path)
