import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, Tuple, List

from ..pre_process import CLIENTS_NAMES


TRAIN_LOGS_PATH = "prototype_1/local/metrics/train_logs.json"
PATH_TO_SAVE_CHARTS = "prototype_1/local/charts/"

if __name__ == "__main__":
    with open(TRAIN_LOGS_PATH, "r+") as file:
        train_logs = json.load(file)

    plot_metrics = {}

    x = list(range(len(train_logs[CLIENTS_NAMES[0]])))
    for client_name in CLIENTS_NAMES:
        epochs_metrics: Dict = train_logs[client_name]
        client_y = [epochs_metrics[f"epoch_{epoch_idx}"]["loss"] for epoch_idx in x]
        plot_metrics[client_name] = client_y

    print(plot_metrics)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    axes_list = axs.flatten()

    x = list(map(lambda value: value + 1, x))
    for idx, client_name in enumerate(CLIENTS_NAMES):
        axes_list[idx].plot(x, plot_metrics[client_name])
        axes_list[idx].set(title=client_name)
        axes_list[idx].set_yscale("logit")
        axes_list[idx].set_ylim(
            min(plot_metrics[client_name]), max(plot_metrics[client_name])
        )
        axes_list[idx].yaxis.set_major_formatter(mticker.ScalarFormatter())

    fig.tight_layout()
    fig.savefig(f"{PATH_TO_SAVE_CHARTS}/local_bce.png")
