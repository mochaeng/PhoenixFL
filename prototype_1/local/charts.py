import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict

from pre_process.pre_process import CLIENTS_NAMES


TRAIN_LOGS_PATH = "local/metrics/train_logs.json"
PATH_TO_SAVE_CHARTS = "local/charts/"

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

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 12))
    axes_list = axs.flatten()

    x = list(map(lambda value: value + 1, x))
    for idx, client_name in enumerate(CLIENTS_NAMES):
        if idx == len(CLIENTS_NAMES):
            break

        y = plot_metrics[client_name]
        axes_list[idx].plot(
            x,
            y,
            marker="^",
            color="#FF8C00",
            # mfc="none",
            markersize=8,
        )
        axes_list[idx].set(title=client_name)
        # axes_list[idx].set_yscale("logit")

        # axes_list[idx].set_ylim(min(y) - 0.05, max(y) + 0.05)
        # axes_list[idx].set_xlim(min(x) - 10, max(x) + 10)

        # axes_list[idx].axis(
        #     [min(x) - 1, max(x) + 1, round(min(y)) - 1, round(max(y)) + 1]
        # )
        axes_list[idx].yaxis.set_major_formatter(mticker.ScalarFormatter())
        axes_list[idx].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        axes_list[idx].grid()

    fig.tight_layout()
    fig.savefig(f"{PATH_TO_SAVE_CHARTS}/local_bce.png")
