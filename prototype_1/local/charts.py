import json
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pre_process.pre_process import CLIENTS_NAMES, CLIENTS_NAMES_CHARTS

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

    chart_letter = "a"
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 12))
    axes_list = axs.flatten()  # type: ignore

    x = list(map(lambda value: value + 1, x))
    for idx, client_name in enumerate(CLIENTS_NAMES):
        if idx == len(CLIENTS_NAMES):
            break

        y = plot_metrics[client_name]
        axes_list[idx].plot(
            x,
            y,
            marker="o",
            color="#FF8C00",
            # mfc="none",
            markersize=8,
        )
        axes_list[idx].set(title=CLIENTS_NAMES_CHARTS[client_name])
        axes_list[idx].yaxis.set_major_formatter(mticker.ScalarFormatter())
        axes_list[idx].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        axes_list[idx].set_xlabel(
            f"Épocas\n({chr(ord(chart_letter))})", linespacing=1.5
        )
        axes_list[idx].set_ylabel("Cross-Entropy Loss")
        axes_list[idx].grid()

        chart_letter = chr(ord(chart_letter) + 1)

    fig.tight_layout()
    fig.savefig(f"{PATH_TO_SAVE_CHARTS}/local_bce.pdf")
