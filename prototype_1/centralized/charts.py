import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure

TRAIN_LOGS_PATH = "centralized/metrics/train_logs.json"
PATH_TO_SAVE_CHARTS = "centralized/charts/"


def get_losses_chart(
    model_metrics: dict,
) -> Figure:
    x = list(range(len(model_losses)))
    y = [model_losses[f"epoch_{epoch_idx}"]["loss"] for epoch_idx in x]

    fig, axs = plt.subplots(figsize=(10, 10))

    axs.plot(
        x,
        y,
        marker="^",
        color="#FF8C00",
        # mfc="none",
        markersize=8,
    )
    axs.set(title="Centralized Training Losses")
    # axes_list[idx].set_yscale("logit")
    # axes_list[idx].set_ylim(min(y) - 0.05, max(y) + 0.05)
    # axes_list[idx].set_xlim(min(x) - 10, max(x) + 10)
    # axes_list[idx].axis(
    #     [min(x) - 1, max(x) + 1, round(min(y)) - 1, round(max(y)) + 1]
    # )
    axs.yaxis.set_major_formatter(mticker.ScalarFormatter())
    axs.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axs.grid()
    return fig


if __name__ == "__main__":
    with open(TRAIN_LOGS_PATH, "r+") as file:
        train_logs = json.load(file)

    plot_metrics = {}

    model_losses = train_logs["0"]

    # x = list(range(len(model_losses)))
    # y = [model_losses[f"epoch_{epoch_idx}"]["loss"] for epoch_idx in x]

    # fig, axs = plt.subplots(figsize=(10, 10))
    # # axes_list = axs.flatten()

    # axs.plot(
    #     x,
    #     y,
    #     marker="^",
    #     color="#FF8C00",
    #     # mfc="none",
    #     markersize=8,
    # )
    # axs.set(title="Centralized Training Losses")
    # # axes_list[idx].set_yscale("logit")

    # # axes_list[idx].set_ylim(min(y) - 0.05, max(y) + 0.05)
    # # axes_list[idx].set_xlim(min(x) - 10, max(x) + 10)

    # # axes_list[idx].axis(
    # #     [min(x) - 1, max(x) + 1, round(min(y)) - 1, round(max(y)) + 1]
    # # )
    # axs.yaxis.set_major_formatter(mticker.ScalarFormatter())
    # axs.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    # axs.grid()

    losses_charts = get_losses_chart(model_metrics=model_losses)
    losses_charts.tight_layout()
    losses_charts.savefig(f"{PATH_TO_SAVE_CHARTS}/local_bce.png")
