import os
import json
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure

from ..pre_process import CLIENTS_NAMES


METRICS_PATH = "prototype_1/federated/metrics/"
CHARTS_PATH = "prototype_1/federated/charts/"


class FederatedMetricsChart:
    def __init__(self, metrics: dict, strategy: str, info: dict) -> None:
        self.metrics = metrics
        self.strategy = strategy
        self.info = info

    def __repr__(self) -> str:
        result = f"{self.get_all_weighted_metrics()} -> {self.strategy}"
        return str(result)

    def get_local_metrics_by_client(
        self, client_name: str, metric_name: str
    ) -> Optional[List[float]]:
        values = []
        for key in self.metrics.keys():
            if not key.startswith("round_"):
                continue
            try:
                value = self.metrics[key][client_name][metric_name]
            except KeyError:
                return None
            values.append(value)
        return values

    def get_all_weighted_metrics(self) -> Optional[dict]:
        try:
            return self.metrics["weighted"]["metrics_distributed"]
        except KeyError:
            return None

    def get_weighted_metrics(self, name) -> Optional[List[float]]:
        try:
            metrics = self.metrics["weighted"]["metrics_distributed"][name]
            return [metric[1] for metric in metrics]
        except KeyError:
            return None

    def has_weighted_metric(self, name) -> bool:
        return self.metrics["weighted"]["metrics_distributed"].get(name) is not None

    def num_rounds(self) -> Optional[int]:
        metrics = self.get_all_weighted_metrics()
        if metrics is not None:
            keys = list(metrics.keys())
            if len(keys) == 0:
                return None
            return len(metrics[keys[0]])


def get_weighted_chart_by_metric(
    metric_name: str, charts_metrics: List[FederatedMetricsChart]
) -> Figure:
    if len(charts_metrics) == 0:
        raise ValueError("error: empty list of models")

    model_by_metric: List[FederatedMetricsChart] = list(
        filter(lambda model: model.has_weighted_metric(metric_name), charts_metrics)
    )

    # print(model_by_metric)

    num_rounds = charts_metrics[0].num_rounds()
    if num_rounds is None:
        raise ValueError("error: cannot find the number of rounds")

    fig, axs = plt.subplots()
    plt.grid(True)

    x = list(range(1, num_rounds + 1))
    for model in charts_metrics:
        y = model.get_weighted_metrics(metric_name)
        axs.plot(x, y, label=model.strategy, marker="o")
        axs.xaxis.get_major_locator().set_params(integer=True)
        axs.legend()

    axs.set_xlabel("Communication rounds\n(a)", linespacing=2)
    axs.set_ylabel("Accuracy")

    return fig


def get_best_models_by_weighted_metric(
    name: str, metrics: dict, strategies: List[str]
) -> dict:
    best_models_name = {strategy: "" for strategy in strategies}
    best_value = {strategy: -1 for strategy in strategies}
    for model_name in metrics:
        model_strategy = metrics[model_name]["strategy"]

        metrics_values: List[List[int]] = metrics[model_name]["weighted"][
            "metrics_distributed"
        ].get(name)

        if metrics_values is None:
            raise KeyError(f"error: no metric with name {name}")

        value = metrics_values[-1][1]

        if value > best_value[model_strategy]:
            best_value[model_strategy] = value
            best_models_name[model_strategy] = model_name

    return best_models_name


def get_weighted_charts(
    metrics: dict,
    metrics_names: List[str],
    metric_to_filter: str,
    strategies: List[str],
) -> List[Tuple[Figure, str]]:
    charts_metrics: List[FederatedMetricsChart] = []

    best_models_names = get_best_models_by_weighted_metric(
        metric_to_filter, metrics, strategies
    )
    for strategy, model_name in best_models_names.items():
        chart_metrics = FederatedMetricsChart(metrics[model_name], strategy, {})
        charts_metrics.append(chart_metrics)

    charts: List[Tuple[Figure, str]] = []
    for metric_name in metrics_names:
        chart = get_weighted_chart_by_metric("accuracy", charts_metrics)
        chart.tight_layout()
        charts.append((chart, f"weighted_{metric_name}"))

    return charts


# def get_local_chart_by_metric()


def get_all_local_charts_by_metric(
    metric_name: str,
    metrics: dict,
    strategies: List[str],
    clients_names: List[str],
    info: dict,
):
    best_models = get_best_models_by_weighted_metric(metric_name, metrics, strategies)
    if len(best_models) == 0:
        raise ValueError("error: no model was found")

    federated_metrics: List[FederatedMetricsChart] = []
    for strategy, model_name in best_models.items():
        chart_metrics = FederatedMetricsChart(metrics[model_name], strategy, {})
        federated_metrics.append(chart_metrics)

    num_rounds = federated_metrics[0].num_rounds()
    if num_rounds is None:
        raise ValueError("error: impossible to found the number of rounds")
    x = list(range(1, num_rounds + 1))

    chart_letter = "a"
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    axs = axs.flatten()

    def set_chart(idx, x, y, client_name, info):
        axs[idx].plot(x, y, label=info["label"], marker=info["marker"])
        # axs[idx].set(title=client_name)
        axs[idx].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        axs[idx].set_xlabel(info["x_label"], linespacing=1.5)
        axs[idx].set_ylabel(info["y_label"])

    for idx, client_name in enumerate(clients_names):
        for federated_metric in federated_metrics:
            client_metrics = federated_metric.get_local_metrics_by_client(
                client_name, metric_name
            )
            marker = "o" if federated_metric.strategy == "fedavg" else "^"
            x_label = f"{info['x_label']}\n({chr(ord(chart_letter))})"
            strategy = "FedAvg" if federated_metric.strategy == "fedavg" else "FedProx"
            set_chart(
                idx=idx,
                x=x,
                y=client_metrics,
                client_name=client_name,
                info={
                    "marker": marker,
                    "x_label": x_label,
                    "y_label": info["y_label"],
                    "label": strategy,
                },
            )
        chart_letter = chr(ord(chart_letter) + 1)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines = [lines[0], lines[1]]
    fig.legend(
        lines,
        labels,
        loc="upper right",
        ncol=4,
    )

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    metrics_file_path = os.path.join(METRICS_PATH, "metrics.json")
    with open(metrics_file_path, "r+") as f:
        try:
            metrics = json.load(f)
        except json.JSONDecodeError as err:
            raise ValueError(f"error: {err}")

    strategies = ["fedavg", "fedprox"]
    metrics_names = ["accuracy"]
    metrics_y_axis = ["ACC"]

    metric_to_filter = metrics_names[0]

    weighted_charts = get_weighted_charts(
        metrics, metrics_names, metric_to_filter, strategies
    )

    local_charts = get_all_local_charts_by_metric(
        metrics_names[0],
        metrics,
        strategies,
        CLIENTS_NAMES,
        {"y_label": "ACC", "x_label": "Communication rounds"},
    ).savefig(f"{os.path.join(CHARTS_PATH, 'local_charts.png')}")

    # acc_chart = get_weighted_chart_by_metric("accuracy", models)
    # acc_chart.tight_layout()
    # acc_chart.savefig(f"{os.path.join(CHARTS_PATH, 'chart.pdf')}")