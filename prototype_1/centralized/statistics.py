import json
import pandas as pd
import os
from scipy import stats


METRICS_PATH = "./prototype_1/centralized/metrics"


def aggregate_metrics_into_dict(all_metrics: dict, metrics_dict: dict) -> dict:
    for _, client in all_metrics.items():
        for client_name, client_metrics in client.items():
            for metric_name, value in client_metrics.items():
                if metric_name != "final_loss":
                    metrics_dict[client_name][metric_name].append(value)
    return metrics_dict


def get_metrics_dataframe_from_dict(metrics_dict: dict) -> dict[str, pd.DataFrame]:
    metrics_df = {
        client_name: pd.DataFrame.from_dict(metrics_dict[client_name])
        for client_name in metrics_dict.keys()
    }
    return metrics_df


if __name__ == "__main__":
    file_path = f"{METRICS_PATH}/metrics.json"

    with open(file_path, "r") as f:
        all_metrics: dict = json.load(f)

    clients_names = ["client-1: ToN", "client-2: BoT", "client-3: UNSW"]
    metrics_names = ["accuracy", "precision", "recall", "f1_score"]

    metrics_dict = {
        client_name: {metric_name: [] for metric_name in metrics_names}
        for client_name in clients_names
    }

    aggregated_metrics_by_client = aggregate_metrics_into_dict(
        all_metrics, metrics_dict
    )

    metrics_df = get_metrics_dataframe_from_dict(aggregated_metrics_by_client)

    for idx, client_name in enumerate(clients_names):
        path = f"{METRICS_PATH}/{client_name}_{idx+1}"
        if not os.path.exists(path):
            os.makedirs(path)

        means = metrics_df[client_name].mean()
        stds = metrics_df[client_name].std()
        covs = metrics_df[client_name].cov()

        means.to_json(f"{path}/means.json", indent=4)
        stds.to_json(f"{path}/stds.json", indent=4)
        covs.to_json(f"{path}/covs.json", indent=4)

        shapiro_means = stats.shapiro(means.to_numpy())
        shapiro_stds = stats.shapiro(stds.to_numpy())
        shapiro_covs = stats.shapiro(covs.to_numpy())

        shapiro = {
            "means": shapiro_means,
            "stds": shapiro_stds,
            "covs": shapiro_covs,
        }

        with open(f"{path}/shapiro.json", "w") as f:
            json.dump(shapiro, f, indent=4)
