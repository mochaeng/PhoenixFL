import json
import os
from scipy import stats

from .centralized_helpers import GroupByCentralizedMetricsRecorder


METRICS_PATH = "./prototype_1/centralized/metrics"
METRICS_FILE_PATH = os.path.join(METRICS_PATH, "metrics.json")
GROUPBY_METRICS_FILE_PATH = os.path.join(METRICS_PATH, "groupby_metrics.json")


if __name__ == "__main__":
    with open(METRICS_FILE_PATH, "r") as f:
        json_dict: dict = json.load(f)

    clients_names = ["client-1: ToN", "client-2: BoT", "client-3: UNSW"]
    metrics_names = ["accuracy", "precision", "recall", "f1_score", "final_loss"]
    grouby_metrics = GroupByCentralizedMetricsRecorder(
        json_dict, clients_names, metrics_names
    )

    for idx, client_name in enumerate(clients_names):
        path = os.path.join(METRICS_PATH, f"{client_name}_{idx+1}")
        if not os.path.exists(path):
            os.makedirs(path)

        means = grouby_metrics.as_df()[client_name].mean()
        stds = grouby_metrics.as_df()[client_name].std()
        covs = grouby_metrics.as_df()[client_name].cov()

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

    with open(GROUPBY_METRICS_FILE_PATH, "w") as f:
        json.dump(grouby_metrics.as_dict(), f, indent=4)
