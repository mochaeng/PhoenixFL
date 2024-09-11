import os
import json
import pandas as pd
from scipy.stats import shapiro


METRICS_FOLDER_PATH = "centralized/metrics"


if __name__ == "__main__":
    metrics_path = os.path.join(METRICS_FOLDER_PATH, "metrics.json")
    with open(metrics_path, "r+") as f:
        metrics = json.load(f)

    results_path = os.path.join(METRICS_FOLDER_PATH, "results.txt")

    file = open(results_path, "w+")

    for client_name in metrics.keys():
        file.write(f"[{client_name}]\n")

        values = metrics[client_name]
        del values["final_loss"]
        df = pd.DataFrame.from_dict(values)

        for metric in df:
            shapiro_test = shapiro(df[metric].values)

            file.write(f"\t[{metric}]:\n")
            file.write(f"\t\tmean: {df[metric].mean() * 100}\n")
            file.write(f"\t\tmedian: {df[metric].median()}\n")
            file.write(f"\t\tvariance: {df[metric].var()}\n")
            file.write(f"\t\tstd: {df[metric].std() * 100}\n")
            file.write(f"\t\tmax: {df[metric].max()}\n")
            file.write(f"\t\tmin: {df[metric].min()}\n")
            file.write(f"\t\tshapiro: {shapiro_test}\n")

        file.write("\n\n")

    file.close()
