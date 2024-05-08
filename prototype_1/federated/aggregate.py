import os
import argparse
import json


METRICS_FOLDER_PATH = "prototype_1/federated/metrics"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combining the TEMP_METRICS")

    files = os.listdir(METRICS_FOLDER_PATH)
    temp_files = [
        file for file in files if file.startswith("TEMP") and file.endswith(".json")
    ]

    aggregated_metrics = {}

    for idx, temp_file in enumerate(temp_files):
        temp_file_path = os.path.join(METRICS_FOLDER_PATH, temp_file)
        model_metrics = {}
        with open(temp_file_path, "r") as f:
            aggregated_metrics[f"model_{idx}"] = json.load(f)

    metrics_path = os.path.join(METRICS_FOLDER_PATH, "metrics.json")
    with open(metrics_path, "w+") as f:
        json.dump(aggregated_metrics, f, indent=4)
