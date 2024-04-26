import argparse
import json
from typing import Dict, List

from .federated_helpers import (
    METRICS_FILE_PATH,
    AggregatedFederatedMetricsRecorder,
)
from ..pre_process import CLIENTS_NAMES, METRICS_NAMES


def concat_all_temp_files(): ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combining the TEMP_METRICS")
    parser.add_argument(
        "--num-models",
        type=int,
        help="The number of evaluated models temp_metrics contains",
        default=1,
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        help="The number of rounds each model was evaluated on.",
        default=1,
    )

    args = parser.parse_args()
    num_models = args.num_models
    num_rounds = args.num_rounds

    if num_models <= 0 or num_rounds <= 0:
        print("ERROR: the values must be positives")
        exit(1)

    metrics_record = AggregatedFederatedMetricsRecorder(
        num_models, num_rounds, CLIENTS_NAMES, METRICS_NAMES
    )

    # current_model = 1
    # with open(TEMP_METRICS_PATH, "r") as file:
    #     for line in file:
    #         if line.strip() == "":
    #             current_model += 1
    #             continue

    #         data: Dict = json.loads(line)
    #         server_round, client = next(iter(data.items()))
    #         client_name, metrics = next(iter(client.items()))

    #         metrics_record.add(
    #             f"model_{current_model}", f"round_{server_round}", client_name, metrics
    #         )

    # with open(METRICS_FILE_PATH, "w") as file:
    #     json.dump(metrics_record.get(), file, indent=4)
