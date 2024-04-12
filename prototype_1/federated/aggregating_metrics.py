import argparse
import json
from typing import Dict, List

from .federated_helpers import TEMP_METRICS_FILE_PATH, METRICS_FILE_PATH
from ..pre_process import CLIENTS_NAMES, METRICS_NAMES


class FederatedMetricsRecorder:
    def __init__(
        self,
        num_models: int,
        num_rounds: int,
        client_names: List[str],
        metrics_names: List[str],
    ) -> None:
        self.client_names = client_names
        self.metrics_names = metrics_names
        self.models = list(range(1, num_models + 1))
        self.rounds = list(range(1, num_rounds + 1))
        self._metrics = {
            f"model_{num_model}": {
                f"round_{num_round}": {
                    client_name: {metric_name: [] for metric_name in self.metrics_names}
                    for client_name in self.client_names
                }
                for num_round in self.rounds
            }
            for num_model in self.models
        }

    def add(self, num_model, num_round, name, values):
        self._metrics[num_model][num_round][name] = values

    def get(self):
        return self._metrics


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

    metrics_record = FederatedMetricsRecorder(
        num_models, num_rounds, CLIENTS_NAMES, METRICS_NAMES
    )

    current_model = 1
    with open(TEMP_METRICS_FILE_PATH, "r") as file:
        for line in file:
            if line.strip() == "":
                current_model += 1
                continue

            data: Dict = json.loads(line)
            server_round, client = next(iter(data.items()))
            client_name, metrics = next(iter(client.items()))

            metrics_record.add(
                f"model_{current_model}", f"round_{server_round}", client_name, metrics
            )

    with open(METRICS_FILE_PATH, "w") as file:
        json.dump(metrics_record.get(), file, indent=4)
