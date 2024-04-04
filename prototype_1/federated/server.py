import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple


def weighted_average(
    metrics: List[Tuple[int, Metrics]]
) -> Metrics:
    
    accuracies = [num_examples * float(m['accuracy']) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
