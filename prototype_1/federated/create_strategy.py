from abc import abstractmethod, ABC
from typing import Callable
from federated.strategies import (
    FedAvgWithFederatedEvaluation,
    FedProxWithFederatedEvaluation,
    FedAdagradWithFederatedEvaluation,
    FedYogiWithFederatedEvaluation,
    FedAdamWithFederatedEvaluation,
    FedMedianWithFederatedEvaluation,
)


class FederatedStrategy(ABC):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def create_strategy(self, on_federated_evaluation_results: Callable) -> object: ...


class FedProxStrategy(FederatedStrategy):
    def create_strategy(self, on_federated_evaluation_results):
        return FedProxWithFederatedEvaluation(
            on_federated_evaluation_results=on_federated_evaluation_results,
            **self.kwargs,
        )


class FedAvgStrategy(FederatedStrategy):
    def create_strategy(self, on_federated_evaluation_results):
        return FedAvgWithFederatedEvaluation(
            on_federated_evaluation_results=on_federated_evaluation_results,
            **self.kwargs,
        )


class FedAdagradStrategy(FederatedStrategy):
    def create_strategy(self, on_federated_evaluation_results) -> object:
        return FedAdagradWithFederatedEvaluation(
            on_federated_evaluation_results=on_federated_evaluation_results,
            **self.kwargs,
        )


class FedYogiStrategy(FederatedStrategy):
    def create_strategy(self, on_federated_evaluation_results) -> object:
        return FedYogiWithFederatedEvaluation(
            on_federated_evaluation_results=on_federated_evaluation_results,
            **self.kwargs,
        )


class FedAdamStrategy(FederatedStrategy):
    def create_strategy(self, on_federated_evaluation_results) -> object:
        return FedAdamWithFederatedEvaluation(
            on_federated_evaluation_results=on_federated_evaluation_results,
            **self.kwargs,
        )


class FedMedianStrategy(FederatedStrategy):
    def create_strategy(self, on_federated_evaluation_results) -> object:
        return FedMedianWithFederatedEvaluation(
            on_federated_evaluation_results=on_federated_evaluation_results,
            **self.kwargs,
        )


def create_federated_strategy(algorithm_name: str, **kwargs) -> FederatedStrategy:
    strategies = {
        "fedprox": FedProxStrategy,
        "fedavg": FedAvgStrategy,
        "fedadagrad": FedAdagradStrategy,
        "fedadam": FedAdamStrategy,
        "fedyogi": FedYogiStrategy,
        "fedmedian": FedYogiStrategy,
    }

    strategy_class = strategies.get(algorithm_name)
    if strategy_class:
        return strategy_class(**kwargs)
    raise ValueError("Invalid federated strategy")
