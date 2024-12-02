from abc import ABC, abstractmethod
from typing import Callable

import flwr as fl

from federated.strategies.common import (
    FeadAvgPlusWithFederatedEvaluation,
    FedAdagradWithFederatedEvaluation,
    FedAdamWithFederatedEvaluation,
    FedAvgWithFederatedEvaluation,
    FedMedianWithFederatedEvaluation,
    FedProxWithFederatedEvaluation,
    FedTrimmedAvgWithFederatedEvaluation,
    FedYogiWithFederatedEvaluation,
)


class FederatedStrategy(ABC):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def create_strategy(
        self, on_federated_evaluation_results: Callable
    ) -> "FederatedStrategy": ...


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
    def create_strategy(self, on_federated_evaluation_results):
        return FedAdagradWithFederatedEvaluation(
            on_federated_evaluation_results=on_federated_evaluation_results,
            **self.kwargs,
        )


class FedYogiStrategy(FederatedStrategy):
    def create_strategy(self, on_federated_evaluation_results):
        return FedYogiWithFederatedEvaluation(
            on_federated_evaluation_results=on_federated_evaluation_results,
            **self.kwargs,
        )


class FedAdamStrategy(FederatedStrategy):
    def create_strategy(self, on_federated_evaluation_results):
        return FedAdamWithFederatedEvaluation(
            on_federated_evaluation_results=on_federated_evaluation_results,
            **self.kwargs,
        )


class FedMedianStrategy(FederatedStrategy):
    def create_strategy(self, on_federated_evaluation_results):
        return FedMedianWithFederatedEvaluation(
            on_federated_evaluation_results=on_federated_evaluation_results,
            **self.kwargs,
        )


class FedTrimmedAvgStrategy(FederatedStrategy):
    def create_strategy(self, on_federated_evaluation_results):
        return FedTrimmedAvgWithFederatedEvaluation(
            on_federated_evaluation_results=on_federated_evaluation_results,
            **self.kwargs,
        )


class FedAvgPlusStrategy(FederatedStrategy):
    def create_strategy(self, on_federated_evaluation_results):
        return FeadAvgPlusWithFederatedEvaluation(
            on_federated_evaluation_results=on_federated_evaluation_results,
            **self.kwargs,
        )


def create_federated_strategy(algorithm_name: str, **kwargs) -> FederatedStrategy:
    strategies = {
        "fedprox": FedProxStrategy,
        "fedavg": FedAvgStrategy,
        "fedavgplus": FedAvgPlusStrategy,
        "fedadagrad": FedAdagradStrategy,
        "fedadam": FedAdamStrategy,
        "fedyogi": FedYogiStrategy,
        "fedmedian": FedMedianStrategy,
        "fedtrimmed": FedTrimmedAvgStrategy,
    }

    strategy_class = strategies.get(algorithm_name)
    if strategy_class:
        return strategy_class(**kwargs)
    raise ValueError("Invalid federated strategy")
