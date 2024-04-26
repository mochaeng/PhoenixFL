from typing import Callable, Dict, List, Tuple, Optional, Union
import flwr as fl
from flwr.common import (
    EvaluateRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    NDArrays,
    FitIns,
)
from flwr.server.client_manager import ClientManager
from flwr.common.typing import Metrics
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common.logger import log
from logging import WARNING


MetricsFederatedEvalutionDataFn = Callable[[int, List[Tuple[int, Metrics]]], None]


class FedAvgWithFederatedEvaluation(fl.server.strategy.FedAvg):
    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        on_federated_evaluation_results: Optional[
            MetricsFederatedEvalutionDataFn
        ] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.on_federated_evaluation_results = on_federated_evaluation_results

    def __repr__(self) -> str:
        rep = f"FedAvgWithFederatedEvaluation(accept_failures={self.accept_failures})"
        return rep

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # results from federated evaluation
        if self.on_federated_evaluation_results:
            federated_evaluation_results = [
                (int(client.cid), res.metrics) for client, res in results
            ]
            self.on_federated_evaluation_results(
                server_round, federated_evaluation_results
            )

        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated


class FedProxWithFederatedEvaluation(FedAvgWithFederatedEvaluation):
    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        on_federated_evaluation_results: Optional[
            MetricsFederatedEvalutionDataFn
        ] = None,
        proximal_mu: float,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_federated_evaluation_results=on_federated_evaluation_results,
        )
        self.proximal_mu = proximal_mu

    def __repr__(self) -> str:
        rep = f"FedProxWithFederatedEvaluation(accept_failures={self.accept_failures})"
        return rep

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config, "proximal_mu": self.proximal_mu},
                ),
            )
            for client, fit_ins in client_config_pairs
        ]


class FederatedStrategy:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def create_strategy(
        self,
        on_federated_evaluation_results: Optional[
            MetricsFederatedEvalutionDataFn
        ] = None,
    ):
        raise NotImplementedError("Subclass must implement create_strategy")


class FedProxStrategy(FederatedStrategy):
    def create_strategy(
        self,
        on_federated_evaluation_results: Optional[
            MetricsFederatedEvalutionDataFn
        ] = None,
    ):
        return FedProxWithFederatedEvaluation(
            on_federated_evaluation_results=on_federated_evaluation_results,
            **self.kwargs,
        )


class FedAvgStrategy(FederatedStrategy):
    def create_strategy(
        self,
        on_federated_evaluation_results: Optional[
            MetricsFederatedEvalutionDataFn
        ] = None,
    ):
        return FedAvgWithFederatedEvaluation(
            on_federated_evaluation_results=on_federated_evaluation_results,
            **self.kwargs,
        )


def create_federated_strategy(algorithm_name: str, **kwargs) -> FederatedStrategy:
    strategies = {
        "fedprox": FedProxStrategy,
        "fedavg": FedAvgStrategy,
    }

    strategy_class = strategies.get(algorithm_name)
    if strategy_class:
        return strategy_class(**kwargs)
    raise ValueError("Invalid federated strategy")
