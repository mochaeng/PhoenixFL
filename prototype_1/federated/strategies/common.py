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
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import Metrics
from flwr.server.client_proxy import ClientProxy
from overrides import override
from typing import Callable, Dict, List, Tuple, Optional, Union


FederatedEvalutionDataFn = Callable[[int, List[Tuple[int, Metrics]]], None]
FederatedEvalutionForFedPlusDataFn = Callable[[int, List[Tuple[str, Metrics]]], None]


class FedAvgWithFederatedEvaluation(fl.server.strategy.FedAvg):
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
        on_federated_evaluation_results: FederatedEvalutionDataFn,
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

    @override
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        print()

        federated_evaluation_results = [
            (int(client.cid), res.metrics) for client, res in results
        ]
        self.on_federated_evaluation_results(server_round, federated_evaluation_results)

        return super().aggregate_evaluate(server_round, results, failures)


class FedProxWithFederatedEvaluation(fl.server.strategy.FedProx):
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
        proximal_mu: float,
        on_federated_evaluation_results: FederatedEvalutionDataFn,
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
            proximal_mu=proximal_mu,
        )
        self.on_federated_evaluation_results = on_federated_evaluation_results

    def __repr__(self) -> str:
        rep = f"FedProxWithFederatedEvaluation(accept_failures={self.accept_failures})"
        return rep

    @override
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        federated_evaluation_results = [
            (int(client.cid), res.metrics) for client, res in results
        ]
        self.on_federated_evaluation_results(server_round, federated_evaluation_results)

        return super().aggregate_evaluate(server_round, results, failures)


class FedAdagradWithFederatedEvaluation(fl.server.strategy.FedAdagrad):
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
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        tau: float = 1e-9,
        on_federated_evaluation_results: FederatedEvalutionDataFn,
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
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            eta=eta,
            eta_l=eta_l,
            tau=tau,
        )
        self.on_federated_evaluation_results = on_federated_evaluation_results

    def __repr__(self) -> str:
        rep = (
            f"FedAdagradWithFederatedEvaluation(accept_failures={self.accept_failures})"
        )
        return rep

    @override
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        federated_evaluation_results = [
            (int(client.cid), res.metrics) for client, res in results
        ]
        self.on_federated_evaluation_results(server_round, federated_evaluation_results)

        return super().aggregate_evaluate(server_round, results, failures)


class FedAdamWithFederatedEvaluation(fl.server.strategy.FedAdam):
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
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-9,
        on_federated_evaluation_results: FederatedEvalutionDataFn,
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
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
        )
        self.on_federated_evaluation_results = on_federated_evaluation_results

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAdamWithFederatedEvaluation(accept_failures={self.accept_failures})"
        return rep

    @override
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        federated_evaluation_results = [
            (int(client.cid), res.metrics) for client, res in results
        ]
        self.on_federated_evaluation_results(server_round, federated_evaluation_results)

        return super().aggregate_evaluate(server_round, results, failures)

    @override
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training

        Sends tau, eta and eta_l constants for fedadam
        """

        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Return client/config pairs with the delta added
        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {
                        **fit_ins.config,
                        "eta_l": self.eta_l,
                    },
                ),
            )
            for client, fit_ins in client_config_pairs
        ]


class FedYogiWithFederatedEvaluation(fl.server.strategy.FedYogi):
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
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-2,
        eta_l: float = 0.0316,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
        on_federated_evaluation_results: FederatedEvalutionDataFn,
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
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
        )
        self.on_federated_evaluation_results = on_federated_evaluation_results

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedYogiWithFederatedEvaluation(accept_failures={self.accept_failures})"
        return rep

    @override
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        federated_evaluation_results = [
            (int(client.cid), res.metrics) for client, res in results
        ]
        self.on_federated_evaluation_results(server_round, federated_evaluation_results)

        return super().aggregate_evaluate(server_round, results, failures)

    @override
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training

        Sends tau, eta and eta_l constants for fedadam
        """

        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Return client/config pairs with the delta added
        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {
                        **fit_ins.config,
                        "eta_l": self.eta_l,
                    },
                ),
            )
            for client, fit_ins in client_config_pairs
        ]


class FedMedianWithFederatedEvaluation(fl.server.strategy.FedMedian):
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
        on_federated_evaluation_results: FederatedEvalutionDataFn,
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
        """Compute a string representation of the strategy."""
        rep = (
            f"FedMedianWithFederatedEvaluation(accept_failures={self.accept_failures})"
        )
        return rep

    @override
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        federated_evaluation_results = [
            (int(client.cid), res.metrics) for client, res in results
        ]
        self.on_federated_evaluation_results(server_round, federated_evaluation_results)

        return super().aggregate_evaluate(server_round, results, failures)


class FedTrimmedAvgWithFederatedEvaluation(fl.server.strategy.FedTrimmedAvg):
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
        beta: float = 0.2,
        on_federated_evaluation_results: FederatedEvalutionDataFn,
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
            beta=beta,
        )
        self.on_federated_evaluation_results = on_federated_evaluation_results

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedTrimmedAvgEvaluation(accept_failures={self.accept_failures})"
        return rep

    @override
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        federated_evaluation_results = [
            (int(client.cid), res.metrics) for client, res in results
        ]
        self.on_federated_evaluation_results(server_round, federated_evaluation_results)
        return super().aggregate_evaluate(server_round, results, failures)


class FeadAvgPlusWithFederatedEvaluation(fl.server.strategy.FedAvg):
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
        on_federated_evaluation_results: FederatedEvalutionForFedPlusDataFn,
        smooth_delta_approximation: float = 0.1,
        lambda_value: float = 0.0,
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
        self.smooth_delta = smooth_delta_approximation
        self.lambda_value = lambda_value

    @override
    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvgPlusWithEvaluation(accept_failures={self.accept_failures})"
        return rep

    @override
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training

        Sends the smooth constant approximation (delta) from the Moreau Envelope
        """

        # Get the standard client/config pairs from the FedAvg super-class
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Return client/config pairs with the delta added
        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {
                        **fit_ins.config,
                        "smooth_delta": self.smooth_delta,
                        "lambda": self.lambda_value,
                    },
                ),
            )
            for client, fit_ins in client_config_pairs
        ]

    @override
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        federated_evaluation_results = [
            (client.cid, res.metrics) for client, res in results
        ]
        self.on_federated_evaluation_results(server_round, federated_evaluation_results)
        return super().aggregate_evaluate(server_round, results, failures)
