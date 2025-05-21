import flwr as fl
import logging
from typing import List, Tuple, Optional
import numpy as np
from flwr.common import Metrics
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from flwr.common.logger import log

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomFedAvg(FedAvg):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.current_round = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[ClientProxy, fl.common.FitRes]],
    ) -> Optional[fl.common.Parameters]:
        """Aggregate model weights and log training progress."""
        self.current_round = server_round
        logging.info(f"\nRound {server_round} - Aggregating fit results from {len(results)} clients")
        
        # Log individual client results
        for client, fit_res in results:
            metrics = fit_res.metrics
            if metrics:
                logging.info(f"Client {client.cid} - Training loss: {metrics.get('train_loss', 'N/A'):.4f}")
        
        # Aggregate weights
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            logging.info(f"Round {server_round} - Successfully aggregated model weights")
        else:
            logging.warning(f"Round {server_round} - Failed to aggregate model weights")
        
        return aggregated_parameters

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
    ) -> Tuple[Optional[float], Optional[Metrics]]:
        """Aggregate evaluation results and log progress."""
        logging.info(f"\nRound {server_round} - Aggregating evaluation results from {len(results)} clients")
        
        # Log individual client results
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        
        for client, eval_res in results:
            metrics = eval_res.metrics
            if metrics:
                client_loss = eval_res.loss
                client_accuracy = metrics.get("accuracy", 0.0)
                client_samples = eval_res.num_examples
                
                total_loss += client_loss * client_samples
                total_accuracy += client_accuracy * client_samples
                total_samples += client_samples
                
                logging.info(
                    f"Client {client.cid} - "
                    f"Loss: {client_loss:.4f}, "
                    f"Accuracy: {client_accuracy:.4f}, "
                    f"Samples: {client_samples}"
                )
        
        if total_samples > 0:
            aggregated_loss = total_loss / total_samples
            aggregated_accuracy = total_accuracy / total_samples
            logging.info(
                f"Round {server_round} - "
                f"Aggregated Loss: {aggregated_loss:.4f}, "
                f"Aggregated Accuracy: {aggregated_accuracy:.4f}"
            )
            return aggregated_loss, {"accuracy": aggregated_accuracy}
        else:
            logging.warning(f"Round {server_round} - No evaluation results to aggregate")
            return None, None

def main():
    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=3,  # Minimum number of clients for training
        min_evaluate_clients=3,  # Minimum number of clients for evaluation
        min_available_clients=3,  # Minimum number of clients to start training
        evaluate_metrics_aggregation_fn=weighted_average,  # Use weighted average for metrics
    )

    # Start server
    logging.info("Starting Flower server...")
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(
            num_rounds=2000,
            round_timeout=600.0,  # 10 minutes timeout per round
        ),
        strategy=strategy
    )

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics using weighted average."""
    # Calculate the weighted average of metrics
    total_examples = sum([num_examples for num_examples, _ in metrics])
    weighted_metrics = {}
    
    for num_examples, client_metrics in metrics:
        for metric_name, metric_value in client_metrics.items():
            if metric_name not in weighted_metrics:
                weighted_metrics[metric_name] = 0.0
            weighted_metrics[metric_name] += metric_value * (num_examples / total_examples)
    
    return weighted_metrics

if __name__ == "__main__":
    main()
