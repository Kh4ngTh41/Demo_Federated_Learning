from flwr.server.strategy import FedAvg
import flwr as fl
import numpy as np

def aggregate_metrics(metrics):
    accuracies = [m[1]["test_accuracy"] for m in metrics]
    return {"test_accuracy": float(np.mean(accuracies))}

strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=aggregate_metrics
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
