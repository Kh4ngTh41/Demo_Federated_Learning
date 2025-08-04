import flwr as fl
from FedStrategy import FedSGDStrategy
from init_model_weights import init_model_weights

print("🚀 Starting Flower Server (FedSGD)...")
strategy = FedSGDStrategy(model_init_fn=init_model_weights, lr=0.01)

fl.server.start_server(
    server_address="127.0.0.1:9091",
    config=fl.server.ServerConfig(num_rounds=3,round_timeout=None),
    strategy=strategy
)
