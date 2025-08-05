import flwr as fl
import torch
from dataset import load_dataset, get_num_classes
from model import FNN
from FedStrategy import FedSGDStrategy
from init_model_weights import model_init_fn

NUM_CLASSES = get_num_classes()

strategy = FedSGDStrategy(model_init_fn=model_init_fn, lr=0.01)

print("🚀 Starting Flower Server (FedSGD)...")
fl.server.start_server(server_address="127.0.0.1:9091", strategy=strategy, config=fl.server.ServerConfig(num_rounds=10))
