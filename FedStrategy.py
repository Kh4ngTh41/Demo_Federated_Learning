import flwr as fl
import numpy as np
from typing import List, Tuple
from flwr.common import FitRes, EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters
import time
# ✅ FedSGD Strategy với LR cố định
class FedSGDStrategy(fl.server.strategy.Strategy):
    def __init__(self, model_init_fn, lr=0.01):
        super().__init__()
        self.model_params = model_init_fn()
        self.lr = lr

    def num_fit_clients(self, num_available_clients: int) -> int:
        return num_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> int:
        return num_available_clients

    def accept_failures(self) -> bool:
        return True

    def initialize_parameters(self, client_manager):
        return ndarrays_to_parameters(self.model_params)

    def configure_fit(self, server_round, parameters, client_manager):
    # ✅ Chờ cho đến khi có ít nhất 1 client
        while client_manager.num_available() == 0:
            print("[Server] Waiting for clients to connect...")
            time.sleep(2)  # chờ 2 giây rồi kiểm tra lại
        
        clients = client_manager.sample(num_clients=self.num_fit_clients(client_manager.num_available()))
        return [(c, fl.common.FitIns(parameters, {})) for c in clients]

    def aggregate_fit(self, server_round, results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], failures):
        if not results:
            return None, {}
        grads, sizes = [], []
        global_w = parameters_to_ndarrays(results[0][1].parameters)
        for _, fit_res in results:
            local_w = parameters_to_ndarrays(fit_res.parameters)
            sizes.append(fit_res.num_examples)
            grads.append([lw - gw for lw, gw in zip(local_w, global_w)])
        total = sum(sizes)
        avg_grad = [sum(g[i] * (n / total) for g, n in zip(grads, sizes)) for i in range(len(global_w))]
        new_w = [gw + self.lr * ag for gw, ag in zip(global_w, avg_grad)]
        self.model_params = new_w
        return ndarrays_to_parameters(new_w), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=self.num_evaluation_clients(client_manager.num_available()))
        return [(c, fl.common.EvaluateIns(parameters, {})) for c in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return 0.0, {}

        metrics_all = {}
        for _, res in results:
            for k, v in res.metrics.items():
                metrics_all.setdefault(k, []).append(v)

        # ✅ Trung bình các metrics
        metrics_avg = {k: float(np.mean(v)) for k, v in metrics_all.items()}

        # ✅ Log tất cả metrics
        print(f"\n🌍 [Round {server_round}] Evaluation Metrics:")
        for k, v in metrics_avg.items():
            print(f"  {k}: {v:.4f}")

        return metrics_avg.get("test_accuracy", 0.0), metrics_avg



    def evaluate(self, server_round, parameters):
        return None
