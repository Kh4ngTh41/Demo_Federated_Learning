import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from dataset import load_partition
from model import FNN

client_id = int(sys.argv[1])
trainset, testset = load_partition(client_id, noniid=True, alpha=0.5)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32)

x_example, y_example = trainset[0]
input_size = x_example.shape[0]
num_classes = len(torch.unique(torch.tensor([y for _, y in trainset])))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FNN(input_size=input_size, hidden_size1=128, hidden_size2=64, num_classes=num_classes).to(device)

def train_one_step_get_grad(model, loader):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1.0)
    optimizer.zero_grad()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    grads = [p.grad.cpu().numpy() for p in model.parameters()]
    return grads

def evaluate_model(model, dataloader):
    model.eval()
    correct, total, loss_total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_total += criterion(out, y).item() * y.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_total / total, correct / total

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config): 
        return [p.cpu().numpy() for p in model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = list(model.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        grads = train_one_step_get_grad(model, trainloader)
        # tạm gửi lại trọng số (server sẽ tính chênh lệch làm gradient)
        return self.get_parameters(config), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = evaluate_model(model, testloader)
        print(f"[Client {client_id}] Accuracy={acc:.4f}")
        return loss, len(testloader.dataset), {"test_accuracy": acc}

if __name__ == "__main__":
    print(f"[Client {client_id}] Connecting to server...")
    fl.client.start_numpy_client(server_address="127.0.0.1:9091", client=FlowerClient())
