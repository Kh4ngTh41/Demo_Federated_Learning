import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl

from dataset import load_partition
from model import FNN

# ======== Nhận ID client từ command line ========
client_id = int(sys.argv[1])  # VD: python client.py 0

# ======== Load dữ liệu tương ứng với client ========
trainset = load_partition(client_id)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# ======== Xác định input size và số lớp đầu ra ========
x_example, y_example = trainset[0]
input_size = x_example.shape[0]
num_classes = len(torch.unique(torch.tensor([y for _, y in trainset])))

# ======== Khởi tạo mô hình và device ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FNN(input_size=input_size, hidden_size1=128, hidden_size2=64, num_classes=num_classes).to(device)

# ======== Hàm huấn luyện ========
def train(model, loader, epochs=200):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
def evaluate_model(model, dataloader):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        correct, total, loss_total = 0, 0, 0.0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                loss_total += loss.item() * y.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        avg_loss = loss_total / total
        accuracy = correct / total
        return avg_loss, accuracy
# ======== Client Flower ========
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(model, trainloader)
        return self.get_parameters(config), len(trainloader.dataset), {}

    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        test_loss, test_acc = evaluate_model(model, trainloader)
        print(f"[Client {client_id}] Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        return test_loss, len(trainloader.dataset), {"test_accuracy": test_acc}





# ======== Khởi động client ========
fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())
