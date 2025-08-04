import torch
from model import FNN

def init_model_weights():
    model = FNN(input_size=30, hidden_size1=128, hidden_size2=64, num_classes=2)
    return [val.detach().numpy() for _, val in model.state_dict().items()]
