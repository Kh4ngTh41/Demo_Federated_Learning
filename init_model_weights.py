from model import FNN  # Import model
from dataset import load_dataset
def model_init_fn():
    trainset, _, num_classes = load_dataset(k_features=30)
    input_dim = trainset.tensors[0].shape[1]
    model = FNN(input_size=input_dim, hidden_size1=128, hidden_size2=64, num_classes=num_classes)
    return [p.detach().numpy() for p in model.state_dict().values()]
