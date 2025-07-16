import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, Subset

def load_dataset(k_features=30):
    df_path = "IoTDIAD.csv"
    df = pd.read_csv(df_path)

    # Chuyển đổi và trích xuất thời gian
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.sort_values(by='Timestamp').reset_index(drop=True)
    df["Year"] = df["Timestamp"].dt.year
    df["Month"] = df["Timestamp"].dt.month
    df["Day"] = df["Timestamp"].dt.day
    df["Hour"] = df["Timestamp"].dt.hour
    df["Minute"] = df["Timestamp"].dt.minute
    df["Second"] = df["Timestamp"].dt.second
    df.drop(columns=["Timestamp"], inplace=True)

    # Mã hóa nhãn
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])

    # Mã hóa các cột định danh
    for col in ["Flow ID", "Src IP", "Dst IP"]:
        df[col] = df[col].astype("category").cat.codes

    # Xử lý NaN và vô hạn
    X = df.drop(columns=['label']).values
    y = df['label'].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X = np.clip(X, -1e10, 1e10)

    # Chia train/test tạm thời để chuẩn hóa
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Chuẩn hóa và chọn đặc trưng
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    selector = SelectKBest(score_func=f_classif, k=min(k_features, X.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)
    selected_columns = [i for i, b in enumerate(selector.get_support()) if b]
    print(f"[INFO] Selected features (columns): {selected_columns}")

    # Chuyển về tensor
    x_tensor = torch.tensor(X_selected, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return TensorDataset(x_tensor, y_tensor)

def load_partition(client_id: int, num_clients: int = 5):
    dataset = load_dataset()

    total_size = len(dataset)
    indices = np.arange(total_size)
    np.random.seed(42)
    np.random.shuffle(indices)

    part_size = total_size // num_clients
    start = client_id * part_size
    end = (client_id + 1) * part_size
    subset = Subset(dataset, indices[start:end])
    return subset
