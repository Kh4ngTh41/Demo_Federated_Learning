import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, Subset
from collections import Counter

# ✅ Trả về số lớp chuẩn
def get_num_classes():
    df = pd.read_csv("IoTDIAD.csv")
    labels = LabelEncoder().fit_transform(df["label"])
    return len(np.unique(labels))

# ✅ Load dataset chuẩn
def load_dataset(k_features=30, test_size=0.2, random_state=42):
    df = pd.read_csv("IoTDIAD.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.sort_values(by='Timestamp').reset_index(drop=True)

    # Trích xuất đặc trưng thời gian
    for col in ["Year","Month","Day","Hour","Minute","Second"]:
        df[col] = getattr(df["Timestamp"].dt, col.lower())
    df.drop(columns=["Timestamp"], inplace=True)

    # Mã hóa label
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])

    # Encode Flow ID, IP
    for col in ["Flow ID", "Src IP", "Dst IP"]:
        df[col] = df[col].astype("category").cat.codes

    X = df.drop(columns=['label']).values
    y = df['label'].values
    X = np.nan_to_num(X)
    X = np.clip(X, -1e10, 1e10)

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Chuẩn hóa
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ANOVA chọn đặc trưng
    selector = SelectKBest(f_classif, k=min(k_features, X.shape[1]))
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    trainset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    testset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.long))
    return trainset, testset, len(np.unique(y))

# ✅ Partition Non-IID với Dirichlet
def partition_noniid(trainset, num_clients=5, num_classes=None, alpha=1, seed=42):
    np.random.seed(seed)
    labels = np.array([y.item() for _, y in trainset])
    idxs = np.arange(len(labels))
    if num_classes is None:
        num_classes = len(np.unique(labels))
    class_indices = [idxs[labels == c] for c in range(num_classes)]
    
    client_dict = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        split_points = (np.cumsum(proportions) * len(class_indices[c])).astype(int)
        split_class = np.split(class_indices[c], split_points[:-1])
        for cid, idx_split in enumerate(split_class):
            client_dict[cid].extend(idx_split)

    # 📌 Thống kê dữ liệu cho từng client
    print("\n📊 [Non-IID Partition Statistics]")
    for cid in client_dict:
        client_labels = labels[client_dict[cid]]
        unique, counts = np.unique(client_labels, return_counts=True)
        total = len(client_labels)
        dist = {int(u): round(c/total, 3) for u, c in zip(unique, counts)}
        print(f"Client {cid}: {total} samples | class distribution: {dist}")

    for cid in client_dict:
        np.random.shuffle(client_dict[cid])
    return client_dict


# ✅ Load partition theo client
def load_partition(client_id, num_clients=5, k_features=30, noniid=False, alpha=0.5):
    trainset, testset, num_classes = load_dataset(k_features=k_features)
    
    if noniid:
        parts = partition_noniid(trainset, num_clients=num_clients, num_classes=num_classes, alpha=alpha)
        client_trainset = Subset(trainset, parts[client_id])
    else:
        all_idx = np.arange(len(trainset))
        split = np.array_split(all_idx, num_clients)
        client_trainset = Subset(trainset, split[client_id])

    # ✅ Đếm số lượng record và tỷ lệ class
    labels = [trainset[i][1].item() for i in client_trainset.indices]
    class_dist = Counter(labels)
    total = len(labels)
    print(f"\n📊 [Client {client_id}] Records: {total}")
    for c, count in class_dist.items():
        print(f"  - Class {c}: {count} ({count/total:.2%})")
    
    return client_trainset, testset


