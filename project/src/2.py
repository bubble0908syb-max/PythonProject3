import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ---------------------------------------------------------
# 1. 定义 ViT-1D 模型架构 (需与你之前的架构一致)
# ---------------------------------------------------------
class ViT1D(nn.Module):
    def __init__(self, seq_len, num_features, num_classes, d_model=128, nhead=8, num_layers=3, dim_feedforward=256,
                 dropout=0.3):
        super(ViT1D, self).__init__()
        # 特征映射到 d_model 维度
        self.embedding = nn.Linear(num_features, d_model)

        # 绝对位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = x + self.pos_encoder  # 添加位置编码

        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # 取序列的平均值作为整条序列的表示 (Global Average Pooling)
        x = x.mean(dim=1)  # (batch_size, d_model)

        out = self.fc(x)  # (batch_size, num_classes)
        return out


# ---------------------------------------------------------
# 2. 数据加载与预处理 (针对未增强的数据)
# ---------------------------------------------------------
def load_and_preprocess_data(csv_path, seq_len=10, test_size=0.2):
    print(f"📦 正在加载原始数据: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. 修正目标列名
    target_col = 'Lith_Section'  # 使用你实际的标签列名

    # 2. 严格剔除所有非测井曲线的元数据列
    exclude_cols = ['Well_Name', 'TopDepth', 'BotDepth', 'DEPT', 'Lith_Section', 'Lith_Encoded']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"🔍 提取到的有效特征列 ({len(feature_cols)}个): {feature_cols}")

    # 处理缺失值 (简单的向前/向后填充)
    df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill')

    # 提取特征和标签
    X_raw = df[feature_cols].values
    y_raw = df[target_col].values

    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    print(f"🏷️ 标签类别: {class_names}")

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 划分滑动窗口 (将一维序列转换为 seq_len 长度的样本)
    print(f"🔪 正在按照序列长度 {seq_len} 划分滑动窗口...")
    X_windows = []
    y_windows = []

    # 使用步长为 1 的滑动窗口
    for i in range(len(X_scaled) - seq_len):
        X_windows.append(X_scaled[i: i + seq_len])
        # 假设窗口的标签取窗口中心点的标签或最后一个点的标签
        # 这里取窗口最后一个点作为该窗口的标签
        y_windows.append(y_encoded[i + seq_len - 1])

    X = np.array(X_windows)
    y = np.array(y_windows)

    print(f"📊 窗口划分完成: X shape={X.shape}, y shape={y.shape}")

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    print(f"📈 训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}")

    return X_train, X_val, y_train, y_val, feature_cols, class_names


# ---------------------------------------------------------
# 3. 训练与评估主函数
# ---------------------------------------------------------
def train_and_evaluate():
    # --- 配置参数 ---
    csv_path = "../data/processed/preprocessed_data.csv"  # 原始未增强数据路径
    model_save_path = "../saved_models/deep_learning/vit_1d_original.pth"
    seq_len = 20  # 序列窗口长度 (可根据需要调整)
    batch_size = 64
    epochs = 50
    learning_rate = 1e-4

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")

    # --- 1. 加载数据 ---
    try:
        X_train, X_val, y_train, y_val, feature_cols, class_names = load_and_preprocess_data(csv_path, seq_len=seq_len)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {csv_path}。请确保路径正确。")
        return

    num_features = X_train.shape[2]
    num_classes = len(class_names)

    # 转换为 Tensor
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 2. 初始化模型、损失函数和优化器 ---
    model = ViT1D(seq_len=seq_len, num_features=num_features, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # --- 3. 训练循环 ---
    print("\n" + "=" * 40)
    print("🏃 开始在未增强数据上训练 ViT 模型...")
    print("=" * 40)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"   🌟 发现更好的模型，已保存至: {model_save_path}")

    # --- 4. 最终评估 ---
    print("\n" + "=" * 40)
    print("🏆 训练完成！加载最佳模型进行最终评估...")

    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    final_preds = []
    final_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.numpy())

    print("\n[未增强数据 - 验证集评估报告]")
    print(classification_report(final_labels, final_preds, target_names=class_names, digits=4))

    macro_f1 = f1_score(final_labels, final_preds, average='macro')
    print(f"🎯 最终验证集 Macro-F1: {macro_f1:.4f}")


if __name__ == "__main__":
    train_and_evaluate()