"""
测井岩性识别项目 - 经典深度学习模型 (1D-CNN + BiLSTM + Attention)
状态：使用纯净 8 特征增强数据，保留 5 分类（不合并类别）。
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==========================================
# 1. 定义经典 Attention 模块
# ==========================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, lstm_outputs):
        attn_weights = self.attention(lstm_outputs)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1)
        return context_vector, attn_weights

# ==========================================
# 2. 定义经典 LithologyNet (CNN + BiLSTM + Attention)
# ==========================================
class LithologyNet(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout_rate=0.4):
        super(LithologyNet, self).__init__()
        # 1D-CNN 提取局部特征
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # BiLSTM 提取全局序列特征
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True, bidirectional=True)

        # Temporal Attention 聚焦关键序列点
        self.attention = TemporalAttention(hidden_dim * 2)

        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 转换形状适配 CNN: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        c_out = self.relu(self.bn1(self.conv1(x)))
        c_out = self.pool(c_out)

        # 转换形状适配 LSTM: (batch, features, new_seq_len) -> (batch, new_seq_len, features)
        c_out = c_out.transpose(1, 2)

        lstm_out, _ = self.lstm(c_out)
        context, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits

def plot_training_curves(train_losses, test_f1_scores, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_f1_scores, 'r-', label='Test Macro-F1')
    plt.title('Test Macro-F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Macro-F1')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'dl_training_curves.png'), dpi=300)

# ==========================================
# 3. 训练与评估流程
# ==========================================
def train_and_evaluate_dl():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 正在使用设备: {device}")

    # 1. 加载最新的 3D 数据张量
    data_dir = "../data/processed/dl_data"
    X_train = np.load(os.path.join(data_dir, 'X_train_dl.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_dl.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_dl.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_dl.npy'))

    # ==========================================
    # 🌟 新增: 动态特征过滤逻辑
    # ==========================================
    # 假设这是你构建 3D 数组时所用的原始特征顺序，必须与预处理时的顺序完全一致！
    original_feature_cols = ['_CAL', '_GR', '_SP', '_LLD', '_LLS', '_AC', '_DEN', '_PEF']
    selected_features_path = "../reports/figures/selected_features.txt"  # 替换为你实际保存txt的路径

    if os.path.exists(selected_features_path):
        print(f"\n🔍 找到特征选择名单: {selected_features_path}")
        with open(selected_features_path, 'r') as f:
            selected_features = [line.strip() for line in f.readlines() if line.strip()]

        # 找出选中特征在原数组特征维度上的索引
        selected_indices = [original_feature_cols.index(feat) for feat in selected_features if
                            feat in original_feature_cols]

        print(f"✅ 原始特征数: {len(original_feature_cols)} -> 筛选后特征数: {len(selected_indices)}")
        print(f"✅ 选中的特征为: {selected_features}")

        # 使用 numpy 切片在第三个维度(特征维度)进行过滤
        X_train = X_train[:, :, selected_indices]
        X_test = X_test[:, :, selected_indices]
    else:
        print(f"\n⚠️ 未找到特征选择名单 ({selected_features_path})，将使用全部原始特征进行训练。")
    # ==========================================

    # 获取真实的类别名称
    try:
        le = joblib.load('../data/processed/ml_data/label_encoder.pkl')
        class_names = list(le.classes_)
    except:
        class_names = [f'Class {i}' for i in range(5)]

    # 计算类别权重 (虽然我们做了 CTGAN 增强，但切分滑动窗口后可能会有微小不平衡，加权重更保险)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 转换为 PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # 使用较大的 batch_size 以加快大量数据的训练速度
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    num_features = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    print(f"📐 模型输入特征数: {num_features}, 输出类别数: {num_classes} (坚持 5 分类)")

    # 初始化模型、损失函数和优化器
    model = LithologyNet(num_features=num_features, hidden_dim=64, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 100
    print(f"\n🔥 开始训练经典 DL 模型...")

    best_f1 = 0.0
    model_save_dir = "../saved_models/deep_learning"
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "classic_cnn_lstm_attention.pth")

    history_train_loss = []
    history_test_f1 = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        current_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        history_test_f1.append(current_macro_f1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_acc = accuracy_score(all_labels, all_preds)
            print(f"Epoch [{epoch+1:03d}/{epochs}] | Loss: {avg_train_loss:.4f} | Test Acc: {current_acc:.4f} | Test F1: {current_macro_f1:.4f}")

        if current_macro_f1 > best_f1:
            best_f1 = current_macro_f1
            torch.save(model.state_dict(), best_model_path)

    print(f"\n✅ 训练完毕！最佳 Macro-F1 分数为: {best_f1:.4f}")

    # ==========================================
    # 4. 最终评估与保存图表
    # ==========================================
    report_dir = "../reports/figures"
    plot_training_curves(history_train_loss, history_test_f1, report_dir)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    final_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())

    print("\n[经典架构: CNN + BiLSTM + Attention] 终极分类报告:")
    print("="*60)
    report_str = classification_report(y_test, final_preds, target_names=class_names)
    print(report_str)

    final_acc = accuracy_score(y_test, final_preds)

    # 保存文字报告
    with open(os.path.join(report_dir, 'classic_dl_report.txt'), 'w', encoding='utf-8') as f:
        f.write("[经典架构: CNN + BiLSTM + Attention] 分类报告:\n" + "="*60 + "\n" + report_str)
        f.write(f"\n🎯 最终 Accuracy 分数: {final_acc:.4f}\n")
        f.write(f"🎯 最终 Macro-F1 分数: {best_f1:.4f}\n")

    # 混淆矩阵
    cm = confusion_matrix(y_test, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Classic DL Confusion Matrix (F1: {best_f1:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "dl_confusion_matrix.png"), dpi=300)
    print(f"📊 混淆矩阵与文字报告已保存至 {report_dir}")

if __name__ == "__main__":
    try:
        train_and_evaluate_dl()
    except Exception as e:
        print(f"❌ 发生错误: {e}")