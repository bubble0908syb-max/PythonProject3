import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ==========================================
# 1. 定义纯 CNN (1D) 模型架构
# ==========================================
class CNN1D(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CNN1D, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # 第二层卷积
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 第三层卷积
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()

        # 全局自适应平均池化，统一序列长度维度
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接分类头
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 输入 x shape: (batch_size, seq_len, num_features)
        # Conv1d 期望输入: (batch_size, in_channels, seq_len)
        x = x.permute(0, 2, 1)

        # 特征提取
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))

        # 池化 & 展平
        x = self.global_pool(x).squeeze(-1)  # shape: (batch_size, 256)

        # 分类预测
        x = self.dropout(torch.relu(self.fc1(x)))
        out = self.fc2(x)
        return out


# ==========================================
# 2. 训练与评估主函数 (直接使用 dl_data)
# ==========================================
def train_and_evaluate_cnn():
    # --- 配置参数 ---
    data_dir = "../data/processed/dl_data"
    le_path = '../data/processed/ml_data/label_encoder.pkl'
    model_save_path = "../saved_models/deep_learning/cnn_1d_model.pth"
    report_dir = "../reports/figures"

    batch_size = 128
    epochs = 50
    learning_rate = 1e-4

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")

    # --- 1. 加载 dl_data 数据 ---
    print("📦 正在从 dl_data 文件夹加载 Numpy 数据...")
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train_dl.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train_dl.npy'))

        # 尝试加载验证集，如果没有单独的验证集则使用测试集作为验证
        try:
            X_val = np.load(os.path.join(data_dir, 'X_val_dl.npy'))
            y_val = np.load(os.path.join(data_dir, 'y_val_dl.npy'))
        except FileNotFoundError:
            print("⚠️ 未找到独立的 X_val_dl.npy，正在使用 X_test_dl.npy 作为验证集...")
            X_val = np.load(os.path.join(data_dir, 'X_test_dl.npy'))
            y_val = np.load(os.path.join(data_dir, 'y_test_dl.npy'))

    except FileNotFoundError as e:
        print(f"❌ 找不到数据文件: {e}")
        return

    # 获取类别名称
    try:
        le = joblib.load(le_path)
        class_names = [str(cls) for cls in le.classes_]
        print(f"✅ 成功加载标签编码器，类别为: {class_names}")
    except FileNotFoundError:
        class_names = [f'Class {i}' for i in range(len(np.unique(y_train)))]
        print(f"⚠️ 未找到标签编码器，使用默认类别名称: {class_names}")

    seq_len = X_train.shape[1]
    num_features = X_train.shape[2]
    num_classes = len(class_names)
    print(f"📊 数据加载完成: X_train shape={X_train.shape}, X_val shape={X_val.shape}")
    print(f"📐 模型输入信息: 序列长度={seq_len}, 特征数={num_features}, 输出类别数={num_classes}")

    # 转换为 Tensor DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 2. 初始化模型 ---
    model = CNN1D(num_features=num_features, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    print("\n" + "=" * 40)
    print("🏃 开始训练 1D-CNN 模型...")
    print("=" * 40)

    best_val_loss = float('inf')

    # --- 3. 训练循环 ---
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

        # --- 验证阶段 ---
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"   🌟 发现更好的模型，已保存权重")

    # ==========================================
    # 4. 最终评估与报告生成
    # ==========================================
    print("\n" + "=" * 40)
    print("🏆 训练完成！加载最佳 CNN 模型进行最终评估...")

    model.load_state_dict(torch.load(model_save_path, weights_only=True))
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

    accuracy = accuracy_score(final_labels, final_preds)
    macro_f1 = f1_score(final_labels, final_preds, average='macro')
    report_str = classification_report(final_labels, final_preds, target_names=class_names, digits=4)

    print("\n[CNN-1D 验证集评估报告]")
    print(report_str)

    # 绘制混淆矩阵
    cm = confusion_matrix(final_labels, final_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12})
    plt.title(f'CNN-1D Evaluation\nAccuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    cm_path = os.path.join(report_dir, "cnn_1d_evaluation_cm.png")
    plt.savefig(cm_path, dpi=300)
    print(f"📊 评估版混淆矩阵已保存至: {cm_path}")

    # 保存文本报告
    report_txt_path = os.path.join(report_dir, "cnn_1d_evaluation_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("🏆 [CNN-1D 验证集评估报告]\n")
        f.write("=" * 60 + "\n")
        f.write(report_str + "\n")
        f.write(f"🎯 整体 Accuracy (准确率) : {accuracy:.4f}\n")
        f.write(f"🎯 整体 Macro-F1 (宏F1)   : {macro_f1:.4f}\n")
        f.write("=" * 60 + "\n")
    print(f"📝 评估报告文本已保存至: {report_txt_path}")


if __name__ == "__main__":
    train_and_evaluate_cnn()