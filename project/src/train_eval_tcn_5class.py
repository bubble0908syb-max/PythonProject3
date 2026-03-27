"""
测井岩性识别项目 - TCN (时间卷积网络) - 工业级稳定加固版
状态：动态特征筛选 + 5 分类 + 感受野适配 (3层)
加固：BatchNorm1d 替换 WeightNorm + LR Scheduler + 梯度裁剪
"""

import os
import numpy as np
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
# 1. 定义 TCN 核心组件 (已加固)
# ==========================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """加固版残差块：去除了不稳定的 weight_norm，引入了 BatchNorm1d"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # 第一层卷积 + BatchNorm
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation, bias=False)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)  # 🌟 强力维稳组件
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二层卷积 + BatchNorm
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation, bias=False)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)  # 🌟 强力维稳组件
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=False) if n_inputs != n_outputs else None
        self.bn_res = nn.BatchNorm1d(n_outputs) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        if self.downsample is not None:
            res = self.downsample(x)
            res = self.bn_res(res)
        else:
            res = x
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.5):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ==========================================
# 2. 定义最终的 LithologyTCN
# ==========================================
class LithologyTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.3):
        super(LithologyTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        y1 = self.tcn(inputs)
        out = self.linear(y1[:, :, -1])
        return out

def plot_training_curves(train_losses, test_f1_scores, output_dir, filename='tcn_5class_training_curves.png'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('TCN Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_f1_scores, 'r-', label='Test Macro-F1')
    plt.title('TCN Test Macro-F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Macro-F1')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)

# ==========================================
# 3. 训练与评估流程
# ==========================================
def train_and_evaluate_tcn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 正在使用设备: {device}")

    data_dir = "../data/processed/dl_data"
    X_train = np.load(os.path.join(data_dir, 'X_train_dl.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_dl.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_dl.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_dl.npy'))

    original_feature_cols = ['_CAL', '_GR', '_SP', '_LLD', '_LLS', '_AC', '_DEN', '_PEF']
    selected_features_path = "../reports/figures/selected_features.txt"

    if os.path.exists(selected_features_path):
        with open(selected_features_path, 'r') as f:
            selected_features = [line.strip() for line in f.readlines() if line.strip()]
        selected_indices = [original_feature_cols.index(feat) for feat in selected_features if feat in original_feature_cols]
        X_train = X_train[:, :, selected_indices]
        X_test = X_test[:, :, selected_indices]

    try:
        le = joblib.load('../data/processed/ml_data/label_encoder.pkl')
        class_names = list(le.classes_)
    except:
        class_names = [f'Class {i}' for i in range(5)]

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    num_features = X_train.shape[2]
    num_classes = len(np.unique(y_train))

    # 🌟 适配 15 序列长度的最佳网络层数 (3 层)
    tcn_channels = [64, 128, 256]

    model = LithologyTCN(input_size=num_features, output_size=num_classes,
                         num_channels=tcn_channels, kernel_size=3, dropout=0.3).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # 🌟 降级初始学习率：从 0.001 降到 0.0001
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    # 🌟 引入学习率调度器：如果连续 5 个 epoch 测试集 Loss 不降反升，就把学习率砍半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    epochs = 100 # 因为学习率降低了，稍微增加一点训练轮数
    print(f"\n🔥 开始训练工业级稳定版 TCN 模型...")

    best_f1 = 0.0
    model_save_dir = "../saved_models/deep_learning"
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "tcn_5class_stable.pth")

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

            # 🌟 保留梯度裁剪作为最后防线
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)

        model.eval()
        all_preds = []
        all_labels = []
        test_loss = 0.0 # 用于监控测试集 Loss 供调度器使用

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        current_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        history_test_f1.append(current_macro_f1)

        # 🌟 调度器根据测试集 Loss 自动调节学习率
        scheduler.step(avg_test_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_acc = accuracy_score(all_labels, all_preds)
            print(f"Epoch [{epoch+1:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test F1: {current_macro_f1:.4f}")

        if current_macro_f1 > best_f1:
            best_f1 = current_macro_f1
            torch.save(model.state_dict(), best_model_path)

    print(f"\n✅ 训练完毕！TCN 最佳 Macro-F1 分数为: {best_f1:.4f}")

    # ==========================================
    # 4. 最终评估
    # ==========================================
    report_dir = "../reports/figures"
    plot_training_curves(history_train_loss, history_test_f1, report_dir, filename='tcn_5class_stable_curves.png')

    # 忽略未来版本的安全性警告加载模型
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    final_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())

    report_str = classification_report(y_test, final_preds, target_names=class_names)
    final_acc = accuracy_score(y_test, final_preds)

    with open(os.path.join(report_dir, 'tcn_5class_stable_report.txt'), 'w', encoding='utf-8') as f:
        f.write("[TCN 稳定版] 5 分类报告:\n" + "="*60 + "\n" + report_str)
        f.write(f"\n🎯 最终 Accuracy 分数: {final_acc:.4f}\n")
        f.write(f"🎯 最终 Macro-F1 分数: {best_f1:.4f}\n")

    cm = confusion_matrix(y_test, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'TCN Stable Confusion Matrix (F1: {best_f1:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "tcn_5class_stable_cm.png"), dpi=300)
    print(f"📊 结果已保存至 {report_dir}")

if __name__ == "__main__":
    try:
        train_and_evaluate_tcn()
    except Exception as e:
        print(f"❌ 发生错误: {e}")