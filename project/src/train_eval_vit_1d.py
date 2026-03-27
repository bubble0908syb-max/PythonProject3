"""
测井岩性识别项目 - Vision Transformer (ViT-1D) 模型
状态：动态特征筛选 + 5 分类 + 全局自注意力机制
核心升级：引入 旋转位置编码 (RoPE) + 全局平均池化 (GAP) 替代绝对位置编码和 CLS Token
"""

import os
import math
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
# 1. 定义 RoPE 核心组件
# ==========================================
class RoPEMultiheadAttention(nn.Module):
    """自定义带有旋转位置编码 (RoPE) 的多头注意力机制"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 预计算 RoPE 的频率矩阵
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def apply_rotary_pos_emb(self, x, sin, cos):
        # x 形状: (Batch, Seq_len, nhead, head_dim)
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        # 旋转操作
        rotated_x = torch.cat([-x2, x1], dim=-1)
        return (x * cos) + (rotated_x * sin)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim)
        k = self.k_proj(x).view(B, L, self.nhead, self.head_dim)
        v = self.v_proj(x).view(B, L, self.nhead, self.head_dim)

        # 动态生成当前序列长度 L 的 sin 和 cos
        t = torch.arange(L, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq) # (L, head_dim//2)
        emb = torch.cat((freqs, freqs), dim=-1) # (L, head_dim)
        cos = emb.cos()[None, :, None, :] # (1, L, 1, head_dim)
        sin = emb.sin()[None, :, None, :] # (1, L, 1, head_dim)

        # 在 Q 和 K 上应用 RoPE
        q = self.apply_rotary_pos_emb(q, sin, cos)
        k = self.apply_rotary_pos_emb(k, sin, cos)

        # 标准的多头注意力计算
        q = q.transpose(1, 2) # (B, nhead, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v) # (B, nhead, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class RoPETransformerEncoderLayer(nn.Module):
    """使用 Pre-LayerNorm 架构的 Transformer Block (收敛更稳)"""
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.3):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        # Pre-LN 结构
        src2 = self.norm1(src)
        src = src + self.dropout1(self.self_attn(src2))
        src2 = self.norm2(src)
        src = src + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src2)))))
        return src

# ==========================================
# 2. 定义 SOTA 版本的 1D Vision Transformer
# ==========================================
class ViT1D(nn.Module):
    def __init__(self, seq_len, num_features, num_classes, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.3):
        super(ViT1D, self).__init__()

        # 1. 线性投影层
        self.feature_projection = nn.Linear(num_features, d_model)
        self.pos_drop = nn.Dropout(p=dropout)

        # 2. 核心层：包含 RoPE 的 Transformer
        # 这里彻底去掉了 nn.Parameter(绝对位置) 和 CLS Token
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # 3. LayerNorm 与 分类头
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, num_features)
        """
        # 1. 特征投影
        x = self.feature_projection(x)
        x = self.pos_drop(x)

        # 2. 通过带有 RoPE 的 Transformer 层
        for layer in self.layers:
            x = layer(x)

        # 3. 全局平均池化 (GAP) 替代 CLS Token提取全局特征
        # 在时间步维度 (dim=1) 求平均
        x = x.mean(dim=1)
        x = self.norm(x)

        # 4. 分类输出
        logits = self.classifier(x)
        return logits


# ==========================================
# 3. 辅助绘图函数
# ==========================================
def plot_training_curves(train_losses, test_f1_scores, output_dir, filename='vit_1d_rope_training_curves.png'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('ViT-1D (RoPE) Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_f1_scores, 'r-', label='Test Macro-F1')
    plt.title('ViT-1D (RoPE) Test Macro-F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Macro-F1')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)


# ==========================================
# 4. 训练与评估流程
# ==========================================
def train_and_evaluate_vit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 正在使用设备: {device}")

    # 加载数据
    data_dir = "../data/processed/dl_data"
    X_train = np.load(os.path.join(data_dir, 'X_train_dl.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_dl.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_dl.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_dl.npy'))

    # 动态特征过滤
    original_feature_cols = ['_CAL', '_GR', '_SP', '_LLD', '_LLS', '_AC', '_DEN', '_PEF']
    selected_features_path = "../reports/figures/selected_features.txt"

    if os.path.exists(selected_features_path):
        with open(selected_features_path, 'r') as f:
            selected_features = [line.strip() for line in f.readlines() if line.strip()]
        selected_indices = [original_feature_cols.index(feat) for feat in selected_features if
                            feat in original_feature_cols]
        X_train = X_train[:, :, selected_indices]
        X_test = X_test[:, :, selected_indices]
        print(f"✅ 使用特征: {selected_features}")

    # 获取类别名称
    try:
        le = joblib.load('../data/processed/ml_data/label_encoder.pkl')
        class_names = list(le.classes_)
    except:
        class_names = [f'Class {i}' for i in range(5)]

    # 类别权重
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

    seq_len = X_train.shape[1]
    num_features = X_train.shape[2]
    num_classes = len(np.unique(y_train))

    print(f"📐 模型输入: 序列长度={seq_len}, 特征数={num_features}, 输出类别数={num_classes}")

    # 初始化 ViT 模型 (采用 128 维度进阶配置)
    model = ViT1D(seq_len=seq_len, num_features=num_features, num_classes=num_classes,
                  d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.3).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)

    # 引入调度器控制学习率衰减
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    epochs = 100
    print(f"\n🔥 开始训练 Vision Transformer (RoPE) 模型...")

    best_f1 = 0.0
    model_save_dir = "../saved_models/deep_learning"
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "vit_1d_rope_5class.pth")

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

            # 梯度裁剪防爆
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)

        # 验证阶段
        model.eval()
        all_preds = []
        all_labels = []
        test_loss = 0.0

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

        scheduler.step(avg_test_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_acc = accuracy_score(all_labels, all_preds)
            print(
                f"Epoch [{epoch + 1:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test F1: {current_macro_f1:.4f}")

        if current_macro_f1 > best_f1:
            best_f1 = current_macro_f1
            torch.save(model.state_dict(), best_model_path)

    print(f"\n✅ 训练完毕！ViT-1D (RoPE) 最佳 Macro-F1 分数为: {best_f1:.4f}")

    # ==========================================
    # 5. 最终评估与保存图表
    # ==========================================
    report_dir = "../reports/figures"
    plot_training_curves(history_train_loss, history_test_f1, report_dir, filename='vit_1d_rope_training_curves.png')

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    final_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())

    print("\n[ViT-1D (RoPE) 架构] 终极分类报告:")
    print("=" * 60)
    report_str = classification_report(y_test, final_preds, target_names=class_names)
    print(report_str)

    final_acc = accuracy_score(y_test, final_preds)

    # 保存文字报告
    with open(os.path.join(report_dir, 'vit_1d_rope_report.txt'), 'w', encoding='utf-8') as f:
        f.write("[ViT-1D (RoPE) 架构] 分类报告:\n" + "=" * 60 + "\n" + report_str)
        f.write(f"\n🎯 最终 Accuracy 分数: {final_acc:.4f}\n")
        f.write(f"🎯 最终 Macro-F1 分数: {best_f1:.4f}\n")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'ViT-1D (RoPE) Confusion Matrix (F1: {best_f1:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "vit_1d_rope_confusion_matrix.png"), dpi=300)
    print(f"📊 混淆矩阵与文字报告已保存至 {report_dir}")


if __name__ == "__main__":
    try:
        train_and_evaluate_vit()
    except Exception as e:
        print(f"❌ 发生错误: {e}")