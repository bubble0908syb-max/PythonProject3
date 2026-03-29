import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 从你的训练文件中导入定义好的模型架构
# 确保此文件与 train_eval_vit_1d.py 在同级目录下
from train_eval_vit_1d import ViT1D


def evaluate_model():
    # ==========================================
    # 1. 环境与路径配置
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 正在使用推理设备: {device}")

    data_dir = "../data/processed/dl_data"
    model_path = "../saved_models/deep_learning/vit_1d_rope_5class.pth"
    selected_features_path = "../reports/figures/selected_features.txt"
    report_dir = "../reports/figures"
    os.makedirs(report_dir, exist_ok=True)

    # ==========================================
    # 2. 加载数据与标签
    # ==========================================
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到模型权重文件: {model_path}")

    print("📦 正在加载测试集数据...")
    X_test = np.load(os.path.join(data_dir, 'X_test_dl.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_dl.npy'))

    # 动态特征过滤 (必须与训练时保持一致)
    original_feature_cols = ['_CAL', '_GR', '_SP', '_LLD', '_LLS', '_AC', '_DEN', '_PEF']

    if os.path.exists(selected_features_path):
        with open(selected_features_path, 'r') as f:
            selected_features = [line.strip() for line in f.readlines() if line.strip()]
        selected_indices = [original_feature_cols.index(feat) for feat in selected_features if
                            feat in original_feature_cols]
        X_test = X_test[:, :, selected_indices]
        print(f"✅ 成功加载特征筛选，当前使用特征: {selected_features}")
    else:
        print(f"⚠️ 未找到特征筛选文件，使用全部 {len(original_feature_cols)} 个特征")

    # 获取类别名称
    try:
        le = joblib.load('../data/processed/ml_data/label_encoder.pkl')
        class_names = list(le.classes_)
    except:
        class_names = [f'Class {i}' for i in range(len(np.unique(y_test)))]

    # 转换为 Tensor DataLoader
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # ==========================================
    # 3. 初始化并加载模型
    # ==========================================
    seq_len = X_test.shape[1]
    num_features = X_test.shape[2]
    num_classes = len(np.unique(y_test))

    print(f"📐 模型输入信息: 序列长度={seq_len}, 特征数={num_features}, 输出类别数={num_classes}")

    # 使用与训练时相同的超参数初始化网络
    model = ViT1D(seq_len=seq_len, num_features=num_features, num_classes=num_classes,
                  d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.3).to(device)

    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式，关闭 Dropout 等
    print("✅ 模型权重加载成功！")

    # ==========================================
    # 4. 执行推理过程
    # ==========================================
    all_preds = []
    all_labels = []

    print("🔍 开始模型推理...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # ==========================================
    # 5. 指标计算与图表生成
    # ==========================================
    # 计算整体指标
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    report_str = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    print("\n" + "=" * 60)
    print("🏆 [ViT-1D (RoPE) 测试集评估报告]")
    print("=" * 60)
    print(report_str)
    print(f"🎯 整体 Accuracy (准确率) : {accuracy:.4f}")
    print(f"🎯 整体 Macro-F1 (宏F1)   : {macro_f1:.4f}")
    print("=" * 60)

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12})
    plt.title(f'ViT-1D (RoPE) Evaluation\nAccuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f}', fontsize=14)
    plt.ylabel('True Label (真实岩性)', fontsize=12)
    plt.xlabel('Predicted Label (预测岩性)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    cm_path = os.path.join(report_dir, "vit_1d_rope_evaluation_cm.png")
    plt.savefig(cm_path, dpi=300)
    print(f"📊 评估版混淆矩阵已保存至: {cm_path}")


if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as e:
        print(f"❌ 评估过程中发生错误: {e}")