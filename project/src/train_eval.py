"""
测井岩性识别项目 - XGBoost 基准模型 (防泄露对齐版)
职责：读取 sequence_builder.py 生成的【对齐版】训练集和测试集，训练 XGBoost。
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

def run_xgboost_aligned(train_path, test_path, model_dir, report_dir):
    print(f"📥 正在加载对齐后的训练集: {train_path}")
    print(f"📥 正在加载对齐后的测试集: {test_path}")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("找不到对齐的数据集，请确保已运行最新的 sequence_builder.py！")

    # 分别读取已经划分好的训练集和测试集
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    target_col = 'Lith_Encoded'

    # 获取特征矩阵和标签 (不再需要剔除 Well_Name 等，因为对齐脚本里只保留了纯净特征)
    y_train = df_train[target_col]
    X_train = df_train.drop(columns=[target_col])

    y_test = df_test[target_col]
    X_test = df_test.drop(columns=[target_col])

    # ==========================================
    # 🌟 动态读取并过滤特征名单
    # ==========================================
    selected_features_path = os.path.join(report_dir, 'selected_features.txt')

    if os.path.exists(selected_features_path):
        print(f"\n🔍 找到特征选择名单: {selected_features_path}")
        with open(selected_features_path, 'r', encoding='utf-8') as f:
            selected_features = [line.strip() for line in f.readlines() if line.strip()]

        valid_features = [feat for feat in selected_features if feat in X_train.columns]
        print(f"✅ 原始特征数: {X_train.shape[1]} -> 筛选后特征数: {len(valid_features)}")
        print(f"✅ 选中的特征为: {valid_features}")

        # 对训练集和测试集同时进行特征过滤
        X_train = X_train[valid_features]
        X_test = X_test[valid_features]
    else:
        print(f"\n⚠️ 未找到特征名单 ({selected_features_path})，将使用全部特征进行训练。")

    # 获取真实的类别名称
    try:
        le = joblib.load('../data/processed/ml_data/label_encoder.pkl')
        class_names = list(le.classes_)
    except:
        class_names = [f'Class {i}' for i in sorted(y_train.unique())]

    # ==========================================
    # 1. 训练 XGBoost 模型
    # ==========================================
    print("\n🚀 开始训练防泄露版 XGBoost 模型...")
    # 参数保持与之前一致
    model = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # ==========================================
    # 2. 在测试集上评估 & 保存文字报告
    # ==========================================
    print("✅ 训练完成，正在测试集上进行真实评估...")
    y_pred = model.predict(X_test)

    report_str = classification_report(y_test, y_pred, target_names=class_names)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    final_acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*50)
    print("[XGBoost 防泄露版] 测试集分类报告:")
    print("="*50)
    print(report_str)
    print(f"🎯 最终 Accuracy 分数: {final_acc:.4f}")
    print(f"🎯 最终 Macro-F1 分数: {macro_f1:.4f}")

    # 保存报告 (使用 aligned 后缀区分)
    os.makedirs(report_dir, exist_ok=True)
    report_txt_path = os.path.join(report_dir, 'xgboost_aligned_report.txt')
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write("[XGBoost 基础模型 (防泄露对齐版)] 测试集分类报告:\n")
        f.write("="*50 + "\n")
        f.write(report_str + "\n")
        f.write(f"🎯 最终 Accuracy 分数: {final_acc:.4f}\n")
        f.write(f"🎯 最终 Macro-F1 分数: {macro_f1:.4f}\n")
    print(f"📝 详细文字分类报告已保存至: {report_txt_path}")

    # ==========================================
    # 3. 绘制并保存混淆矩阵
    # ==========================================
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'XGBoost Aligned Confusion Matrix (F1: {macro_f1:.4f})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    cm_path = os.path.join(report_dir, 'xgboost_aligned_cm.png')
    plt.savefig(cm_path, dpi=300)
    print(f"📊 混淆矩阵已保存至: {cm_path}")

    # ==========================================
    # 4. 保存对齐版模型
    # ==========================================
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'xgboost_aligned.json')
    model.save_model(model_path)
    print(f"💾 防泄露 XGBoost 模型已保存至: {model_path}")

if __name__ == "__main__":
    # 🌟 指向新生成的、已经严格划分好的对齐数据集
    TRAIN_PATH = "../data/processed/ml_aligned_data/train_aligned.csv"
    TEST_PATH = "../data/processed/ml_aligned_data/test_aligned.csv"
    MODEL_DIR = "../saved_models/xgboost"
    REPORT_DIR = "../reports/figures"

    try:
        run_xgboost_aligned(TRAIN_PATH, TEST_PATH, MODEL_DIR, REPORT_DIR)
    except Exception as e:
        print(f"❌ 发生错误: {e}")