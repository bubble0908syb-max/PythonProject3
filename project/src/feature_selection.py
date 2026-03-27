"""
测井岩性识别项目 - 智能特征选择模块 (LightGBM + RF-RFE-CV)
职责：融合论文思路，评估特征重要性，并使用递归特征消除法寻找最优特征子集。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold


def perform_feature_selection(data_path, output_dir):
    """执行 LightGBM 特征评估与 RF-RFE-CV 特征消除"""
    print(f"正在加载基础训练集: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件，请检查路径: {data_path}")

    df = pd.read_csv(data_path)

    # 提取特征和标签
    feature_cols = ['_CAL', '_GR', '_SP', '_LLD', '_LLS', '_AC', '_DEN', '_PEF']
    target_col = 'Lith_Encoded'
    X = df[feature_cols]
    y = df[target_col]

    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 步骤 1: LightGBM 初始特征重要性评估
    # ==========================================
    print("\n" + "=" * 40)
    print("🌟 步骤 1: LightGBM 初始特征重要性评估")
    print("=" * 40)

    # 初始化 LightGBM 分类器
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1  # 关闭不必要的警告输出
    )

    # 训练模型以获取重要性
    lgb_model.fit(X, y)

    # 提取特征重要性 (基于分裂次数 'split' 或信息增益 'gain')
    importance = lgb_model.feature_importances_

    # 构建 DataFrame 并排序
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("\nLightGBM 评估出的特征重要性排名:")
    print(feature_importance_df)

    # 绘制特征重要性条形图
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'][::-1], feature_importance_df['Importance'][::-1], color='skyblue')
    plt.title('Feature Importance (LightGBM)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    importance_plot_path = os.path.join(output_dir, 'lightgbm_feature_importance.png')
    plt.savefig(importance_plot_path, dpi=300)
    print(f"📊 特征重要性图已保存至: {importance_plot_path}")

    # ==========================================
    # 步骤 2: RF-RFE-CV 递归特征消除
    # ==========================================
    print("\n" + "=" * 40)
    print("🌟 步骤 2: RF-RFE-CV 递归特征消除")
    print("=" * 40)

    # 初始化作为评估器的随机森林模型
    rf_estimator = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # 初始化 RFECV (使用 5 折分层交叉验证)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rfecv = RFECV(
        estimator=rf_estimator,
        step=1,  # 每次剔除 1 个特征
        cv=cv,  # 交叉验证策略
        scoring='f1_macro',  # 我们最关心的指标是 Macro-F1
        n_jobs=-1
    )

    print("正在执行交叉验证与特征消除 (这可能需要几分钟的时间)...")
    rfecv.fit(X, y)

    print(f"\n✅ 优化完成！最优的特征数量为: {rfecv.n_features_}")

    # 获取被选中的特征名称
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfecv.support_[i]]
    print(f"最优特征子集: {selected_features}")

    # 获取被淘汰的特征名称
    eliminated_features = [feature_cols[i] for i in range(len(feature_cols)) if not rfecv.support_[i]]
    if eliminated_features:
        print(f"被淘汰的冗余特征: {eliminated_features}")
    else:
        print("无冗余特征被淘汰。")

    # 绘制交叉验证得分随特征数量变化的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'], marker='o',
             linestyle='-', color='purple')
    plt.title('RF-RFE-CV: Macro-F1 vs Number of Features')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Cross-Validation Macro-F1 Score')
    plt.grid(True)
    plt.tight_layout()
    rfe_plot_path = os.path.join(output_dir, 'rf_rfecv_curve.png')
    plt.savefig(rfe_plot_path, dpi=300)
    print(f"📊 RFE-CV 验证曲线图已保存至: {rfe_plot_path}")

    # 将最优特征列表保存为文本文件，供后续深度学习模型读取
    with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    print(f"最优特征列表已保存至: {os.path.join(output_dir, 'selected_features.txt')}")


if __name__ == "__main__":
    # 使用包含原始不平衡数据的基线训练集进行特征评估最为准确
    TRAIN_DATA_PATH = "../data/processed/augmented_data.csv"
    REPORT_DIR = "../reports/figures"

    try:
        perform_feature_selection(TRAIN_DATA_PATH, REPORT_DIR)
    except Exception as e:
        print(f"发生错误: {e}")