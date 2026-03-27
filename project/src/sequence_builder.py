"""
测井岩性识别项目 - 时间序列/滑动窗口构建模块 (防泄露对齐版)
职责：读取增强数据，重构为 3D 张量，并【同步生成】完全对齐的 2D 表格数据供树模型使用。
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_sequences_by_well(df, sequence_length, feature_cols, target_col):
    X_list = []
    y_list = []

    df = df.sort_values(by=['Well_Name', 'TopDepth']).reset_index(drop=True)
    grouped = df.groupby('Well_Name')

    for well_name, group in grouped:
        group_X = group[feature_cols].values
        group_y = group[target_col].values

        if len(group) < sequence_length:
            continue

        for i in range(len(group) - sequence_length + 1):
            window_X = group_X[i : i + sequence_length, :]
            window_y = group_y[i + sequence_length - 1]

            X_list.append(window_X)
            y_list.append(window_y)

    return np.array(X_list), np.array(y_list)

def process_and_save_aligned_data(data_dir, output_dir, sequence_length=15):
    print(f"🔄 正在读取全局增强数据集，构建完全对齐的 2D & 3D 数据集...")

    augmented_file = os.path.join(data_dir, 'augmented_data.csv')
    df = pd.read_csv(augmented_file)

    feature_cols = ['_CAL', '_GR', '_SP', '_LLD', '_LLS', '_AC', '_DEN', '_PEF']
    target_col = 'Lith_Encoded'

    # 1. 构建 3D 序列
    X_3d, y_3d = create_sequences_by_well(df, sequence_length, feature_cols, target_col)

    # 2. 核心修复：基于 3D 数据集的随机种子进行统一的 Train/Test 划分
    X_train_3d, X_test_3d, y_train, y_test = train_test_split(
        X_3d, y_3d, test_size=0.2, random_state=42, stratify=y_3d
    )

    # 3. 提取 3D 张量的最后一个时间步，作为 2D 树模型的训练数据
    # 这保证了 XGBoost 训练集里的样本，绝对不会跑到 TCN 的测试集里去！
    X_train_2d = X_train_3d[:, -1, :]
    X_test_2d = X_test_3d[:, -1, :]

    # ==========================================
    # 保存深度学习 (3D) 数据
    # ==========================================
    dl_dir = os.path.join(output_dir, "dl_data")
    os.makedirs(dl_dir, exist_ok=True)
    np.save(os.path.join(dl_dir, 'X_train_dl.npy'), X_train_3d)
    np.save(os.path.join(dl_dir, 'y_train_dl.npy'), y_train)
    np.save(os.path.join(dl_dir, 'X_test_dl.npy'), X_test_3d)
    np.save(os.path.join(dl_dir, 'y_test_dl.npy'), y_test)

    # ==========================================
    # 保存传统机器学习 (2D) 对齐数据
    # ==========================================
    ml_align_dir = os.path.join(output_dir, "ml_aligned_data")
    os.makedirs(ml_align_dir, exist_ok=True)

    # 转回 DataFrame 方便 XGBoost/CatBoost 带特征名读取
    df_train_2d = pd.DataFrame(X_train_2d, columns=feature_cols)
    df_train_2d[target_col] = y_train
    df_train_2d.to_csv(os.path.join(ml_align_dir, 'train_aligned.csv'), index=False)

    df_test_2d = pd.DataFrame(X_test_2d, columns=feature_cols)
    df_test_2d[target_col] = y_test
    df_test_2d.to_csv(os.path.join(ml_align_dir, 'test_aligned.csv'), index=False)

    print(f"✅ 对齐完毕！深度学习数据存至: {dl_dir}")
    print(f"✅ 对齐完毕！树模型专属数据存至: {ml_align_dir}")

if __name__ == "__main__":
    DATA_DIR = "../data/processed"
    OUTPUT_BASE_DIR = "../data/processed"
    process_and_save_aligned_data(DATA_DIR, OUTPUT_BASE_DIR, sequence_length=30)