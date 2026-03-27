"""
测井岩性识别项目 - 数据增强模块 (适配纯净数据版)
职责：读取 preprocessed_data.csv，对极少数类使用 SMOTE 打底，使用 CTGAN 扩充至多数类规模。
注意：为生成的虚拟数据构建了“虚拟井名(Synthetic_Well)”，以防止后续序列构建代码报错。
"""

import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from ctgan import CTGAN

def augment_data(input_filepath, output_dir, model_save_dir, smote_threshold=300, epochs=150):
    print(f"📥 正在加载预处理后的全局数据: {input_filepath}")
    df = pd.read_csv(input_filepath)

    feature_cols = ['_CAL', '_GR', '_SP', '_LLD', '_LLS', '_AC', '_DEN', '_PEF']
    target_col = 'Lith_Encoded'

    # 分离真实数据的特征和标签
    X_real = df[feature_cols]
    y_real = df[target_col]

    class_counts = y_real.value_counts()
    max_count = class_counts.max()
    print(f"\n📊 [初始状态] 各类别数量分布:\n{class_counts.sort_index()}")
    print(f"🎯 目标平衡数量 (多数类样本数): {max_count}")

    # ==========================================
    # 步骤 1: SMOTE 局部打底 (处理极少数类)
    # ==========================================
    print(f"\n{'-'*40}")
    print(f"🛠️ 第一阶段: SMOTE 局部扩充 (门槛设为 {smote_threshold})")

    smote_strategy = {}
    for c, count in class_counts.items():
        if count < smote_threshold:
            smote_strategy[c] = smote_threshold
        else:
            smote_strategy[c] = count

    smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
    X_smote, y_smote = smote.fit_resample(X_real, y_real)

    df_smote = pd.DataFrame(X_smote, columns=feature_cols)
    df_smote[target_col] = y_smote

    smote_counts = df_smote[target_col].value_counts()
    print(f"📊 [SMOTE后] 各类别数量分布:\n{smote_counts.sort_index()}")

    # ==========================================
    # 步骤 2 & 3: CTGAN 训练与终极生成
    # ==========================================
    print(f"\n{'-'*40}")
    print("🧬 第二阶段: 基于 SMOTE 数据训练 CTGAN 并生成对齐样本")

    synthetic_data_list = []
    os.makedirs(model_save_dir, exist_ok=True)

    for class_label, count in smote_counts.items():
        if count >= max_count:
            continue

        needed_samples = max_count - count
        print(f"\n🚀 准备为类别 [{class_label}] 训练 CTGAN，并补齐 {needed_samples} 个样本...")

        class_data = df_smote[df_smote[target_col] == class_label][feature_cols]

        # 训练 CTGAN
        ctgan = CTGAN(epochs=epochs, verbose=True)
        print(f"开始炼丹... (Epochs={epochs})")
        ctgan.fit(class_data)

        model_path = os.path.join(model_save_dir, f"ctgan_hybrid_class_{class_label}.pkl")
        ctgan.save(model_path)

        # 生成样本
        print(f"正在生成 {needed_samples} 个虚拟样本...")
        synthetic_samples = ctgan.sample(needed_samples)
        synthetic_samples[target_col] = class_label

        # 🌟 核心适配：为虚拟样本分配假的井名和连续深度，防止序列切分时报错
        synthetic_samples['Well_Name'] = f"Synthetic_Well_Class_{class_label}"
        # 伪造一个连续的深度，每隔 0.125 米一个点
        synthetic_samples['TopDepth'] = np.arange(0, needed_samples * 0.125, 0.125)

        synthetic_data_list.append(synthetic_samples)
        print(f"✅ 类别 [{class_label}] 混合增强完毕！")

    # ==========================================
    # 步骤 4: 合并真实数据与虚拟数据并保存
    # ==========================================
    print(f"\n{'-'*40}")
    print("正在合并【原始数据 + CTGAN 虚拟数据】...")

    # 注意：这里我们将原始最纯净的 df 和新生成的 CTGAN 列表合并 (摒弃 SMOTE 产生的脏数据)
    # 因为 SMOTE 只是用来给 CTGAN 提供起步训练数据的！真正的产出我们只拿 CTGAN 生成的。
    final_df = pd.concat([df] + synthetic_data_list, ignore_index=True)

    # 按井名和深度排序，保持地质序列连续性
    if 'TopDepth' in final_df.columns:
        final_df = final_df.sort_values(by=['Well_Name', 'TopDepth']).reset_index(drop=True)

    print("\n📊 [增强后 (最终态)] 全局数据各类别数量分布:")
    print(final_df[target_col].value_counts().sort_index())

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'augmented_data.csv')
    final_df.to_csv(output_path, index=False)

    print(f"\n✅ 数据增强执行完毕！")
    print(f"🚀 终极增强数据集已保存至: {output_path}")

if __name__ == "__main__":
    PREPROCESSED_FILE = "../data/processed/preprocessed_data.csv"
    ML_OUTPUT_DIR = "../data/processed"
    HYBRID_MODEL_DIR = "../saved_models/ctgan_hybrid_synthesizer"

    try:
        # 可以先设 epochs=10 测试，跑通后再改回 150 慢慢炼丹
        augment_data(PREPROCESSED_FILE, ML_OUTPUT_DIR, HYBRID_MODEL_DIR, smote_threshold=300, epochs=150)
    except Exception as e:
        print(f"❌ 发生错误: {e}")