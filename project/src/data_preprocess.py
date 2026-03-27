"""
测井岩性识别项目 - 数据预处理模块 (多井文件合并版)
职责：遍历读取多口井的独立 CSV 文件，自动拼接，然后统一进行缺失值处理、标准化和标签编码。
注意：此版本不包含人工衍生特征。
"""

import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def load_and_merge_wells(raw_data_dir):
    """
    遍历读取目录下的所有 CSV 文件并合并为一个 DataFrame。
    如果原始数据里没有 Well_Name 列，会自动用文件名作为井名填充。
    """
    print(f"📂 正在读取目录: {raw_data_dir}")
    # 查找所有 csv 文件
    csv_files = glob.glob(os.path.join(raw_data_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"在 {raw_data_dir} 目录下没有找到任何 CSV 文件！请检查路径。")

    print(f"找到 {len(csv_files)} 个井文件，正在合并...")

    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file)

        # 提取文件名(不含后缀)作为默认井名
        well_name_from_file = os.path.splitext(os.path.basename(file))[0]

        # 如果数据本身没有 Well_Name 列，我们就帮它加上（这对深度学习切分序列至关重要）
        if 'Well_Name' not in temp_df.columns:
            temp_df['Well_Name'] = well_name_from_file

        df_list.append(temp_df)

    # 将 12 口井的数据上下拼接成一个超级大表
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"✅ 合并成功！总数据行数: {len(merged_df)}")
    return merged_df

def preprocess_data(raw_data_dir, output_dir):
    """
    执行数据预处理的核心逻辑。
    """
    # 1. 加载并合并多口井的数据
    df = load_and_merge_wells(raw_data_dir)

    # 2. 定义列名 (请根据你的实际原始 CSV 列名微调)
    original_target_col = 'Lith_Section' if 'Lith_Section' in df.columns else 'Lithology'
    encoded_target_col = 'Lith_Encoded'

    # 纯正的 8 条物理测井特征
    feature_cols = ['_CAL', '_GR', '_SP', '_LLD', '_LLS', '_AC', '_DEN', '_PEF']
    meta_cols = ['Well_Name', 'TopDepth', 'BotDepth']

    # 3. 缺失值处理
    print("\n🧹 正在处理缺失值 (异常值 -999.25 将被视为 NaN 并剔除)...")
    df[feature_cols] = df[feature_cols].replace(-999.25, pd.NA)

    before_drop = len(df)
    # 剔除在核心特征或标签上有缺失的行
    df = df.dropna(subset=feature_cols + [original_target_col])
    after_drop = len(df)
    print(f"   -> 剔除了 {before_drop - after_drop} 行含有缺失值的数据。当前剩余干净数据量: {after_drop}")

    # 4. 标签编码 (Label Encoding)
    print("\n🏷️ 正在对岩性标签进行统一编码...")
    le = LabelEncoder()
    df[encoded_target_col] = le.fit_transform(df[original_target_col])

    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"   -> 全局标签映射关系: {mapping}")

    # 5. 特征标准化 (StandardScaler - 全局统一标准)
    print("\n📏 正在对物理测井曲线进行全局标准化 (Z-score 归一化)...")
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 6. 保存处理后的数据和模型对象
    os.makedirs(output_dir, exist_ok=True)

    final_cols = [col for col in meta_cols if col in df.columns] + feature_cols + [original_target_col, encoded_target_col]
    final_df = df[final_cols]

    # 按照井名和深度排序，确保地质序列的连续性！
    if 'TopDepth' in final_df.columns:
        final_df = final_df.sort_values(by=['Well_Name', 'TopDepth']).reset_index(drop=True)

    output_filepath = os.path.join(output_dir, 'preprocessed_data.csv')
    final_df.to_csv(output_filepath, index=False)
    print(f"\n✅ 预处理完成！合并且清洗后的干净数据已保存至: {output_filepath}")

    joblib.dump(scaler, os.path.join(output_dir, 'standard_scaler.pkl'))
    joblib.dump(le, os.path.join(output_dir, 'label_encoder.pkl'))
    print("💾 StandardScaler 和 LabelEncoder 已保存。")

if __name__ == "__main__":
    # ⚠️ 重要：这里请填入存放你 12 个井 CSV 文件的文件夹路径
    RAW_DATA_DIR = "../data/raw"
    PROCESSED_DIR = "../data/processed"

    try:
        preprocess_data(RAW_DATA_DIR, PROCESSED_DIR)
    except Exception as e:
        print(f"❌ 发生错误: {e}")