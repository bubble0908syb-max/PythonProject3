import pandas as pd
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_csv('D:\python\PythonProject3\project\data\processed\preprocessed_data.csv')
lithology_col = 'Lith_Section'

# 统计岩性分布
lithology_stats = data[lithology_col].value_counts().reset_index()
lithology_stats.columns = ['岩性类型', '样本数量']
total_count = lithology_stats['样本数量'].sum()
lithology_stats['占比(%)'] = (lithology_stats['样本数量'] / total_count * 100).round(2)

# 生成饼图
colors = ['#4a78d4', '#79bc52', '#ed8b33', '#f2c037', '#36c6c1']
pie_labels = [f'{row["岩性类型"]}\n{row["样本数量"]}, {row["占比(%)"]}%' for _, row in lithology_stats.iterrows()]

fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
ax.pie(
    lithology_stats['样本数量'],
    labels=pie_labels,
    colors=colors,
    autopct='',
    startangle=90,
    textprops={'fontsize': 12, 'fontweight': 'bold'},
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
)
ax.axis('equal')
plt.title('目标层段岩性频率分布饼图', fontsize=18, fontweight='bold', pad=30)
plt.tight_layout()
plt.show()