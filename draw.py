import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 1. 核心设置：保证导出 SVG/PDF 后，Illustrator 里文本依然可以无损编辑
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'  # 使用科研常见的无衬线字体

# seed
# x = ['128', '192', '256', '512', '1024'] 
# y = [90.50, 92.47, 93.76, 92.14, 92.49]

# seed-iv
x = ['64', '96', '128', '196', '256'] 
y = [81.68, 82.65, 80.91, 81.30, 81.93]


# 3. 创建画布 (8x5 是很符合视觉审美的比例)
fig, ax = plt.subplots(figsize=(8, 5))

# 4. 绘制柱状图
# color 使用了高级的莫兰迪蓝色，edgecolor 加了黑色边框提升质感，alpha 让颜色稍微柔和
bars = ax.bar(x, y, color='#4C72B0', edgecolor='black', linewidth=1.2, width=0.6, alpha=0.85)

# 5. 设置坐标轴标签 (这里我先用了常见的 Placeholder，你可以自行替换)
ax.set_xlabel('Size of FIFO Queue', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')

# 6. 【关键细节】调整 Y 轴显示范围
# 因为你的数据都在 90 以上，如果从 0 开始画，柱子高度看起来几乎一样。
# 截断 Y 轴（比如从 89 开始）可以清晰放大出 93.76 这个最佳表现。
ax.set_ylim(78, 84)

# 7. 在每根柱子顶部标注具体数值
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),  # 垂直向上偏移 6 个像素
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# 8. 美化网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray') # 仅保留横向辅助线
ax.set_axisbelow(True) # 让网格线处于柱子的图层下方

# 9. 隐藏顶部和右侧的边框 (APA等学术期刊极简风格规范)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 10. 自动调整布局并保存为矢量图
plt.tight_layout()
plt.savefig('seed-iv.svg', format='svg', transparent=True)
# plt.savefig('pretty_bar_chart.pdf', format='pdf', transparent=True)

# plt.show()