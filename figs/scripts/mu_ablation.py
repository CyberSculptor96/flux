import matplotlib.pyplot as plt

# 1. 数据准备 (已从图片中提取)
# 注意：表格中的 mu 是乱序的 (0.0 在最后)，为了画折线图，我们需要按 mu 从小到大排序
raw_data = [
    {'mu': 0.6,  'n10': [0.8654, 1.0779, 6.7556], 'n20': [0.8709, 1.1780, 6.8436]},
    {'mu': 1.15, 'n10': [0.8651, 1.1387, 6.8344], 'n20': [0.8683, 1.1873, 6.8439]},
    {'mu': 1.4,  'n10': [0.8649, 1.1428, 6.8465], 'n20': [0.8662, 1.1905, 6.8795]},
    {'mu': 1.8,  'n10': [0.8622, 1.1165, 6.8320], 'n20': [0.8658, 1.1788, 6.9101]},
    {'mu': 0.0,  'n10': [0.8573, 0.7654, 6.5238], 'n20': [0.8667, 1.1484, 6.7340]},
]

# 2. 数据排序与整理
# 按照 mu 的值进行升序排序，保证 x 轴连线正确
sorted_data = sorted(raw_data, key=lambda x: x['mu'])

mu_vals = [d['mu'] for d in sorted_data]

# 提取 n=10 的各项指标
n10_clip = [d['n10'][0] for d in sorted_data]
n10_ir   = [d['n10'][1] for d in sorted_data]
n10_aes  = [d['n10'][2] for d in sorted_data]

# 提取 n=20 的各项指标
n20_clip = [d['n20'][0] for d in sorted_data]
n20_ir   = [d['n20'][1] for d in sorted_data]
n20_aes  = [d['n20'][2] for d in sorted_data]

# 3. 设置绘图风格
plt.style.use('bmh') # 使用一种比较干净的科研风格
fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # 1行3列
fig.suptitle('Ablation Study Analysis of Mu (n=10 and n=20)', fontsize=16, y=1.05)

# 定义画图辅助函数
def plot_metric(ax, x, y1, y2, title, ylabel):
    ax.plot(x, y1, marker='o', label='n=10', linewidth=2, linestyle='-')
    ax.plot(x, y2, marker='^', label='n=20', linewidth=2, linestyle='--')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Mu', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    ax.grid(True)

    ax.set_xticks(x)

    # 标出数据点数值（可选，防止重叠过于密集可注释掉）
    for i, txt in enumerate(y1):
        ax.annotate(f"{txt:.4f}", (x[i], y1[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
    for i, txt in enumerate(y2):
        ax.annotate(f"{txt:.4f}", (x[i], y2[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# 4. 绘制三个子图
# 子图 1: CLIP Score
plot_metric(axes[0], mu_vals, n10_clip, n20_clip, 'CLIP Score', 'Score')

# # 子图 2: Image Reward
# plot_metric(axes[1], mu_vals, n10_ir, n20_ir, 'Image Reward', 'Reward')

# 子图 3: Aesthetic Score
plot_metric(axes[1], mu_vals, n10_aes, n20_aes, 'Aesthetic Score', 'Score')

# 子图 2: Image Reward
plot_metric(axes[2], mu_vals, n10_ir, n20_ir, 'Image Reward', 'Reward')

plt.tight_layout()
plt.savefig("mu_ablation.png", dpi=300, bbox_inches='tight')
