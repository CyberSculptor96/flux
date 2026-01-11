import matplotlib.pyplot as plt
import numpy as np

# 1. 原始数据（沿用你之前的）
raw_data = [
    {'mu': 0.6,  'n10': [0.8654, 1.0779, 6.7556], 'n20': [0.8709, 1.1780, 6.8436], 'n50': [0.8665, 1.2089, 6.8573]},
    {'mu': 1.15, 'n10': [0.8651, 1.1387, 6.8344], 'n20': [0.8683, 1.1873, 6.8439], 'n50': [0.8685, 1.2102, 6.8649]},
    {'mu': 1.4,  'n10': [0.8649, 1.1428, 6.8465], 'n20': [0.8662, 1.1905, 6.8795], 'n50': [0.8681, 1.1925, 6.8514]},
    {'mu': 1.8,  'n10': [0.8622, 1.1165, 6.8320], 'n20': [0.8658, 1.1788, 6.9101], 'n50': [0.8649, 1.1468, 6.8844]},
    {'mu': 0.0,  'n10': [0.8573, 0.7654, 6.5238], 'n20': [0.8667, 1.1484, 6.7340], 'n50': [0.8680, 1.1915, 6.7929]},
]

# ---------------------------
# 2. 计算三个指标的全局 min-max（用于归一化）
# ---------------------------
all_clip, all_ir, all_aes = [], [], []

for item in raw_data:
    for key in ['n10', 'n20', 'n50']:
        clip, ir, aes = item[key]
        all_clip.append(clip)
        all_ir.append(ir)
        all_aes.append(aes)

clip_min, clip_max = min(all_clip), max(all_clip)
ir_min, ir_max     = min(all_ir),   max(all_ir)
aes_min, aes_max   = min(all_aes),  max(all_aes)

def normalize_triplet(triplet):
    clip, ir, aes = triplet
    norm_clip = (clip - clip_min) / (clip_max - clip_min)
    norm_ir   = (ir   - ir_min)   / (ir_max   - ir_min)
    norm_aes  = (aes  - aes_min)  / (aes_max  - aes_min)
    overall   = (norm_clip + norm_ir + norm_aes) / 3
    return overall


# 2. 方便按 mu 查询：转成 dict
data_by_mu = {item['mu']: item for item in raw_data}

# 3. 取出对照组数据
# 对照组 1: n=10, mu=1.15  vs  n=20, mu=0.0
g1_a = data_by_mu[1.15]['n10']  # [CLIP, ImageReward, Aesthetic]
g1_b = data_by_mu[0.0]['n20']

# 对照组 2: n=20, mu=1.15  vs  n=50, mu=0.0
g2_a = data_by_mu[1.15]['n20']
g2_b = data_by_mu[0.0]['n50']

# g1_a_overall = sum(g1_a) / 3
# g1_b_overall = sum(g1_b) / 3
# g2_a_overall = sum(g2_a) / 3
# g2_b_overall = sum(g2_b) / 3

g1_a_overall = normalize_triplet(g1_a)
g1_b_overall = normalize_triplet(g1_b)
g2_a_overall = normalize_triplet(g2_a)
g2_b_overall = normalize_triplet(g2_b)

# 拼回数组
g1_a = g1_a + [g1_a_overall]
g1_b = g1_b + [g1_b_overall]
g2_a = g2_a + [g2_a_overall]
g2_b = g2_b + [g2_b_overall]

metrics = ['CLIP', 'ImageReward', 'Aesthetic', 'Overall']
x = np.arange(len(metrics))
width = 0.25    # 0.35

# 4. 画图
plt.style.use('bmh')
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Key Group Comparisons', y=1.05, fontsize=14)

# -------- 图 1：对照组 1 --------
ax1 = axes[0]
bars1 = ax1.bar(x - width/2, g1_a, width, label='n=10, μ=1.15', color='tab:blue')
bars2 = ax1.bar(x + width/2, g1_b, width, label='n=20, μ=0.0', color='tab:olive')

ax1.set_title('Group 1: n=10,μ=1.15 / n=20,μ=0.0')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, rotation=0)
ax1.set_ylabel('Score')
ax1.legend()
ax1.grid(True, axis='y')

# 可选：在柱子上标数值
for bar in bars1 + bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords='offset points',
                 ha='center', va='bottom', fontsize=7)

# -------- 图 2：对照组 2 --------
ax2 = axes[1]
bars3 = ax2.bar(x - width/2, g2_a, width, label='n=20, μ=1.15', color='tab:green')
bars4 = ax2.bar(x + width/2, g2_b, width, label='n=50, μ=0.0', color='tab:cyan')

ax2.set_title('Group 2: n=20,μ=1.15 / n=50,μ=0.0')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, rotation=0)
ax2.set_ylabel('Score')
ax2.legend()
ax2.grid(True, axis='y')

for bar in bars3 + bars4:
    height = bar.get_height()
    ax2.annotate(f'{height:.3f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords='offset points',
                 ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig("mu_key_groups_barplot.png", dpi=300, bbox_inches='tight')
# plt.show()  # 如需交互查看可以打开这一行
