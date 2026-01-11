import pandas as pd
import matplotlib.pyplot as plt

# 1. 原始数据
raw_data = [
    {'mu': 0.6,  'n10': [0.8654, 1.0779, 6.7556], 'n20': [0.8709, 1.1780, 6.8436], 'n50': [0.8665, 1.2089, 6.8573]},
    {'mu': 1.15, 'n10': [0.8651, 1.1387, 6.8344], 'n20': [0.8683, 1.1873, 6.8439], 'n50': [0.8685, 1.2102, 6.8649]},
    {'mu': 1.4,  'n10': [0.8649, 1.1428, 6.8465], 'n20': [0.8662, 1.1905, 6.8795], 'n50': [0.8681, 1.1925, 6.8514]},
    {'mu': 1.8,  'n10': [0.8622, 1.1165, 6.8320], 'n20': [0.8658, 1.1788, 6.9101], 'n50': [0.8649, 1.1468, 6.8844]},
    {'mu': 0.0,  'n10': [0.8573, 0.7654, 6.5238], 'n20': [0.8667, 1.1484, 6.7340], 'n50': [0.8680, 1.1915, 6.7929]},
]

# ---------------------------
# 全局 min-max（用于归一化）
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


# 2. 数据处理
processed_list = []

for item in raw_data:
    mu = item['mu']
    
    # 提取三个列表
    v10 = item['n10']
    v20 = item['n20']
    v50 = item['n50']
    
    # 计算对应位置的平均值 (Element-wise Average)
    # 假设列表顺序依然是 [CLIP, ImageReward, Aesthetic]
    avg_clip = (v10[0] + v20[0] + v50[0]) / 3
    avg_ir   = (v10[1] + v20[1] + v50[1]) / 3
    avg_aes  = (v10[2] + v20[2] + v50[2]) / 3
    
    # overall_avg = (avg_clip + avg_ir + avg_aes) / 3
    overall_avg = normalize_triplet([avg_clip, avg_ir, avg_aes])

    processed_list.append({
        'mu': mu,
        'avg_clip': avg_clip,
        'avg_ir': avg_ir,
        'avg_aes': avg_aes,
        'overall_avg': overall_avg
    })

# 3. 排序 (按 mu 从小到大)
processed_list.sort(key=lambda x: x['mu'])

# 4. 打印结果 (使用 Pandas 表格化输出)
df = pd.DataFrame(processed_list)

# 设置打印精度，方便查看
pd.set_option('display.precision', 4)

print("=== 计算结果 (Averaged over n=10, 20, 50) ===")
print(df)

# -------------------------------------------------
# 额外：如果你想简单画出这三个平均指标随 mu 的变化趋势
# -------------------------------------------------
plt.style.use('bmh')
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle('Average Metrics vs Mu (n=10/20/50)', y=1.05)

# 画图
axes[0].plot(df['mu'], df['avg_clip'], marker='o', color='tab:blue')
axes[0].set_title('Avg CLIP Score')
axes[1].plot(df['mu'], df['avg_aes'],  marker='o', color='tab:green')
axes[1].set_title('Avg Aesthetic Score')
axes[2].plot(df['mu'], df['avg_ir'],   marker='o', color='tab:orange')
axes[2].set_title('Avg Image Reward')

axes[3].plot(df['mu'], df['overall_avg'], marker='o', color='tab:red')
axes[3].set_title('Overall Average')

for ax in axes:
    ax.set_xlabel('Mu')
    ax.grid(True)

    ax.set_xticks(df['mu'])

plt.tight_layout()
plt.savefig("mu_ablation_stats_all.png", dpi=300, bbox_inches='tight')