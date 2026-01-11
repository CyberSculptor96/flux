import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# 1. 原始质量数据 raw_data
# ---------------------------
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

# ---------------------------
# 3. 选取关键对照组并计算 overall_norm
# ---------------------------
data_by_mu = {item['mu']: item for item in raw_data}

# 四个关键配置
key_configs = [
    (1.15, 10),  # Group 1 - A
    (0.0,  20),  # Group 1 - B
    (1.15, 20),  # Group 2 - A
    (0.0,  50),  # Group 2 - B
]

quality_dict = {}
for mu, n in key_configs:
    key = f"n{n}"
    overall_norm = normalize_triplet(data_by_mu[mu][key])
    quality_dict[(mu, n)] = overall_norm

# ---------------------------
# 4. 填入推理速度数据（s / image）
# ---------------------------
# ⭐⭐ 你只要改这里 ⭐⭐
speed = {
    (1.15, 10): 7.52,   # s/image
    (0.0,  20): 11.81,
    (1.15, 20): 12.25,
    (0.0,  50): 24.02,
}

# ---------------------------
# 5. 绘制质量-速度 Trade-off 散点图
# ---------------------------
plt.style.use('bmh')
fig, ax = plt.subplots(figsize=(7, 5))

colors = {
    (1.15, 10): 'tab:blue',
    (0.0,  20): 'tab:orange',   # 'tab:orange'
    (1.15, 20): 'tab:green',
    (0.0,  50): 'tab:red'   # 'tab:red'
}

for (mu, n) in key_configs:
    x = speed[(mu, n)]                  # 推理速度（s）
    y = quality_dict[(mu, n)]           # 归一化质量
    label1 = f"n={n}, μ={mu}"

    if mu == 1.15:
        marker_style = '*'
        marker_size = 220  # ⭐ 更大更突出
        label2 = "ours"
    else:
        marker_style = 'o'
        marker_size = 80   # 圆圈变小
        label2 = "baseline"

    ax.scatter(x, y, s=marker_size, color=colors[(mu, n)], label=label2, marker=marker_style)

    ax.annotate(label1, (x, y), xytext=(5,5), textcoords='offset points', fontsize=9)


# ours
ours_points = [(speed[(mu,n)], quality_dict[(mu,n)]) 
               for (mu,n) in key_configs if mu == 1.15]
ours_points = sorted(ours_points)
ax.plot([p[0] for p in ours_points], [p[1] for p in ours_points],
        linestyle="--", color="tab:green", linewidth=1.5,
        dashes=(5, 6)
)

# baseline
base_points = [(speed[(mu,n)], quality_dict[(mu,n)]) 
               for (mu,n) in key_configs if mu == 0.0]
base_points = sorted(base_points)
ax.plot([p[0] for p in base_points], [p[1] for p in base_points],
        linestyle="--", color="tab:red", linewidth=1.5,
        dashes=(5, 6)
)


ax.set_xlabel("Inference time (s / image)")
ax.set_ylabel("Normalized Overall Quality")
ax.set_title("Quality–Speed Trade-off for Key Configurations")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.savefig("tradeoff_scatter.png", dpi=300, bbox_inches='tight')
# plt.show()
