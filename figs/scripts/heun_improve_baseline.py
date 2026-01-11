import matplotlib.pyplot as plt
import numpy as np

# ======================
# 原始实验数据（来自你的表格）
# ======================
metrics = ["CLIP-score", "Image-reward", "Aesthetic-score"]

# n = 10
n10_euler = [0.8573, 0.7654, 6.5238]
n10_heun  = [0.8687, 1.0615, 6.6581]

# n = 20
n20_euler = [0.8667, 1.1484, 6.7340]
n20_heun  = [0.8674, 1.1823, 6.7465]

data = {
    "n=10": {"Euler": n10_euler, "Heun": n10_heun},
    "n=20": {"Euler": n20_euler, "Heun": n20_heun},
}

# ======================
# 图形设置
# ======================
plt.style.use("bmh")
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle("Effect of Higher-order Solver (Euler vs Heun)", fontsize=16, y=1.05)

colors = {
    "Euler": "#4C72B0",  # 柔和蓝
    "Heun":  "#55A868",  # 柔和绿
}

bar_width = 0.35
x = np.arange(2)  # n=10, n=20 两组

# ======================
# 绘制每个指标的柱状图
# ======================
for idx, metric in enumerate(metrics):
    ax = axes[idx]

    # 取指标数据
    euler_vals = [n10_euler[idx], n20_euler[idx]]
    heun_vals  = [n10_heun[idx],  n20_heun[idx]]

    # 计算百分比提升
    improvement = [(h - e) / e * 100 for e, h in zip(euler_vals, heun_vals)]

    # 画柱子
    bars1 = ax.bar(x - bar_width/2, euler_vals, bar_width, label="Euler", color=colors["Euler"])
    bars2 = ax.bar(x + bar_width/2, heun_vals,  bar_width, label="Heun",  color=colors["Heun"])

    # 轴设置
    ax.set_title(metric, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["n=10", "n=20"])
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    # ---------- 在柱状图上标注数值 ----------
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    # ---------- 标注提升百分比（只标 Heun 柱） ----------
    for i, bar in enumerate(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                h + 0.05,
                f"+{improvement[i]:.1f}%",
                ha="center", va="bottom", color="#C44E52", fontsize=10, fontweight="bold")

    # 图例
    if idx == 0:
        ax.legend()

plt.tight_layout()
plt.savefig("metric_comparison_barplots.png", dpi=300, bbox_inches="tight")
# plt.show()
