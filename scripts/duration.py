import os
import re
import glob
import numpy as np

n = 20
method = "heun"
version = ""
LOG_DIR = f"logs/inference/n_{n}-{method}{version}"
OUTPUT_FILE = f"results/duration/n_{n}_{method}{version}.txt"

pattern = re.compile(r"Done in ([\d\.]+)s\. ")

times = []
files = glob.glob(os.path.join(LOG_DIR, f"n_{n}*{method}*.log"))

print(f"Found {len(files)} log files.")

for fp in files:
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    match = pattern.search(content)
    if match:
        t = float(match.group(1))
        times.append(t)
    else:
        print(f"[WARN] Missing 'Done in ...' line → {fp}")

# # --- 检查文件数量 ---
# if len(times) != 100:
#     raise ValueError(f"Valid logs = {len(times)}, expected = 100")

times.sort()
times_np = np.array(times)
mean_time = times_np.mean()

# --- 输出 ---
with open(OUTPUT_FILE, "w") as f:
    f.write("Extracted Times (seconds):\n")
    f.write(str(times) + "\n\n")
    f.write(f"Mean time: {mean_time:.4f} seconds\n")

print("Done! Results saved to:", OUTPUT_FILE)
print(f"Mean time = {mean_time:.4f}s")

# # --- 重命名原始 log 文件 ---
# for fp in files:
#     dirname = os.path.dirname(fp)
#     basename = os.path.basename(fp)

#     new_name = basename.replace("-mu_0.0-n_20-gpu_", "-method_euler-mu_0.0-n_20-gpu_")
#     new_fp = os.path.join(dirname, new_name)

#     os.rename(fp, new_fp)

# print("Renamed all log files to include method_euler.")

