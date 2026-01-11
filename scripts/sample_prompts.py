#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 short.json 中随机抽取 N 条，使用 prompt_id 作为 key。
输出格式：
{
  "915": "prompt text",
  "872": "...",
  ...
}
"""

import json
import os
import random

# === 配置区 ===
BASE_DIR = "data/t2i-diversity-evalprompts/"
INPUT_FILE = os.path.join(BASE_DIR, "short_gpt4o.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "prompts_short_100_by_id.json")

N = 100
SEED = 42


def main():
    print(f"[INFO] Loading: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Total prompts loaded: {len(data)}")

    random.seed(SEED)

    sampled = sorted(random.sample(data, N), key=lambda x: x["prompt_id"])

    # 使用 prompt_id 做 key
    result = {}
    for item in sampled:
        pid = str(item["prompt_id"])
        result[pid] = item["prompt"]

    print(f"[INFO] Saving {len(result)} prompts to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("[DONE] prompts_short_100_by_id.json generated successfully.")


if __name__ == "__main__":
    main()
