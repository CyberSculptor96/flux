#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取 parquet 文件，根据 type 字段分组，并输出为多个 JSON 文件。
每个 JSON 文件仅保留：prompt, prompt_id
"""

import os
import json
import pandas as pd

# 输入文件路径
INPUT_PATH = "data/t2i-diversity-evalprompts/train-00000-of-00001.parquet"

# 输出目录
OUTPUT_DIR = "data/t2i-diversity-evalprompts/"

def main():
    print(f"[INFO] Loading parquet from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    # 确认字段是否存在
    expected_cols = {"prompt", "type", "prompt_id"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    print(f"[INFO] Total rows: {len(df)}")

    # 获取所有 type
    types = sorted(df["type"].unique())
    print(f"[INFO] Found types: {types}")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for t in types:
        sub_df = df[df["type"] == t][["prompt", "prompt_id"]]

        # 转换为 list[dict]
        records = sub_df.to_dict(orient="records")

        # 输出文件，例如 short_gpt4o.json
        output_path = os.path.join(OUTPUT_DIR, f"{t}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"[OK] Saved {len(records)} items to {output_path}")

    print("[DONE] All types exported successfully.")


if __name__ == "__main__":
    main()
