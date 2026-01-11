#!/bin/bash

BASE_DIR="output/prompts"

cd "$BASE_DIR" || exit 1

for d in shift_*; do
    # 只处理目录
    [ -d "$d" ] || continue

    # 提取 mu=... 和 n=...
    mu=$(echo "$d" | sed -n 's/.*_mu_\([0-9.]*\)_n_.*/\1/p')
    n=$(echo "$d" | sed -n 's/.*_n_\([0-9]*\)_resolution.*/\1/p')

    # 若匹配失败则跳过
    if [[ -z "$mu" || -z "$n" ]]; then
        echo "[SKIP] $d"
        continue
    fi

    # 构造新的目录名：把 n 和 mu 换位置
    new=$(echo "$d" | sed -E "s/_mu_[0-9.]+_n_[0-9]+/_n_${n}_mu_${mu}/")

    # 执行重命名
    echo "$d → $new"
    mv "$d" "$new"
done
