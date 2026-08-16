"""
实验运行器 — IterativeSplitAttack v1 vs v2 对比

运行方式:
    python run_v2_experiment.py

对比:
    - v1 (原始迭代拆分攻击)
    - v2 (置信度校准版)

配置:
    - 2-XOR, 4-XOR (快速验证)
    - 6-XOR (核心目标)
"""

import os
import sys
import csv
import json
from time import time

# 确保项目路径在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.PUFs import XORPUFModel
from utils.data import makeData, splitDataLoader
from utils.iterative_attack import IterativeSplitAttack
from utils.iterative_attack_v2 import IterativeSplitAttackV2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_experiment(
    xor_num: int,
    data_size: int,
    model_version: str,  # "v1" or "v2"
    attempt: int,
    puf_length: int = 64,
    epochs: int = 500,
    device: str = "cuda",
) -> dict:
    """
    运行单次实验。

    Returns:
        dict with keys: accuracy, success, duration, avg_conf, params
    """
    print(f"\n{'='*60}")
    print(f"  实验: {xor_num}-XOR, {data_size//1000}k data, "
          f"{model_version}, attempt {attempt}")
    print(f"{'='*60}")

    # 生成 PUF 和数据集
    PUF_sample = XORPUFModel.randomSample(xor_num, puf_length)
    dataset = makeData(PUF_sample, data_size)
    train_loader, valid_loader, test_loader = splitDataLoader(dataset)

    # 选择攻击器版本
    if model_version == "v2":
        attacker = IterativeSplitAttackV2(
            train_loader, valid_loader, test_loader,
            xor_num, puf_length,
            dropout_p=0.1,
            mc_samples=10,
            mc_interval=3,
        )
        train_duration, history = attacker.train(
            epochs=epochs, device=device, log_interval=50
        )
        accuracy = attacker.test()
    else:
        attacker = IterativeSplitAttack(
            train_loader, valid_loader, test_loader,
            xor_num, puf_length,
        )
        train_duration = attacker.train(epochs=epochs, device=device, log=50)
        accuracy = attacker.test()

    success = 1 if accuracy > 0.90 else 0

    # v2 的额外指标
    avg_conf = None
    if model_version == "v2" and history:
        avg_conf = history[-1].get("avg_conf", None)

    result = {
        "XOR_num": xor_num,
        "Data_Size": data_size,
        "Model": f"IterSplit_{model_version}",
        "Attempt": attempt,
        "Accuracy": accuracy,
        "Success": success,
        "Duration_s": train_duration,
        "Avg_Confidence": avg_conf,
    }

    return result


def main():
    """运行 v1 vs v2 对比实验"""

    # ---- 实验配置 ----
    # Phase 1 快速验证: 先用小配置测试
    configs = [
        (2, 64, int(50e3)),    # 2-XOR 快速验证
        (4, 64, int(200e3)),   # 4-XOR 中等难度
        # (6, 64, int(400e3)),  # 6-XOR 核心目标（等验证完再加）
    ]

    versions = ["v1", "v2"]
    num_attempts = 3  # 快速阶段每配置3次
    epochs = 500

    # ---- CSV 输出 ----
    csv_file = "results/v2_comparison_results.csv"
    os.makedirs("results", exist_ok=True)

    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "XOR_num", "Data_Size", "Model", "Attempt",
                "Accuracy", "Success", "Duration_s", "Avg_Confidence"
            ])

    # ---- 运行实验 ----
    total_experiments = len(configs) * len(versions) * num_attempts
    exp_count = 0

    for xor_num, puf_len, data_size in configs:
        for version in versions:
            for attempt in range(1, num_attempts + 1):
                exp_count += 1
                print(f"\n▶ 进度: {exp_count}/{total_experiments}")

                result = run_experiment(
                    xor_num=xor_num,
                    data_size=data_size,
                    model_version=version,
                    attempt=attempt,
                    puf_length=puf_len,
                    epochs=epochs,
                )

                # 写入 CSV
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        result["XOR_num"],
                        result["Data_Size"],
                        result["Model"],
                        result["Attempt"],
                        f"{result['Accuracy']*100:.2f}%",
                        result["Success"],
                        f"{result['Duration_s']:.2f}",
                        f"{result.get('Avg_Confidence', 'N/A')}",
                    ])

                print(f"  → Accuracy: {result['Accuracy']*100:.2f}%, "
                      f"Success: {result['Success']}, "
                      f"Time: {result['Duration_s']:.2f}s")

    print(f"\n✅ 实验完成！结果保存在 {csv_file}")


if __name__ == "__main__":
    main()
