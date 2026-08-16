"""
无人值守自动实验流水线
- 顺序执行所有实验配置
- 自动处理失败（跳过继续）
- 每个配置完成后飞书通知
- 全部完成后发送汇总报告

实验队列 (预计总时间 4-6 小时):
  6-XOR 400k  (baseline, fc, v2)  ~20min
  6-XOR 600k  (baseline, fc, v2)  ~30min
  8-XOR 1M    (baseline, fc, v2)  ~50min
  8-XOR 2M    (baseline, fc, v2)  ~90min
"""

import sys, os, csv, json
from time import time, strftime
from datetime import datetime
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
from utils.PUFs import XORPUFModel
from utils.data import makeData, splitDataLoader
from utils.split_attack import splitXORAPUF
from utils.iterative_attack_v2 import IterativeSplitAttackV2
from utils.feishu_notifier import get_notifier

# ---- 配置 ----
RESULTS_CSV = "results/experiment_results.csv"
QUEUE_FILE = "results/experiment_queue_status.json"
feishu = get_notifier()

# 实验队列: (xor_num, data_size, epochs)
QUEUE = [
    (6, 400_000, 500),
    (6, 600_000, 500),
    (8, 1_000_000, 500),
    (8, 2_000_000, 500),
]

MODELS = ["NNModel_baseline", "NNModel_fc", "IterSplit_v2"]


def notify(msg: str, msg_type: str = "general"):
    """发送飞书通知（免打扰时段自动跳过）"""
    now = datetime.now()
    hour = now.hour
    if 8 <= hour < 23:
        feishu.send_text(msg, msg_type=msg_type)
    else:
        print(f"[Feishu Quiet] {msg[:80]}...")


def save_queue_status(config_idx: int, model_idx: int, status: str):
    """保存队列状态用于断点续跑"""
    with open(QUEUE_FILE, "w") as f:
        json.dump({
            "config_idx": config_idx,
            "model_idx": model_idx,
            "status": status,
            "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
        }, f)


def run_one(xor_num: int, data_size: int, model_name: str, epochs: int) -> dict:
    """运行单次实验，异常时返回失败结果"""
    torch.cuda.empty_cache()

    try:
        puf = XORPUFModel.randomSample(xor_num, 64)
        dataset = makeData(puf, data_size)
        train_loader, valid_loader, test_loader = splitDataLoader(dataset)

        if model_name == "IterSplit_v2":
            attacker = IterativeSplitAttackV2(
                train_loader, valid_loader, test_loader,
                xor_num, 64, dropout_p=0.1, mc_samples=10, mc_interval=3,
            )
            dt, hist = attacker.train(epochs=epochs, device="cuda", log_interval=100)
            acc = attacker.test()
            params = sum(p.numel() for p in attacker.model.parameters())
        else:
            attacker = splitXORAPUF(train_loader, valid_loader, test_loader, xor_num, 64)
            dt = attacker.train(epochs=epochs, device="cuda", log=100, model_type=model_name)
            acc = attacker.test()
            params = sum(p.numel() for p in attacker.model.parameters())

        success = 1 if acc > 0.90 else 0
        return {"xor": xor_num, "data": data_size, "model": model_name,
                "acc": acc, "success": success, "time": dt, "params": params,
                "error": None}

    except torch.cuda.OutOfMemoryError:
        return {"xor": xor_num, "data": data_size, "model": model_name,
                "acc": 0, "success": 0, "time": 0, "params": 0, "error": "OOM"}
    except Exception as e:
        return {"xor": xor_num, "data": data_size, "model": model_name,
                "acc": 0, "success": 0, "time": 0, "params": 0,
                "error": str(e)[:100]}


def save_result(r: dict):
    """追加结果到 CSV"""
    os.makedirs("results", exist_ok=True)
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "xor", "data", "model", "acc", "success",
                       "time_s", "params", "error"])
        w.writerow([
            strftime("%Y-%m-%d %H:%M:%S"),
            r["xor"], r["data"], r["model"],
            f"{r['acc']*100:.2f}%" if r["acc"] > 0 else "FAIL",
            r["success"],
            f"{r['time']:.1f}",
            r["params"],
            r.get("error", ""),
        ])


def main():
    start_time = time()
    all_results = []

    # 检查断点续跑
    config_start = 0
    model_start = 0
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE) as f:
            state = json.load(f)
            if state["status"] == "running":
                config_start = state["config_idx"]
                model_start = state["model_idx"]
                print(f"从断点续跑: config={config_start}, model={model_start}")

    total = len(QUEUE) * len(MODELS)
    count = config_start * len(MODELS) + model_start

    print(f"{'='*60}")
    print(f"  自动实验流水线启动")
    print(f"  队列: {len(QUEUE)} 配置 × {len(MODELS)} 模型 = {total} 实验")
    print(f"  从第 {count+1}/{total} 开始")
    print(f"{'='*60}")

    notify(f"🤖 无人值守模式启动\n队列: {len(QUEUE)}配置 × {len(MODELS)}模型\n预计4-6小时",
           "milestone")

    for ci in range(config_start, len(QUEUE)):
        xor_num, data_size, epochs = QUEUE[ci]
        config_results = []

        for mi in range(model_start if ci == config_start else 0, len(MODELS)):
            model_name = MODELS[mi]
            count += 1

            print(f"\n[{count}/{total}] {xor_num}-XOR {data_size//1000}k {model_name}")
            save_queue_status(ci, mi, "running")

            t0 = time()
            result = run_one(xor_num, data_size, model_name, epochs)
            elapsed = time() - t0

            save_result(result)
            config_results.append(result)
            all_results.append(result)

            # 状态输出
            if result["error"]:
                status_icon = "💥"
                status_text = f"FAIL ({result['error']})"
            elif result["success"]:
                status_icon = "✅"
                status_text = f"{result['acc']*100:.2f}%"
            else:
                status_icon = "❌"
                status_text = f"{result['acc']*100:.2f}%"

            print(f"  {status_icon} {model_name}: {status_text} | "
                  f"{elapsed:.0f}s | {result['params']:,} params")

            # 每完成一个配置的所有模型，发通知
            if mi == len(MODELS) - 1:
                successes = sum(1 for r in config_results if r["success"])
                best_acc = max((r["acc"] for r in config_results if r["acc"] > 0), default=0)
                notify(
                    f"📊 {xor_num}-XOR {data_size//1000}k 完成\n"
                    f"成功: {successes}/{len(MODELS)}\n"
                    f"最佳: {best_acc*100:.2f}%\n"
                    f"进度: {count}/{total}",
                    "experiment_complete"
                )

        model_start = 0  # 下一个配置从头开始

    # ---- 全部完成 ----
    total_time = time() - start_time
    n_success = sum(1 for r in all_results if r["success"])

    # 汇总
    print(f"\n{'='*60}")
    print(f"  全部完成! 耗时 {total_time/3600:.1f}h")
    print(f"  成功: {n_success}/{len(all_results)}")
    print(f"{'='*60}")

    summary_lines = [
        f"**耗时**: {total_time/3600:.1f} 小时",
        f"**成功**: {n_success}/{len(all_results)}",
        "",
        "**各配置最佳**:",
    ]
    for xor_num, data_size, _ in QUEUE:
        config_best = max(
            (r for r in all_results
             if r["xor"] == xor_num and r["data"] == data_size and r["acc"] > 0),
            key=lambda r: r["acc"], default=None
        )
        if config_best:
            summary_lines.append(
                f"  {xor_num}-XOR {data_size//1000}k: "
                f"{config_best['model']} = {config_best['acc']*100:.2f}%"
            )

    try:
        feishu.send_card(
            "🏁 PUF实验全部完成",
            summary_lines,
            color="green",
            msg_type="milestone"
        )
    except Exception:
        pass

    # 清理状态文件
    if os.path.exists(QUEUE_FILE):
        os.remove(QUEUE_FILE)

    # 保存最终结果 JSON
    with open("results/final_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
