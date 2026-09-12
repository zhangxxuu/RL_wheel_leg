#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ckpt.py —— 单个 checkpoint 的【定量评估】（固定指令、关噪声/域随机化）

为什么需要它：
  训练时的 tensorboard 指标受"速度课程"影响（每个 run 都会重置课程），跨 run 不可比；
  这个脚本用**固定指令序列**跑一遍，输出可横向比较的数字。

用法（在服务器 plane/ 目录下）：
  cd ~/fudan_rl_wheel_leg/plane
  python tools/eval_ckpt.py --task=wheel_legged \
      --load_run=Sep12_14-27-58_ --checkpoint=100 \
      --num_envs=64 --headless --stage_steps=200

  批量对比多个 checkpoint（结果追加到 csv）：
  for ck in 100 1300 5100; do
    python tools/eval_ckpt.py --task=wheel_legged --load_run=<run> --checkpoint=$ck \
        --num_envs=64 --headless --stage_steps=200 | tee -a eval_result.csv
  done

输出列（csv 行以 EVAL, 开头，可直接 grep）：
  stage, fall%, h_err(离地高度误差 m), vx_err(速度跟踪误差 m/s),
  yaw_err(转向误差 rad/s), a_rate(动作抖动), torque_rms, h_real(实际离地高度)
"""
import os
import shutil
import sys

import torch

import isaacgym  # noqa: F401  （必须先 import，否则 gymtorch 起不来）
from wheel_legged_gym.envs import *  # noqa: F401,F403
from wheel_legged_gym.utils import get_args, task_registry

# (标签, 目标 vx, 目标 yaw_rate, 目标离地高度)
STAGES = [
    ("站立",      0.0,  0.0, 0.18),
    ("前进1.0",   1.0,  0.0, 0.18),
    ("前进2.0",   2.0,  0.0, 0.18),
    ("后退1.0",  -1.0,  0.0, 0.18),
    ("转向1.0",   0.0,  1.0, 0.18),
    ("站立收尾",  0.0,  0.0, 0.18),
]


def split_own_args():
    """把本脚本独有的参数从 sys.argv 里摘出来，剩下的丢给框架的 get_args()。"""
    argv, rest, ours = sys.argv[1:], [], {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--stage_steps") or a.startswith("--reset_between_stages"):
            key = a.split("=")[0].lstrip("-")
            if "=" in a:
                val = a.split("=", 1)[1]
                i += 1
            else:
                val = argv[i + 1]
                i += 2
            ours[key] = val
        else:
            rest.append(a)
            i += 1
    sys.argv = [sys.argv[0]] + rest
    return ours


def make_env_cfg(task, num_envs):
    """评估用的环境配置：固定指令、关噪声、关域随机化、小规模地形。"""
    env_cfg, train_cfg = task_registry.get_cfgs(name=task)
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = 20
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    env_cfg.commands.curriculum = False      # 不让速度课程改指令
    env_cfg.noise.add_noise = False
    for k in dir(env_cfg.domain_rand):        # 关掉所有随机化（含 push）
        if k.startswith("randomize") or k == "push_robots":
            try:
                setattr(env_cfg.domain_rand, k, False)
            except Exception:
                pass
    return env_cfg, train_cfg


def main():
    ours = split_own_args()
    args = get_args()                                   # 框架参数：--task/--load_run/--checkpoint/--num_envs...
    stage_steps = int(ours.get("stage_steps", 200))
    reset_between = str(ours.get("reset_between_stages", "1")) not in ("0", "false", "False")
    num_envs = getattr(args, "num_envs", None) or 64

    print(f"\n=== 评估 {args.task} / run={args.load_run} / ckpt={args.checkpoint} "
          f"（{num_envs} envs × {stage_steps} 步/阶段）===\n")

    env_cfg, train_cfg = make_env_cfg(args.task, num_envs)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)
    is_seq = bool(runner.alg.actor_critic.is_sequence)

    obs, obs_history = env.get_observations()

    print(f"{'阶段':<10}{'摔倒率':>8}{'h_err':>9}{'vx_err':>9}{'yaw_err':>9}"
          f"{'a_rate':>9}{'torque':>9}{'h_real':>9}")
    print("-" * 74)

    lines = []
    for label, vx, yaw, h in STAGES:
        if reset_between:
            env.reset()                                  # 每个阶段从站立开始
            obs, obs_history = env.get_observations()

        falls, n = 0, 0
        acc = {"h": 0.0, "vx": 0.0, "yaw": 0.0, "ar": 0.0, "tq": 0.0, "hr": 0.0}
        prev_a = None
        with torch.no_grad():
            for _ in range(stage_steps):
                env.commands[:, 0] = vx               # 每步都写，防止被课程/重采样覆盖
                env.commands[:, 1] = yaw
                env.commands[:, 2] = h
                if is_seq:
                    actions, _ = policy(obs, obs_history)
                else:
                    actions = policy(obs)
                obs, _, _, dones, _, obs_history = env.step(actions)

                falls += int(((dones > 0) & (~env.time_out_buf)).sum().item())
                n += env.num_envs
                acc["h"] += float((env.base_height - h).abs().mean())
                acc["vx"] += float((env.base_lin_vel[:, 0] - vx).abs().mean())
                acc["yaw"] += float((env.base_ang_vel[:, 2] - yaw).abs().mean())
                acc["tq"] += float(env.torques.pow(2).mean().sqrt())
                acc["hr"] += float(env.base_height.mean())
                if prev_a is not None:
                    acc["ar"] += float((actions - prev_a).abs().mean())
                prev_a = actions.clone()

        k = float(stage_steps)
        row = dict(stage=label, fall_pct=100.0 * falls / max(n, 1),
                   h_err=acc["h"] / k, vx_err=acc["vx"] / k, yaw_err=acc["yaw"] / k,
                   a_rate=acc["ar"] / max(k - 1, 1), torque=acc["tq"] / k, h_real=acc["hr"] / k)
        print(f"{label:<10}{row['fall_pct']:>7.2f}%{row['h_err']:>9.3f}{row['vx_err']:>9.3f}"
              f"{row['yaw_err']:>9.3f}{row['a_rate']:>9.4f}{row['torque']:>9.2f}{row['h_real']:>9.3f}")
        lines.append("EVAL,{run},{ckpt},{stage},{fall:.2f},{h:.4f},{vx:.4f},{yaw:.4f},{ar:.5f},{tq:.3f},{hr:.4f}"
                     .format(run=args.load_run, ckpt=args.checkpoint, stage=label,
                             fall=row['fall_pct'], h=row['h_err'], vx=row['vx_err'],
                             yaw=row['yaw_err'], ar=row['a_rate'], tq=row['torque'], hr=row['h_real']))

    print("-" * 74)
    print("（参考：摔倒率 0% 最好；h_err 越小越贴目标高度；vx_err 越小速度跟得越准；a_rate 越小动作越平滑）")
    print("\n--- 可粘贴/汇总的 csv 行 ---")
    for l in lines:
        print(l)

    # 评估用不到日志目录：如果 make_alg_runner 建了临时目录，顺手清掉
    # （它永远是"当前时间戳"的新目录，不会碰到你已有的训练 run）
    try:
        if runner.log_dir and os.path.isdir(runner.log_dir):
            shutil.rmtree(runner.log_dir, ignore_errors=True)
            print(f"\n(已清理评估临时目录: {runner.log_dir})")
    except Exception:
        pass


if __name__ == "__main__":
    main()
