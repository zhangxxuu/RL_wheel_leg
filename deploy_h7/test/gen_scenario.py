#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成随机权重 checkpoint + 随机场景, 用 torch 算出期望动作, 供 C 对拍"""
import sys, os

import isaacgym  # 必须先于 torch 相关 import (与训练一致)

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/isaacgym/fudan_rl_wheel_leg/plane"))

from wheel_legged_gym.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence

HERE = os.path.dirname(os.path.abspath(__file__))
NUM_OBS, HIST, LATENT, NUM_ACT = 25, 5, 3, 6
NUM_ENC = NUM_OBS * HIST

# ---- 常量 (与 export_weights_c.py / mujoco 工具一致) ----
DEFAULT_DOF_POS = np.array([-0.23, -0.65, 0.0, 0.23, 0.65, 0.0], np.float32)
OBS_IDX = [0, 1, 3, 4]
S_ANG, S_POS, S_VEL = 0.25, 1.0, 0.05
CMD_SCALE = np.array([3.0, 0.25, 5.0], np.float32)
CLIP_OBS = 100.0

def quat_rotate_inverse_xyzw(q, v):
    qv, qw = q[:3], q[3]
    t = 2.0 * np.cross(-qv, v)          # conj 后
    return v + qw * t + np.cross(-qv, t)

def build_obs(gyro, quat, dof_pos, dof_vel, cmd, last_actions):
    g_body = quat_rotate_inverse_xyzw(quat, np.array([0.0, 0.0, -1.0], np.float32))
    cmd_raw = np.clip(cmd, [-2.8, -6.0, 0.14], [2.8, 6.0, 0.25])
    obs_raw = np.concatenate([
        gyro * S_ANG,
        g_body,
        cmd_raw * CMD_SCALE,
        (np.array([dof_pos[i] for i in OBS_IDX]) - np.array([DEFAULT_DOF_POS[i] for i in OBS_IDX])) * S_POS,
        dof_vel * S_VEL,
        last_actions,
    ]).astype(np.float32)
    return obs_raw, np.clip(obs_raw, -CLIP_OBS, CLIP_OBS).astype(np.float32)

def main():
    torch.manual_seed(0)
    model = ActorCriticSequence(
        num_obs=NUM_OBS, num_critic_obs=1, num_actions=NUM_ACT,
        num_encoder_obs=NUM_ENC, latent_dim=LATENT,
        encoder_hidden_dims=[128, 64], actor_hidden_dims=[128, 64, 32],
        critic_hidden_dims=[256, 128, 64], activation="elu",
    )
    ckpt = {"model_state_dict": model.state_dict()}
    torch.save(ckpt, os.path.join(HERE, "fake_ckpt.pt"))
    print("[gen] fake_ckpt.pt 已生成")

    # 随机场景: 6 步
    rng = np.random.default_rng(42)
    steps = 6
    seq = []
    # 与训练一致: 历史缓冲零初始化 (legged_gym obs_history_buf = zeros)
    history = np.zeros((HIST, NUM_OBS), np.float32)
    last_actions = np.zeros(NUM_ACT, np.float32)
    expected = []
    with torch.no_grad():
        for _ in range(steps):
            gyro = rng.uniform(-1.5, 1.5, 3).astype(np.float32)
            quat_raw = rng.uniform(-1, 1, 4).astype(np.float32)
            quat = quat_raw / np.linalg.norm(quat_raw)
            dof_pos = (rng.uniform(-1.2, 1.2, 6) + DEFAULT_DOF_POS).astype(np.float32)
            dof_vel = rng.uniform(-8, 8, 6).astype(np.float32)
            cmd = np.array([rng.uniform(-1, 1), rng.uniform(-2, 2),
                            rng.uniform(0.14, 0.25)], np.float32)
            seq.append((gyro, quat, dof_pos, dof_vel, cmd))

            obs_raw, obs = build_obs(gyro, quat, dof_pos, dof_vel, cmd, last_actions)
            history[:-1] = history[1:]
            history[-1] = obs_raw
            o = torch.from_numpy(obs).unsqueeze(0)
            h = torch.from_numpy(history.reshape(1, -1))
            latent = model.encoder(h)
            act = model.actor(torch.cat([o, latent], dim=-1)).numpy().ravel()
            act = np.clip(act, -100, 100).astype(np.float32)
            last_actions = act.copy()
            expected.append(act)

    # 写场景文件: [n_steps][每步: gyro3 quat4 dof_pos6 dof_vel6 cmd3 + 期望 actions6] (交错, 与 test_main.c 一致)
    with open(os.path.join(HERE, "scenario.bin"), "wb") as f:
        f.write(np.int32(steps).tobytes())
        for (gyro, quat, dof_pos, dof_vel, cmd), a in zip(seq, expected):
            f.write(gyro.tobytes()); f.write(quat.tobytes())
            f.write(dof_pos.tobytes()); f.write(dof_vel.tobytes())
            f.write(cmd.tobytes())
            f.write(a.tobytes())
    print(f"[gen] scenario.bin 已生成, {steps} 步, 期望动作:")
    for i, a in enumerate(expected):
        print(f"  step{i}: {np.round(a, 5)}")

if __name__ == "__main__":
    main()
