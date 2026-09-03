#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用复旦预训练 ONNX 生成对拍场景: C 推理 vs onnxruntime 期望输出"""
import os, sys
import numpy as np
import onnxruntime as ort  # 用 mujoco 环境跑: conda activate mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(HERE, "..", "..", "mujoco", "actor", "yuntai", "上台阶3_angz+.onnx")

DEFAULT_DOF_POS = np.array([-0.23, -0.65, 0.0, 0.23, 0.65, 0.0], np.float32)
OBS_IDX = [0, 1, 3, 4]
S_ANG, S_POS, S_VEL = 0.25, 1.0, 0.05
CMD_SCALE = np.array([3.0, 0.25, 5.0], np.float32)
CLIP_OBS = 100.0

def quat_rotate_inverse_xyzw(q, v):
    qv, qw = -q[:3], q[3]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)

def build_obs(gyro, quat, dof_pos, dof_vel, cmd, last_actions):
    g_body = quat_rotate_inverse_xyzw(quat, np.array([0.0, 0.0, -1.0], np.float32))
    cmd_raw = np.clip(cmd, [-2.8, -6.0, 0.14], [2.8, 6.0, 0.25])
    obs_raw = np.concatenate([
        gyro * S_ANG, g_body, cmd_raw * CMD_SCALE,
        (np.array([dof_pos[i] for i in OBS_IDX]) - np.array([DEFAULT_DOF_POS[i] for i in OBS_IDX])) * S_POS,
        dof_vel * S_VEL, last_actions,
    ]).astype(np.float32)
    return obs_raw, np.clip(obs_raw, -CLIP_OBS, CLIP_OBS).astype(np.float32)

def main():
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    ins = {i.name: i for i in sess.get_inputs()}
    obs_name, hist_name = "obs", "obs_history"

    rng = np.random.default_rng(7)
    steps = 6
    history = np.zeros((5, 25), np.float32)   # 与训练一致: 零初始化
    last_actions = np.zeros(6, np.float32)
    seq, expected = [], []

    for _ in range(steps):
        gyro = rng.uniform(-1.5, 1.5, 3).astype(np.float32)
        qr = rng.uniform(-1, 1, 4).astype(np.float32)
        quat = qr / np.linalg.norm(qr)
        dof_pos = (rng.uniform(-1.2, 1.2, 6) + DEFAULT_DOF_POS).astype(np.float32)
        dof_vel = rng.uniform(-8, 8, 6).astype(np.float32)
        cmd = np.array([rng.uniform(-1, 1), rng.uniform(-2, 2), rng.uniform(0.14, 0.25)], np.float32)
        seq.append((gyro, quat, dof_pos, dof_vel, cmd))

        obs_raw, obs = build_obs(gyro, quat, dof_pos, dof_vel, cmd, last_actions)
        history[:-1] = history[1:]
        history[-1] = obs_raw
        act = sess.run(None, {obs_name: obs.reshape(1, -1).astype(np.float32),
                              hist_name: history.reshape(1, -1).astype(np.float32)})[0]
        act = np.clip(act.ravel(), -100, 100).astype(np.float32)
        last_actions = act.copy()
        expected.append(act)

    with open(os.path.join(HERE, "scenario.bin"), "wb") as f:
        f.write(np.int32(steps).tobytes())
        for (gyro, quat, dof_pos, dof_vel, cmd), a in zip(seq, expected):
            f.write(gyro.tobytes()); f.write(quat.tobytes())
            f.write(dof_pos.tobytes()); f.write(dof_vel.tobytes())
            f.write(cmd.tobytes()); f.write(a.tobytes())
    print("[gen] scenario.bin 已生成 (基准=复旦ONNX), 期望动作:")
    for i, a in enumerate(expected):
        print(f"  step{i}: {np.round(a,5)}")

if __name__ == "__main__":
    main()
