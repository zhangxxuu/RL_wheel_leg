#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练 checkpoint (.pt) 导出策略权重为 C 头文件，供 STM32H7 下位机推理使用。
用法:
    python export_weights_c.py --pt <model_XXX.pt> [--out policy_weights.h] [--verify]
生成: policy_weights.h (encoder+actor 权重/偏置 + 全部部署常量)

注意: 本脚本假设网络结构与 fudan_rl_wheel_leg 的 ActorCriticSequence 一致:
    encoder: 125 -> 128(ELU) -> 64(ELU) -> 3
    actor  : 28  -> 128(ELU) -> 64(ELU) -> 32(ELU) -> 6
    (obs 25 维 + latent 3 维拼接后进入 actor)
"""
import argparse
import os

import torch
import numpy as np

# ---------- 与训练完全一致的网络参数 (见 export_onnx.py) ----------
NUM_OBS = 25
OBS_HISTORY_LEN = 5
LATENT_DIM = 3
NUM_ACTIONS = 6
NUM_ENCODER_OBS = NUM_OBS * OBS_HISTORY_LEN  # 125
ENCODER_HIDDEN = [128, 64]
ACTOR_HIDDEN = [128, 64, 32]
ACTIVATION = "elu"

# ---------- 部署常量 (与 mujoco/python_tools/onnx_mj_chuanlian.py 对齐) ----------
DOF_NAMES = ["lf0_Joint", "lf1_Joint", "l_wheel_Joint",
             "rf0_Joint", "rf1_Joint", "r_wheel_Joint"]
DEFAULT_DOF_POS = [-0.23, -0.65, 0.0, 0.23, 0.65, 0.0]
OBS_DOF_POS_IDXS = [0, 1, 3, 4]
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
COMMAND_SCALE = [3.0, 0.25, 5.0]     # lin_vel, yaw_vel, height
POS_ACTION_SCALE = 0.5
VEL_ACTION_SCALE = 10.0
P_GAINS = [15.0, 15.0, 0.0, 15.0, 15.0, 0.0]   # plane 预设
D_GAINS = [1.0, 1.0, 0.1, 1.0, 1.0, 0.1]
TAU_MAX = 30.0
CLIP_ACTIONS = 100.0
CLIP_OBSERVATIONS = 100.0
LIN_VEL_MIN, LIN_VEL_MAX = -2.8, 2.8
YAW_VEL_MIN, YAW_VEL_MAX = -6.0, 6.0
HEIGHT_MIN, HEIGHT_MAX = 0.14, 0.25
DEFAULT_CMD_HEIGHT = 0.10


def build_model():
    from wheel_legged_gym.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence
    return ActorCriticSequence(
        num_obs=NUM_OBS,
        num_critic_obs=1,
        num_actions=NUM_ACTIONS,
        num_encoder_obs=NUM_ENCODER_OBS,
        latent_dim=LATENT_DIM,
        encoder_hidden_dims=ENCODER_HIDDEN,
        actor_hidden_dims=ACTOR_HIDDEN,
        critic_hidden_dims=[256, 128, 64],
        activation=ACTIVATION,
    )


def fmt_matrix(name, mat: np.ndarray, per_line=8):
    """按 C 行主序输出二维数组: type name[ROWS][COLS] = {...};"""
    rows, cols = mat.shape
    lines = [f"static const float {name}[{rows}][{cols}] = {{"]
    for r in range(rows):
        vals = ", ".join(f"{v:.8e}f" for v in mat[r])
        lines.append(f"    {{{vals}}},")
    lines.append("};")
    return "\n".join(lines)


def fmt_vec(name, vec: np.ndarray):
    vals = ", ".join(f"{v:.8e}f" for v in vec)
    return f"static const float {name}[{len(vec)}] = {{{vals}}};"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="训练 checkpoint 路径, 如 logs/wheel_legged/xxx/model_1000.pt")
    ap.add_argument("--out", default="policy_weights.h")
    ap.add_argument("--verify", action="store_true", help="导出前用 torch 前向验证")
    args = ap.parse_args()

    model = build_model()
    ckpt = torch.load(args.pt, map_location="cpu")
    state_dict = {k: v for k, v in ckpt["model_state_dict"].items()
                  if not k.startswith("critic.")}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    enc, act = model.encoder, model.actor
    # encoder: [Linear, ELU, Linear, ELU, Linear]
    e_w = [enc[0].weight.detach().numpy(), enc[2].weight.detach().numpy(), enc[4].weight.detach().numpy()]
    e_b = [enc[0].bias.detach().numpy(), enc[2].bias.detach().numpy(), enc[4].bias.detach().numpy()]
    # actor: [Linear, ELU, Linear, ELU, Linear, ELU, Linear]
    a_w = [act[0].weight.detach().numpy(), act[2].weight.detach().numpy(),
           act[4].weight.detach().numpy(), act[6].weight.detach().numpy()]
    a_b = [act[0].bias.detach().numpy(), act[2].bias.detach().numpy(),
           act[4].bias.detach().numpy(), act[6].bias.detach().numpy()]

    if args.verify:
        with torch.no_grad():
            o = torch.zeros(1, NUM_OBS)
            h = torch.zeros(1, NUM_ENCODER_OBS)
            latent = model.encoder(h)
            a = model.actor(torch.cat([o, latent], dim=-1))
        print("[verify] torch 前向 OK, 零输入 action =", a.numpy().ravel().round(4))

    parts = []
    parts.append("// 由 export_weights_c.py 自动生成 —— 请勿手改")
    parts.append("// encoder: 125->128(ELU)->64(ELU)->3 ; actor: 28->128(ELU)->64(ELU)->32(ELU)->6")
    parts.append("#pragma once")
    parts.append(f"#define RL_NUM_OBS {NUM_OBS}")
    parts.append(f"#define RL_HISTORY_LEN {OBS_HISTORY_LEN}")
    parts.append(f"#define RL_NUM_ENCODER_OBS {NUM_ENCODER_OBS}")
    parts.append(f"#define RL_LATENT_DIM {LATENT_DIM}")
    parts.append(f"#define RL_NUM_ACTIONS {NUM_ACTIONS}")
    parts.append(f"#define RL_TAU_MAX {TAU_MAX:.1f}f")
    parts.append(f"#define RL_CLIP_ACTIONS {CLIP_ACTIONS:.1f}f")
    parts.append(f"#define RL_CLIP_OBS {CLIP_OBSERVATIONS:.1f}f")
    parts.append(f"#define RL_POS_ACTION_SCALE {POS_ACTION_SCALE:.4f}f")
    parts.append(f"#define RL_VEL_ACTION_SCALE {VEL_ACTION_SCALE:.4f}f")
    parts.append(f"#define RL_OBS_SCALE_ANG_VEL {OBS_SCALE_ANG_VEL:.6f}f")
    parts.append(f"#define RL_OBS_SCALE_DOF_POS {OBS_SCALE_DOF_POS:.6f}f")
    parts.append(f"#define RL_OBS_SCALE_DOF_VEL {OBS_SCALE_DOF_VEL:.6f}f")
    parts.append("")
    parts.append(fmt_vec("RL_DEFAULT_DOF_POS", np.array(DEFAULT_DOF_POS, dtype=np.float32)))
    parts.append(fmt_vec("RL_OBS_DOF_POS_IDXS", np.array(OBS_DOF_POS_IDXS, dtype=np.int32)))
    parts.append(fmt_vec("RL_COMMAND_SCALE", np.array(COMMAND_SCALE, dtype=np.float32)))
    parts.append(fmt_vec("RL_P_GAINS", np.array(P_GAINS, dtype=np.float32)))
    parts.append(fmt_vec("RL_D_GAINS", np.array(D_GAINS, dtype=np.float32)))
    parts.append("")
    for i, (w, b) in enumerate(zip(e_w, e_b)):
        parts.append(fmt_matrix(f"ENC_W{i}", w))
        parts.append(fmt_vec(f"ENC_B{i}", b))
        parts.append("")
    for i, (w, b) in enumerate(zip(a_w, a_b)):
        parts.append(fmt_matrix(f"ACT_W{i}", w))
        parts.append(fmt_vec(f"ACT_B{i}", b))
        parts.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(parts))
    print(f"[ok] 已写出 {args.out} "
          f"(权重 {sum(w.size for w in e_w + a_w)} 参数, "
          f"浮点占用 ~{(sum(w.size for w in e_w + a_w) * 4) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
