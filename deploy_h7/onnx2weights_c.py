#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把复旦预训练 ONNX (上台阶3_angz+.onnx) 转换成 rl_policy.c 用的 policy_weights.h
用法: python onnx2weights_c.py --onnx <xxx.onnx> [--out policy_weights.h]

网络结构(已验证与 rl_policy.c 一致):
    encoder: obs_history[125] -> 128(ELU) -> 64(ELU) -> 3(latent)
    actor  : obs[25]+latent[3]=28 -> 128(ELU) -> 64(ELU) -> 32(ELU) -> 6
"""
import argparse
import sys
sys.path.insert(0, "/home/zx/isaacgym/.pylibs")  # 沙箱环境: onnx 装在 .pylibs
import numpy as np
import onnx


# 部署常量 —— 与 mujoco/python_tools/onnx_mj_chuanlian.py 的 plane 预设一致
DEFAULT_DOF_POS = [-0.23, -0.65, 0.0, 0.23, 0.65, 0.0]
OBS_DOF_POS_IDXS = [0, 1, 3, 4]
COMMAND_SCALE = [3.0, 0.25, 5.0]
P_GAINS = [15.0, 15.0, 0.0, 15.0, 15.0, 0.0]
D_GAINS = [1.0, 1.0, 0.1, 1.0, 1.0, 0.1]
POS_ACTION_SCALE = 0.5
VEL_ACTION_SCALE = 10.0
TAU_MAX = 30.0
CLIP_ACTIONS = 100.0
CLIP_OBS = 100.0
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
NUM_OBS, HIST_LEN, LATENT, NUM_ACT = 25, 5, 3, 6


def fmt_matrix(name, mat):
    rows, cols = mat.shape
    lines = [f"static const float {name}[{rows}][{cols}] = {{"]
    for r in range(rows):
        vals = ", ".join(f"{v:.8e}f" for v in mat[r])
        lines.append(f"    {{{vals}}},")
    lines.append("};")
    return "\n".join(lines)


def fmt_vec(name, vec):
    vals = ", ".join(f"{v:.8e}f" for v in vec)
    return f"static const float {name}[{len(vec)}] = {{{vals}}};"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", default="policy_weights.h")
    args = ap.parse_args()

    m = onnx.load(args.onnx)
    init = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}

    expected = {
        "encoder.0.weight": (128, 125), "encoder.0.bias": (128,),
        "encoder.2.weight": (64, 128),  "encoder.2.bias": (64,),
        "encoder.4.weight": (3, 64),    "encoder.4.bias": (3,),
        "actor.0.weight": (128, 28),    "actor.0.bias": (128,),
        "actor.2.weight": (64, 128),    "actor.2.bias": (64,),
        "actor.4.weight": (32, 64),     "actor.4.bias": (32,),
        "actor.6.weight": (6, 32),      "actor.6.bias": (6,),
    }
    for name, shape in expected.items():
        if name not in init:
            raise SystemExit(f"缺少权重: {name}")
        if tuple(init[name].shape) != shape:
            raise SystemExit(f"形状不符: {name} {init[name].shape} != {shape}")

    p = []
    p.append("// 由 onnx2weights_c.py 从复旦预训练 ONNX 自动生成 —— 请勿手改")
    p.append(f"// 来源: {args.onnx}")
    p.append("#pragma once")
    p.append(f"#define RL_NUM_OBS {NUM_OBS}")
    p.append(f"#define RL_HISTORY_LEN {HIST_LEN}")
    p.append(f"#define RL_NUM_ENCODER_OBS {NUM_OBS * HIST_LEN}")
    p.append(f"#define RL_LATENT_DIM {LATENT}")
    p.append(f"#define RL_NUM_ACTIONS {NUM_ACT}")
    p.append(f"#define RL_TAU_MAX {TAU_MAX:.1f}f")
    p.append(f"#define RL_CLIP_ACTIONS {CLIP_ACTIONS:.1f}f")
    p.append(f"#define RL_CLIP_OBS {CLIP_OBS:.1f}f")
    p.append(f"#define RL_POS_ACTION_SCALE {POS_ACTION_SCALE:.4f}f")
    p.append(f"#define RL_VEL_ACTION_SCALE {VEL_ACTION_SCALE:.4f}f")
    p.append(f"#define RL_OBS_SCALE_ANG_VEL {OBS_SCALE_ANG_VEL:.6f}f")
    p.append(f"#define RL_OBS_SCALE_DOF_POS {OBS_SCALE_DOF_POS:.6f}f")
    p.append(f"#define RL_OBS_SCALE_DOF_VEL {OBS_SCALE_DOF_VEL:.6f}f")
    p.append("")
    p.append(fmt_vec("RL_DEFAULT_DOF_POS", np.array(DEFAULT_DOF_POS, np.float32)))
    p.append(fmt_vec("RL_OBS_DOF_POS_IDXS", np.array(OBS_DOF_POS_IDXS, np.int32)))
    p.append(fmt_vec("RL_COMMAND_SCALE", np.array(COMMAND_SCALE, np.float32)))
    p.append(fmt_vec("RL_P_GAINS", np.array(P_GAINS, np.float32)))
    p.append(fmt_vec("RL_D_GAINS", np.array(D_GAINS, np.float32)))
    p.append("")

    enc_map = [("encoder.0", "ENC_W0", "ENC_B0"), ("encoder.2", "ENC_W1", "ENC_B1"),
               ("encoder.4", "ENC_W2", "ENC_B2")]
    act_map = [("actor.0", "ACT_W0", "ACT_B0"), ("actor.2", "ACT_W1", "ACT_B1"),
               ("actor.4", "ACT_W2", "ACT_B2"), ("actor.6", "ACT_W3", "ACT_B3")]
    for prefix, wn, bn in enc_map + act_map:
        p.append(fmt_matrix(wn, init[f"{prefix}.weight"]))
        p.append(fmt_vec(bn, init[f"{prefix}.bias"]))
        p.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(p))
    total = sum(v.size for v in init.values())
    print(f"[ok] 已写出 {args.out}  ({total} 参数, 约 {total*4//1024} KB float32)")


if __name__ == "__main__":
    main()
