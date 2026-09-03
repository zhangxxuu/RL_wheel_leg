#!/bin/bash
# ============================================================
# 复旦星云 fudan_rl_wheel_leg —— 环境变量一键配置
# 用法: 每次开新终端先  source ~/isaacgym/fudan_rl_wheel_leg/env.sh
# 注意: 需先  conda activate leg
# ============================================================
export PATH=/home/zx/miniconda3/envs/leg/bin:$PATH
export LD_LIBRARY_PATH=/home/zx/miniconda3/envs/leg/lib:$LD_LIBRARY_PATH

# 仓库根（plane 平地版；跑 jump 时改成 jump 目录）
export FUDAN_RL_ROOT=/home/zx/isaacgym/fudan_rl_wheel_leg/plane
export PYTHONPATH=$FUDAN_RL_ROOT:$PYTHONPATH

# gymtorch JIT 编译缓存（避免每次重新编译）
export TORCH_EXTENSIONS_DIR=/home/zx/isaacgym/.torch_ext

# matplotlib 缓存（避免写入 ~/.config 报错）
export MPLCONFIGDIR=/home/zx/isaacgym/.mplconfig

echo "[env.sh] LD_LIBRARY_PATH 已设置，PYTHONPATH=$FUDAN_RL_ROOT"
