#!/bin/bash
# ============================================================
# 复旦轮腿 RL —— 云 GPU 服务器一键部署（3090/4090 通用）
# 用法: 在租到的服务器上执行  bash setup_server.sh
# ============================================================
set -e

echo "===== [0/7] 预检 ====="
nvidia-smi || { echo "❌ 没有 nvidia-smi，请确认租的实例带 GPU 驱动"; exit 1; }
which conda || { echo "❌ 没有 conda，请先装 Miniconda (https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh)"; exit 1; }
which git || { echo "❌ 没有 git"; exit 1; }

# conda 激活（兼容多种安装路径）
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh

echo "===== [1/7] 创建 Python 3.8 环境 ====="
conda create -n leg python=3.8 -y
conda activate leg

echo "===== [2/7] 安装 Isaac Gym ====="
# 前提：本机已把 ~/isaacgym 同步到服务器 ~/isaacgym（见脚本开头注释的 rsync 命令）
if [ ! -d ~/isaacgym/python ]; then
  echo "❌ 没找到 ~/isaacgym/python，请先在本地执行："
  echo "   rsync -av ~/isaacgym/ user@服务器:~/isaacgym/"
  exit 1
fi
cd ~/isaacgym/python && pip install -e . -q

echo "===== [3/7] 修补 np.float（numpy 1.24 兼容） ====="
sed -i 's/dtype=np.float,/dtype=float,/' ~/isaacgym/python/isaacgym/torch_utils.py

echo "===== [4/7] 获取复旦仓库 ====="
cd ~
if [ -d ~/fudan_rl_wheel_leg ]; then
  echo "已存在 ~/fudan_rl_wheel_leg，跳过 clone"
else
  git clone https://github.com/yly-true/fudan_rl_wheel_leg.git
fi
cd ~/fudan_rl_wheel_leg/plane && pip install -e . -q

echo "===== [5/7] 安装训练依赖 ====="
pip install tensorboard gitpython matplotlib -q

echo "===== [6/7] 持久化环境变量（写入 leg 环境） ====="
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib
conda env config vars set TORCH_EXTENSIONS_DIR=~/isaacgym/.torch_ext
conda deactivate && conda activate leg

echo "===== [7/7] 快速自检：导入 + 列出任务 ====="
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd ~/fudan_rl_wheel_leg/plane
python - <<'EOF'
import isaacgym
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import task_registry
print("import OK, tasks:", list(task_registry.task_classes.keys()))
EOF

echo ""
echo "======================================================"
echo "✅ 部署完成！训练命令："
echo ""
echo "  conda activate leg"
echo "  cd ~/fudan_rl_wheel_leg/plane"
echo "  python wheel_legged_gym/scripts/train.py --task=wheel_legged --headless --num_envs=4096"
echo ""
echo "  # 另一终端看曲线："
echo "  tensorboard --logdir logs --port 8080"
echo "======================================================"
