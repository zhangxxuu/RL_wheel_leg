#!/usr/bin/env bash
# ============================================================================
#  本机把备份里的 .pt 导出成 ONNX —— 纯 CPU，不需要服务器/GPU，随时可用
#
#  用法（在【本机】任意目录执行）：
#     bash ~/isaacgym/export_local.sh --list
#         列出本地备份里有哪些 run 和可用 checkpoint
#
#     bash ~/isaacgym/export_local.sh --run Sep11_19-34-22_ --ckpt 1300
#         默认输出到 ~/isaacgym/onnx/Sep11_19-34-22__1300.onnx
#
#     bash ~/isaacgym/export_local.sh --run Sep11_19-34-22_ --ckpt 1300 --out /tmp/abc.onnx
#         导出到指定文件（目录不存在会自动创建）
#
#     bash ~/isaacgym/export_local.sh --run Sep11_19-34-22_ --ckpt 1300 --out-dir ~/桌面
#         只指定目录，文件名自动生成 <run>_<ckpt>.onnx
#
#  可用环境变量：PYTHON（解释器，默认用 leg 环境）、BASE（默认 ~/isaacgym）
# ============================================================================
set -u

BASE="${BASE:-$HOME/isaacgym}"
PLANE="$BASE/fudan_rl_wheel_leg/plane"
LOG_ROOT="$BASE/server_backup/plane/logs/wheel_legged"
OUT_DIR_DEFAULT="$BASE/onnx"
PY="${PYTHON:-$HOME/miniconda3/envs/leg/bin/python}"
export PYTHONPATH="$BASE/.pylibs${PYTHONPATH:+:$PYTHONPATH}"

RUN=""; CKPT=""; OUT=""; OUT_DIR=""; LIST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --run)      RUN="$2"; shift 2 ;;
        --ckpt)     CKPT="$2"; shift 2 ;;
        --out)      OUT="$2"; shift 2 ;;
        --out-dir)  OUT_DIR="$2"; shift 2 ;;
        --list)     LIST=1; shift ;;
        -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "未知参数: $1（用 --help 看用法）"; exit 1 ;;
    esac
done

# ---- 列出可用内容 ----
if [ "$LIST" = 1 ]; then
    [ -d "$LOG_ROOT" ] || { echo "❌ 没有本地备份目录: $LOG_ROOT"; exit 1; }
    echo "本地备份里的 run（含 .pt 的）："
    for d in "$LOG_ROOT"/*/; do
        n=$(ls "$d"model_*.pt 2>/dev/null | wc -l)
        [ "$n" -gt 0 ] || continue
        ckpts=$(ls "$d"model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tr '\n' ' ')
        printf "  %-40s %s\n" "$(basename "$d")" "[$ckpts]"
    done
    exit 0
fi

[ -n "$RUN" ] || { echo "❌ 必须指定 --run（或先用 --list 看有哪些）"; exit 1; }
[ -n "$CKPT" ] || { echo "❌ 必须指定 --ckpt"; exit 1; }

PT="$LOG_ROOT/$RUN/model_$CKPT.pt"
[ -f "$PT" ] || { echo "❌ 本地备份里没有这个 checkpoint: $PT"; echo "   用 --list 看看该 run 有哪些"; exit 1; }

# ---- 决定输出路径 ----
if [ -z "$OUT" ]; then
    [ -n "$OUT_DIR" ] || OUT_DIR="$OUT_DIR_DEFAULT"
    mkdir -p "$OUT_DIR" || { echo "❌ 无法创建目录 $OUT_DIR"; exit 1; }
    OUT="$OUT_DIR/${RUN}_${CKPT}.onnx"
fi
mkdir -p "$(dirname "$OUT")" 2>/dev/null || true

echo "输入: $PT"
echo "输出: $OUT"
cd "$PLANE" || exit 1
"$PY" export_onnx/export_onnx.py \
    --log_root "$LOG_ROOT" \
    --load_run "$RUN" --checkpoint "$CKPT" --out "$OUT" 2>&1 | tail -2

if [ -f "$OUT" ]; then
    echo "✅ 导出成功: $(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")  ($(du -h "$OUT" | cut -f1))"
    echo
    echo "播放（本机 mujoco）："
    echo "  conda activate mujoco"
    echo "  cd $BASE/fudan_rl_wheel_leg/mujoco"
    echo "  python python_tools/onnx_mj_chuanlian.py --onnx $OUT"
else
    echo "❌ 导出失败"; exit 1
fi
