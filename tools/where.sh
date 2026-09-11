#!/usr/bin/env bash
# ============================================================================
#  where.sh —— 代码导航：给我一个关键词，告诉我它在哪、干什么
#
#  用法（在 ~/isaacgym/fudan_rl_wheel_leg 下执行）：
#      bash tools/where.sh 奖励            # 列出所有奖励函数（含上面的注释块）
#      bash tools/where.sh base_height     # 搜参数/关键词，带上下文
#      bash tools/where.sh 我要改站立高度    # 查手册里的"我要做什么"表
#      bash tools/where.sh param           # 打印所有可调参数所在文件:行
#
#  说明：自动带上每个函数上方的【接口】/【输出】注释块，所以你能一眼看懂它干什么。
# ============================================================================
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLANE="$ROOT/plane"
KW="${1:-help}"
CTX="${2:-3}"

show_hit() {  # $1=文件 $2=行号
    local f="$1" ln="$2"
    echo "── $f:$ln"
    awk -v n="$ln" -v c="$CTX" '
        NR>=n-c && NR<=n+c { printf "%s%5d| %s\n", (NR==n?"▶ ":"  "), NR, $0 }
    ' "$f"
    echo
}

case "$KW" in
  help|-h|"")
    sed -n '2,16p' "$0"; exit 0 ;;

  param|参数)
    echo "=== 所有可调参数（文件:行）==="
    grep -rn "^\s*[a-z_]* = \|^\s*[a-z_]*=" "$PLANE/wheel_legged_gym/envs/base/legged_robot_config.py" \
      | sed "s|$PLANE/||" | head -80
    echo
    echo "=== 任务级覆盖（这台车专属）==="
    grep -rn "^\s*[a-z_]* = " "$PLANE/wheel_legged_gym/envs/wheel_legged/wheel_legged_config.py" | sed "s|$PLANE/||"
    exit 0 ;;

  奖励|reward)
    echo "=== 奖励函数清单（配置里的权重名自动对应 _reward_<名字>）==="
    grep -n "def _reward_" "$PLANE/wheel_legged_gym/envs/base/legged_robot.py" | sed "s|$PLANE/||"
    echo
    echo "=== 权重表位置 ==="
    grep -n "class scales" -A 30 "$PLANE/wheel_legged_gym/envs/base/legged_robot_config.py" | head -35
    exit 0 ;;
esac

echo "=== 搜「$KW」==="
FILES=$(grep -rl --include="*.py" --include="*.md" \
      --exclude-dir=logs --exclude-dir=__pycache__ --exclude-dir=.git \
      "$KW" "$PLANE/wheel_legged_gym" "$PLANE/export_onnx" "$PLANE/wheel_legged_gym/scripts" "$ROOT"/*.md 2>/dev/null | head -20)
[ -z "$FILES" ] && { echo "没搜到。换个词试试（或 bash tools/where.sh param 看所有参数）"; exit 1; }

for f in $FILES; do
    while read -r ln; do
        [ -z "$ln" ] && continue
        show_hit "$f" "$ln"
    done < <(grep -n "$KW" "$f" | cut -d: -f1 | head -6)
done
