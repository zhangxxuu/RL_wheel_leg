#!/usr/bin/env bash
# ============================================================================
#  本地改完代码 → 一条命令同步到服务器
#
#  为什么不用 git pull：服务器直连 GitHub 不通（实测 HTTPS 报 GnuTLS 错误、
#  SSH 没私钥）。所以用 git bundle 走 scp —— 只传服务器缺的提交，通常几十 KB。
#
#  用法（在【本机】执行）：
#     bash ~/isaacgym/sync_to_server.sh          # 同步 main 到服务器
#     bash ~/isaacgym/sync_to_server.sh status   # 只看两边版本差异，不同步
#
#  服务器侧铁律：只 pull、不改代码（脚本会自动还原服务器上的未提交改动）
# ============================================================================
set -u

BASE="${BASE:-$HOME/isaacgym}"
CONF="$BASE/backup_sync.conf"
KEY="$HOME/.ssh/id_ed25519"
KH="$BASE/.server_known_hosts"
REPO="$BASE/fudan_rl_wheel_leg"
BUNDLE="/tmp/rl_sync.bundle"
REMOTE_BUNDLE="/root/rl_sync.bundle"
REMOTE_REPO="/root/fudan_rl_wheel_leg"

HOST="183.147.142.40"; PORTS="30407"
[ -f "$CONF" ] && . "$CONF"

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o "UserKnownHostsFile=$KH" -o ConnectTimeout=10 -i "$KEY")

find_port() {
    local p
    for p in $PORTS; do
        if ssh "${SSH_OPTS[@]}" -p "$p" "root@$HOST" true 2>/dev/null; then
            echo "$p"; return 0
        fi
    done
    return 1
}

port=$(find_port) || { echo "❌ 服务器不可达（HOST=$HOST PORTS=$PORTS）—— 实例可能关机/被占用，或端口变了（改 $CONF）"; exit 1; }
echo "服务器端口: $port"

cd "$REPO" || { echo "❌ 找不到仓库 $REPO"; exit 1; }

# ---- status：只看差异 ----
if [ "${1:-sync}" = "status" ]; then
    echo "本地 HEAD:  $(git --no-pager log --oneline -1)"
    echo "服务器状态:"
    ssh "${SSH_OPTS[@]}" -p "$port" "root@$HOST" "cd $REMOTE_REPO && export GIT_PAGER=cat && git --no-pager log --oneline -1 && echo '--- 未提交改动（应尽量为空）---' && (git status --porcelain | grep -v '^??' || echo '(无)')"
    exit 0
fi

# ---- 本地：提醒未提交改动 ----
if [ -n "$(git status --porcelain | grep -v '^??')" ]; then
    echo "⚠️ 本机有未提交的修改（不会被同步，先 commit 再跑本脚本）："
    git status --short | grep -v '^??'
fi

echo "本地 HEAD:  $(git --no-pager log --oneline -1)"

# ---- 取服务器 HEAD，先判断是否已同步 ----
remote_head=$(ssh "${SSH_OPTS[@]}" -p "$port" "root@$HOST" "cd $REMOTE_REPO && git rev-parse HEAD" 2>/dev/null || true)
local_head=$(git rev-parse HEAD)
if [ "$remote_head" = "$local_head" ]; then
    echo "✅ 服务器已是最新（$(git --no-pager log --oneline -1 | cut -c1-60)），无需同步"
    exit 0
fi

# ---- 打包增量 bundle（只含服务器缺的提交，通常几十 KB）----
rm -f "$BUNDLE"
if [ -n "$remote_head" ] && git cat-file -e "$remote_head" 2>/dev/null; then
    echo "服务器当前: $(git --no-pager log --oneline -1 "$remote_head" 2>/dev/null)"
    git bundle create "$BUNDLE" "$remote_head..main" >/dev/null 2>&1 \
        || git bundle create "$BUNDLE" main >/dev/null 2>&1
else
    echo "（服务器 HEAD 不在本地历史里，做全量打包）"
    git bundle create "$BUNDLE" main >/dev/null 2>&1 || { echo "❌ bundle 创建失败"; exit 1; }
fi
echo "bundle 大小: $(du -h "$BUNDLE" | cut -f1)"

# ---- 上传 ----
scp -P "$port" "${SSH_OPTS[@]}" "$BUNDLE" "root@$HOST:$REMOTE_BUNDLE" >/dev/null 2>&1 \
    || { echo "❌ 上传失败"; exit 1; }
echo "已上传到服务器: $REMOTE_BUNDLE"

# ---- 服务器：还原本地改动后拉取 ----
ssh "${SSH_OPTS[@]}" -p "$port" "root@$HOST" "REMOTE_REPO='$REMOTE_REPO' REMOTE_BUNDLE='$REMOTE_BUNDLE' bash -s" <<'EOS'
set -e
export GIT_PAGER=cat
cd "$REMOTE_REPO"
dirty=$(git status --porcelain | grep -v '^??' || true)
if [ -n "$dirty" ]; then
    echo "[服务器] 有未提交改动，按'只 pull 不改'原则还原："
    echo "$dirty"
    git checkout -- .
fi
git pull "$REMOTE_BUNDLE" main 2>&1 | tail -4
echo "[服务器] HEAD: $(git --no-pager log --oneline -1)"
EOS

echo "✅ 同步完成（服务器已是最新代码）"
