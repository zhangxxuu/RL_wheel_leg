#!/usr/bin/env bash
# ============================================================================
#  本地定时增量拉取：把服务器上的 .pt 模型 / 训练日志 / tensorboard 曲线拉回本地
#  目的：GPU 实例被占用、关机、被释放时，模型不会丢（本地留副本）
#
#  用法：
#     bash backup_sync.sh once      # 立刻同步一次（手动拉）
#     bash backup_sync.sh start     # 后台常驻，每 5 分钟自动拉一次
#     bash backup_sync.sh status    # 看运行状态 + 最近日志
#     bash backup_sync.sh stop      # 停止
#
#  改服务器地址/端口：编辑 $BASE/backup_sync.conf
#  日志：$BASE/backup_sync.log    备份落地：$BASE/server_backup/plane
# ============================================================================
set -u

BASE="${BASE:-$HOME/isaacgym}"   # 默认 ~/isaacgym，可用环境变量 BASE 覆盖
CONF="$BASE/backup_sync.conf"
PIDFILE="$BASE/backup_sync.pid"
LOGFILE="$BASE/backup_sync.log"
DEST="$BASE/server_backup/plane"
KEY="/home/zx/.ssh/id_ed25519"
KH="$BASE/.server_known_hosts"
INTERVAL=300          # 同步间隔（秒）

HOST="183.147.142.40"
PORTS="30407"
REMOTE_DIR="/root/fudan_rl_wheel_leg/plane/"
[ -f "$CONF" ] && . "$CONF"

SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=$KH -o ConnectTimeout=6 -i $KEY"

log() { echo "[$(date '+%F %T')] $*"; }

# 在候选端口里找到能免密登录的那个（实例重启后端口会变，写进 conf 的 PORTS 即可）
find_port() {
    local p
    for p in $PORTS; do
        if ssh $SSH_OPTS -p "$p" "root@$HOST" true 2>/dev/null; then
            echo "$p"; return 0
        fi
    done
    return 1
}

do_sync() {
    local port
    if ! port=$(find_port); then
        log "服务器不可达（HOST=$HOST PORTS=$PORTS）—— 实例可能已关机/被占用/端口变了，下轮重试"
        return 1
    fi

    mkdir -p "$DEST"
    log "端口 $port 连通，开始增量同步..."
    if rsync -azm --timeout=60 \
            -e "ssh $SSH_OPTS -p $port" \
            --include='*/' \
            --include='*.pt' \
            --include='*.py' \
            --include='train*.log' \
            --include='events.out.tfevents*' \
            --exclude='*' \
            "root@$HOST:$REMOTE_DIR" "$DEST/" 2>&1 | sed 's/^/    /'; then
        local n size
        n=$(find "$DEST" -name '*.pt' 2>/dev/null | wc -l)
        size=$(du -sh "$DEST" 2>/dev/null | cut -f1)
        log "同步完成：本地已有 $n 个 .pt，总计 ${size:-0}"
    else
        log "rsync 失败（网络抖动？）下轮重试"
        return 1
    fi
}

running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

case "${1:-status}" in
    once)
        do_sync | tee -a "$LOGFILE"
        ;;
    loop)
        echo $$ > "$PIDFILE"
        log "后台同步已启动（PID $$，每 ${INTERVAL}s 一次，目标 $DEST）" >> "$LOGFILE"
        while true; do
            do_sync >> "$LOGFILE" 2>&1
            sleep "$INTERVAL"
        done
        ;;
    start)
        if running; then
            echo "已在运行（PID $(cat "$PIDFILE")），无需重复启动"
            exit 0
        fi
        setsid nohup bash "$0" loop >/dev/null 2>&1 &
        sleep 2
        if running; then
            echo "✅ 已启动（PID $(cat "$PIDFILE")），每 ${INTERVAL}s 自动拉一次"
            echo "   日志: $LOGFILE"
        else
            echo "❌ 启动失败，看日志: $LOGFILE"
        fi
        ;;
    stop)
        if running; then
            kill "$(cat "$PIDFILE")" && rm -f "$PIDFILE" && echo "已停止"
        else
            echo "没在运行"
            rm -f "$PIDFILE"
        fi
        ;;
    status)
        if running; then echo "运行中（PID $(cat "$PIDFILE")）"; else echo "未运行"; fi
        echo "--- 最近日志 ---"
        tail -6 "$LOGFILE" 2>/dev/null || echo "(暂无日志)"
        echo "--- 本地已备份 ---"
        find "$DEST" -name '*.pt' 2>/dev/null | tail -5
        du -sh "$DEST" 2>/dev/null || echo "(备份目录为空)"
        ;;
    *)
        echo "用法: bash $0 {once|start|stop|status}"
        ;;
esac
