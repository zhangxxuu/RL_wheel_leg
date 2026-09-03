/*
 * rl_policy.c —— 轮腿 RL 策略下位机推理实现 (纯 C, 无外部依赖)
 * 权重来自 policy_weights.h (export_weights_c.py 生成)
 *
 * 性能估算 (STM32H7 @480MHz, 单精度):
 *   encoder: 125*128 + 128*64 + 64*3  ≈ 24.4k MACs
 *   actor  : 28*128 + 128*64 + 64*32 + 32*6 ≈ 14.0k MACs
 *   合计 ~38.5k MACs -> 每次前向 <0.15ms, 100Hz 下 CPU 占用 ~1.5%
 *   权重 float32 约 155KB, 建议放 AXI SRAM 或外部 QSPI Flash 加载.
 */
#include <math.h>
#include <string.h>

#include "rl_policy.h"
#include "policy_weights.h"

/* ---------- 内部状态 ---------- */
static float s_history[RL_HISTORY_LEN][RL_NUM_OBS]; /* 0=最旧, 最后=最新; 存未缩放 obs_raw, 与训练/mujoco工具一致 */
static float s_last_obs_raw[RL_NUM_OBS];           /* build_obs 时更新 */
static float s_last_actions[RL_NUM_ACTIONS];

/* ---------- 工具函数 ---------- */
static inline float elu(float x)
{
    /* PyTorch nn.ELU(alpha=1.0) */
    return x > 0.0f ? x : (expf(x) - 1.0f);
}

static inline float clipf(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

/* y = W*x + b, 行主序 W[out][in] */
static void matvec(const float *W, int out, int in, const float *b,
                   const float *x, float *y)
{
    for (int o = 0; o < out; o++) {
        float acc = b ? b[o] : 0.0f;
        const float *row = W + (size_t)o * in;
        for (int i = 0; i < in; i++)
            acc += row[i] * x[i];
        y[o] = acc;
    }
}

/* ---------- 公共 API ---------- */
void rl_policy_init(void)
{
    memset(s_history, 0, sizeof(s_history));
    memset(s_last_actions, 0, sizeof(s_last_actions));
}

void rl_quat_rotate_inverse(const float q[4], const float v[3], float out[3])
{
    /* 与 onnx_mj_chuanlian.py 的 quat_rotate_inverse_xyzw 完全一致:
     *   conj(q) = [-x,-y,-z,w]; t = 2 * qv x v; out = v + qw*t + qv x t */
    float qv[3] = {-q[0], -q[1], -q[2]};
    float qw = q[3];
    float t[3];
    t[0] = 2.0f * (qv[1] * v[2] - qv[2] * v[1]);
    t[1] = 2.0f * (qv[2] * v[0] - qv[0] * v[2]);
    t[2] = 2.0f * (qv[0] * v[1] - qv[1] * v[0]);
    out[0] = v[0] + qw * t[0] + (qv[1] * t[2] - qv[2] * t[1]);
    out[1] = v[1] + qw * t[1] + (qv[2] * t[0] - qv[0] * t[2]);
    out[2] = v[2] + qw * t[2] + (qv[0] * t[1] - qv[1] * t[0]);
}

void rl_policy_build_obs(const float gyro_body[3], const float quat_xyzw[4],
                         const float dof_pos[6], const float dof_vel[6],
                         const float cmd[3],
                         float obs_raw[25], float obs[25])
{
    /* obs 布局 (与训练/onnx_mj_chuanlian.py 一致):
     *  [0:3]    base_ang_vel(机体角速度) * 0.25
     *  [3:6]    投影重力 (重力[0,0,-1] 旋转到机体系)
     *  [6:9]    指令 [vx, yaw_vel, height] * command_scale
     *  [9:13]   腿关节角 (lf0,lf1,rf0,rf1) - default * 1.0
     *  [13:19]  6 关节速度 * 0.05
     *  [19:25]  last_actions (6)
     */
    float g_world[3] = {0.0f, 0.0f, -1.0f};
    float g_body[3];
    int k = 0;

    rl_quat_rotate_inverse(quat_xyzw, g_world, g_body);

    for (int i = 0; i < 3; i++)
        obs_raw[k++] = gyro_body[i] * RL_OBS_SCALE_ANG_VEL;
    for (int i = 0; i < 3; i++)
        obs_raw[k++] = g_body[i];
    for (int i = 0; i < 3; i++)
        obs_raw[k++] = cmd[i] * RL_COMMAND_SCALE[i];
    for (int j = 0; j < 4; j++) {
        int idx = RL_OBS_DOF_POS_IDXS[j];
        obs_raw[k++] = (dof_pos[idx] - RL_DEFAULT_DOF_POS[idx]) * RL_OBS_SCALE_DOF_POS;
    }
    for (int i = 0; i < 6; i++)
        obs_raw[k++] = dof_vel[i] * RL_OBS_SCALE_DOF_VEL;
    for (int i = 0; i < RL_NUM_ACTIONS; i++)
        obs_raw[k++] = s_last_actions[i];

    memcpy(s_last_obs_raw, obs_raw, RL_NUM_OBS * sizeof(float));
    for (int i = 0; i < 25; i++)
        obs[i] = clipf(obs_raw[i], -RL_CLIP_OBS, RL_CLIP_OBS);
}

void rl_policy_step(const float obs[25], float actions[6])
{
    /* 1. 历史缓冲: 左移一帧, 最新帧放最后; 存未缩放 obs_raw (与 mujoco 工具一致) */
    for (int t = 0; t < RL_HISTORY_LEN - 1; t++)
        memcpy(s_history[t], s_history[t + 1], RL_NUM_OBS * sizeof(float));
    memcpy(s_history[RL_HISTORY_LEN - 1], s_last_obs_raw, RL_NUM_OBS * sizeof(float));

    float hist[RL_NUM_ENCODER_OBS];
    for (int t = 0; t < RL_HISTORY_LEN; t++)
        memcpy(&hist[t * RL_NUM_OBS], s_history[t], RL_NUM_OBS * sizeof(float));

    /* 2. encoder: 125 -> 128(ELU) -> 64(ELU) -> 3 */
    float h1[128], h2[64], latent[RL_LATENT_DIM];
    matvec(&ENC_W0[0][0], 128, 125, ENC_B0, hist, h1);
    for (int i = 0; i < 128; i++) h1[i] = elu(h1[i]);
    matvec(&ENC_W1[0][0], 64, 128, ENC_B1, h1, h2);
    for (int i = 0; i < 64; i++) h2[i] = elu(h2[i]);
    matvec(&ENC_W2[0][0], RL_LATENT_DIM, 64, ENC_B2, h2, latent);

    /* 3. actor 输入: obs(25) + latent(3) = 28 */
    float ain[28];
    memcpy(ain, obs, 25 * sizeof(float));
    memcpy(ain + 25, latent, RL_LATENT_DIM * sizeof(float));

    /* 4. actor: 28 -> 128(ELU) -> 64(ELU) -> 32(ELU) -> 6 */
    float a1[128], a2[64], a3[32], aout[RL_NUM_ACTIONS];
    matvec(&ACT_W0[0][0], 128, 28, ACT_B0, ain, a1);
    for (int i = 0; i < 128; i++) a1[i] = elu(a1[i]);
    matvec(&ACT_W1[0][0], 64, 128, ACT_B1, a1, a2);
    for (int i = 0; i < 64; i++) a2[i] = elu(a2[i]);
    matvec(&ACT_W2[0][0], 32, 64, ACT_B2, a2, a3);
    for (int i = 0; i < 32; i++) a3[i] = elu(a3[i]);
    matvec(&ACT_W3[0][0], RL_NUM_ACTIONS, 32, ACT_B3, a3, aout);

    /* 5. 输出 clip + 记录 last_actions */
    for (int i = 0; i < RL_NUM_ACTIONS; i++) {
        actions[i] = clipf(aout[i], -RL_CLIP_ACTIONS, RL_CLIP_ACTIONS);
        s_last_actions[i] = actions[i];
    }
}

void rl_policy_apply_actions(const float actions[6], const float dof_pos[6],
                             const float dof_vel[6], float tau[6])
{
    for (int i = 0; i < RL_NUM_ACTIONS; i++) {
        float act = clipf(actions[i], -RL_CLIP_ACTIONS, RL_CLIP_ACTIONS);
        float pos_ref = act * RL_POS_ACTION_SCALE; /* 腿: 位置目标增量 */
        float vel_ref = act * RL_VEL_ACTION_SCALE; /* 轮: 速度目标 */
        /* 腿(0,1,3,4)用位置PD, 轮(2,5)用速度环; P_GAINS/D_GAINS 里已经置0区分 */
        tau[i] = RL_P_GAINS[i] * (pos_ref + RL_DEFAULT_DOF_POS[i] - dof_pos[i])
               + RL_D_GAINS[i] * (vel_ref - dof_vel[i]);
        tau[i] = clipf(tau[i], -RL_TAU_MAX, RL_TAU_MAX);
    }
}
