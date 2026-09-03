/*
 * rl_policy.h —— 轮腿 RL 策略下位机推理接口 (STM32H7)
 * 配套: policy_weights.h (由 export_weights_c.py 生成)
 *
 * 数据流(100Hz):
 *   传感器(IMU/编码器/CAN电机反馈) -> rl_policy_build_obs() 组装 25 维 obs
 *   -> rl_policy_step() 维护 5 帧历史 + 神经网络推理 -> 6 维 action
 *   -> rl_policy_apply_actions() 解码为关节力矩 -> 电流环/CAN 下发
 *
 * 注意:
 *  1. 与训练对齐的所有常量在 policy_weights.h 中, 不要手改;
 *  2. 控制频率必须是 100Hz (训练 sim dt=0.005, decimation=2);
 *  3. 关节角/速度单位: 弧度, 弧度/s; 指令: 线速度 m/s, 偏航角速度 rad/s, 高度 m.
 */
#ifndef RL_POLICY_H
#define RL_POLICY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 初始化: 清空历史缓冲与 last_actions */
void rl_policy_init(void);

/*
 * 组装 25 维原始 obs (未缩放的历史帧用 obs_raw 存, 缩放+截断后的喂给网络)
 * 输入(需要你在主控里填好):
 *   gyro_body[3]    机体坐标系角速度 rad/s   (与训练一致, 注意 IMU 安装方向)
 *   quat_xyzw[4]    姿态四元数 (x,y,z,w)
 *   dof_pos[6]      6 关节角 rad: [lf0, lf1, l_wheel, rf0, rf1, r_wheel]
 *   dof_vel[6]      6 关节速度 rad/s (同序; 轮子用电机速度反馈)
 *   cmd[3]          指令: [线速度 m/s, 偏航角速度 rad/s, 车身高度 m]
 * 输出:
 *   obs_raw[25]     未缩放 obs (写入历史缓冲用)
 *   obs[25]         缩放+截断 obs (喂给网络)
 */
void rl_policy_build_obs(const float gyro_body[3], const float quat_xyzw[4],
                         const float dof_pos[6], const float dof_vel[6],
                         const float cmd[3],
                         float obs_raw[25], float obs[25]);

/* 推理一步: 内部维护历史缓冲, 输出 6 维动作 (已 clip 到 ±100) */
void rl_policy_step(const float obs[25], float actions[6]);

/*
 * 动作解码 -> 关节力矩 (与训练完全一致):
 *   腿(0,1,3,4): 位置PD  tau = P*(act*0.5 + default - q) + D*(0 - qd)
 *   轮(2,5):     速度环  tau = D*(act*10.0 - qd)
 *   输出 tau[6] 为关节力矩 Nm, 已 clip 到 ±30 Nm
 * 注意: 关节力矩 -> 电机电流 的换算 (Kt * 减速比) 需要按你的机械实现.
 */
void rl_policy_apply_actions(const float actions[6], const float dof_pos[6],
                             const float dof_vel[6], float tau[6]);

/* 工具: 机体系四元数旋转逆变换 (投影重力用), 输入输出均为 xyzw 序 */
void rl_quat_rotate_inverse(const float q[4], const float v[3], float out[3]);

#ifdef __cplusplus
}
#endif

#endif /* RL_POLICY_H */
