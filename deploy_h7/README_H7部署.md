# H7 下位机 RL 部署工具包（fudan_rl_wheel_leg）

把训练好的轮腿 RL 策略（encoder+actor 小网络）直接跑在 STM32H7 主控上，**小电脑（自瞄）只负责发指令**，与你的 LQR 传统控制架构兼容。

部署有两条路线（复旦星云用的是路线 B）：
- **路线 A（本包默认）**：手写 C 矩阵乘推理（已对拍验证，误差 7e-8），不依赖 ST 工具链
- **路线 B（推荐）**：**STM32Cube.AI** 导入 ONNX 自动生成网络 C 代码，H7 直接 CAN 连电调

## 文件

| 文件 | 作用 |
|---|---|
| `export_weights_c.py` | 训练 checkpoint (.pt) → `policy_weights.h`（权重 + 全部部署常量） |
| `rl_policy.h` / `rl_policy.c` | 策略推理 C 实现（obs 组装 / 历史缓冲 / 网络前向 / 动作解码） |
| `test/gen_scenario.py` | 生成随机场景 + torch 期望值（对拍基准） |
| `test/test_main.c` | C 侧对拍程序 |

## 网络结构与性能（已验证）

- encoder: 125 → 128(ELU) → 64(ELU) → 3 (latent)
- actor  : 28  → 128(ELU) → 64(ELU) → 32(ELU) → 6
- 参数: 38400 (~150KB float32)；前向 ~38.5k MACs → **STM32H7 480MHz 每次 <0.15ms**，100Hz 控制下 CPU 占用 ~1.5%
- ✅ 已用随机权重对拍验证：C 与 PyTorch 输出最大误差 **7.45e-08**

## 使用流程（路线 A：手写 C 推理）

### 1. 导出权重（训练出模型后，在 leg 环境执行）

```bash
export PYTHONPATH=~/isaacgym/fudan_rl_wheel_leg/plane:$PYTHONPATH
cd ~/isaacgym/fudan_rl_wheel_leg/deploy_h7
python export_weights_c.py --pt <训练输出的 model_XXXX.pt> --out policy_weights.h
```

### 2. 移植到 H7 工程

1. 把 `policy_weights.h`、`rl_policy.h`、`rl_policy.c` 拷入你的 STM32 工程
2. 在 100Hz 定时任务里：
   ```c
   rl_policy_init();                       // 上电初始化一次
   // 每个控制周期:
   rl_policy_build_obs(gyro, quat, dof_pos, dof_vel, cmd, obs_raw, obs);
   rl_policy_step(obs, actions);           // 推理
   rl_policy_apply_actions(actions, dof_pos, dof_vel, tau);  // 关节力矩
   // tau[Nm] -> 电机电流: I = tau / (Kt * 减速比)，走现有电流环/CAN
   ```

### 3. 对拍验证（强烈建议，换权重后必做）

```bash
cd test
python gen_scenario.py          # 生成 fake 权重 + 场景 + torch 期望
gcc -O2 -I.. -I. ../rl_policy.c test_main.c -lm -o test_main
./test_main scenario.bin        # 全部 PASS 且误差 <1e-3 才合格
```
（用你真实权重替换后，对拍脚本会把你的权重导出成头文件再对拍。）

## obs 布局（25 维，顺序不可错）

| 位置 | 内容 | 缩放 |
|---|---|---|
| 0:3 | 机体角速度 (gyro, rad/s) | ×0.25 |
| 3:6 | 投影重力（重力[0,0,-1] 旋转到机体系） | 原值 |
| 6:9 | 指令 [线速度, 偏航角速度, 高度] | ×[3.0, 0.25, 5.0] |
| 9:13 | 腿关节角 lf0,lf1,rf0,rf1（-default） | ×1.0 |
| 13:19 | 6 关节速度（同 DOF 序） | ×0.05 |
| 19:25 | 上次动作（6 维） | 原值 |

DOF 序: `[lf0, lf1, l_wheel, rf0, rf1, r_wheel]`，default = `[-0.23, -0.65, 0, 0.23, 0.65, 0]`

历史缓冲: 5 帧 × 25 维，存**未缩放 obs_raw**，滚动窗口（旧→新），零初始化（与训练一致）。

## 动作解码（与训练完全一致）

- 腿 (idx 0,1,3,4): 位置PD — `tau = P*(act*0.5 + default - q) + D*(0 - qd)`，P=15, D=1
- 轮 (idx 2,5): 速度环 — `tau = D*(act*10.0 - qd)`，D=0.1
- 输出 clip ±30 Nm

## 使用流程（路线 B：STM32Cube.AI，推荐/复旦路线）

### 1. 导出 ONNX（训练后执行）

```bash
cd ~/fudan_rl_wheel_leg/plane
python export_onnx/export_onnx.py --load_run=<run名> --checkpoint=<步数> --out policy.onnx
# 或 history=10 版本用 export_onnx_his10.py
```
ONNX 有两个输入: `obs[1,25]` 和 `obs_history[1,125]`，输出 `actions[1,6]` —— Cube.AI 支持多输入。

### 2. STM32CubeMX + X-CUBE-AI

1. CubeMX 选你的 H7 型号 → 启用 FDCAN/CAN 外设（CAN 直连电调）→ 安装并启用 **X-CUBE-AI** 中间件
2. AI 页签 → 导入 `policy.onnx` → **Analyze**（看 RAM/Flash/周期数，H7 浮点推理无压力）→ **Validate**（可用对拍数据验证 float32 精度）→ **Generate Code**
3. Cube.AI 会生成 `network.c` / `network_data.c`（权重）和 `ai_network_run()` 等 API

### 3. 主控代码（100Hz 定时任务）

```c
// 1. 组装 obs（25 维）——照本包 rl_policy_build_obs 的逻辑写（含历史缓冲 5 帧）
//    注意 obs 布局/缩放/四元数全部要与训练一致，见下方表格
// 2. 喂输入 + 推理
ai_network_inputs_set(network, inputs);   // inputs[0]=obs[25], inputs[1]=obs_history[125]
ai_network_run(network);
ai_network_outputs_get(network, outputs); // outputs[0]=actions[6]
// 3. 动作解码为关节力矩（照 rl_policy_apply_actions 的逻辑写）
// 4. CAN 直连电调: 发送电流指令帧 (0x200/0x1FF...), 接收电机反馈帧 (0x201/0x202...)
```

**obs 组装、历史缓冲、动作解码、CAN 通信仍需自己写**——直接用本包的 `rl_policy.h/.c` 里的
`rl_policy_build_obs` / `rl_policy_apply_actions` 逻辑（把中间的网络推理换成 `ai_network_run` 即可）。

### 4. 对拍验证（换权重后必做）

Cube.AI 的 Validate 需要输入/输出数据，用本包 test 生成：
```bash
cd test && python gen_scenario.py   # 生成 scenario 与期望输出
```
把输入喂给 Cube.AI 生成的模型，输出与 `scenario.bin` 中的期望对比（误差 <1e-3 合格）。

## 集成与安全（重点）

1. **模式切换**：保留 LQR 代码，加一个模式变量（遥控器/串口切换 `MODE_LQR <-> MODE_RL`）。切换瞬间需要平滑过渡（力矩斜坡限幅），防止策略首次输出跳变。
2. **关节角符号/方向**：电机反馈角度符号必须与训练 URDF 一致，先用 mujoco sim2sim 的数据做开环对比（给同样 obs，比对动作输出）。
3. **IMU 安装方向**：gyro 与四元数必须是机体系，若 IMU 安装方向不同要加旋转矩阵校正。
4. **力矩→电流**：按你的减速比和电机 Kt 换算（LQR 时代应该已有），注意齿轮箱效率。
5. **实车步骤**：悬挂（轮离地）→ 小范围 → 遥控器急停全程在手。
6. 若 H750（128KB Flash）放不下 150KB float32 权重：选项 A 权重放外部 QSPI Flash（A板自带 8MB）启动时拷到 AXI SRAM；选项 B 用 float16（~75KB）；选项 C int8 量化（~38KB，需要重标定）。

## 频率与指令接口

- 控制频率 **100Hz**（sim dt=0.005, decimation=2）
- 小电脑（自瞄）串口协议扩展：把现有的目标指令加上 `线速度 m/s（±2.8）、偏航角速度 rad/s（±6.0）、车身高度 m（0.14~0.25）` 即可，策略天然吃这三个指令。
