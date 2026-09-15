from wheel_legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class CodBalanceCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.28]          # 出生高度 = 轮心0.259 + 余量；别抄复旦的 0.1（会插地）
        default_joint_angles = {        # 键名 = URDF 关节名，缺一个就 KeyError
            "Left_rear_joint": 0.0, "Left_rear_child1_joint": 0.0, "Left_Wheel_joint": 0.0,
            "Right_rear_joint": 0.0, "Right_rear_child1_joint": 0.0, "Right_Wheel_joint": 0.0,
        }
    class control(LeggedRobotCfg.control):
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        stiffness = {"Left_rear_joint": 20.0, "Right_rear_joint": 20.0,
                     "Left_rear_child1_joint": 20.0, "Right_rear_child1_joint": 20.0,
                     "Left_Wheel_joint": 0.0, "Right_Wheel_joint": 0.0}   # 轮 Kp=0
        damping = {"Left_rear_joint": 1.0, "Right_rear_joint": 1.0,
                   "Left_rear_child1_joint": 1.0, "Right_rear_child1_joint": 1.0,
                   "Left_Wheel_joint": 0.2, "Right_Wheel_joint": 0.2}
    class commands(LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]     # 第 5 步会讲为什么从 ±2 改成 ±0.5
            height = [0.30, 0.33]       # 第 5 步会讲为什么从 [0.20,0.30] 改成这个
    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cod_V1/urdf/COD_2026_Balance_2_0.urdf"  # 19kg 版
        name = "CodBalance"
        l1 = 0.215                     # 髋→膝 0.215 m
        l2 = 0.254                     # 膝→轮心 0.254 m
        self_collisions = 1            # 关自碰撞
        flip_visual_attachments = False

class CodBalanceCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "cod_balance"
        max_iterations = 50000