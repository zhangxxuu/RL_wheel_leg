from wheel_legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class CodBalanceCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.28]         
        default_joint_angles = {        
            "Left_rear_joint": 0.0, "Left_rear_child1_joint": 0.0, "Left_Wheel_joint": 0.0,
            "Right_rear_joint": 0.0, "Right_rear_child1_joint": 0.0, "Right_Wheel_joint": 0.0,
        }
    class control(LeggedRobotCfg.control):
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        stiffness = {"Left_rear_joint": 40.0, "Right_rear_joint": 40.0,
                     "Left_rear_child1_joint": 40.0, "Right_rear_child1_joint": 40.0,
                     "Left_Wheel_joint": 0.0, "Right_Wheel_joint": 0.0}   # 轮 Kp=0
        damping = {"Left_rear_joint": 2.0, "Right_rear_joint": 2.0,
                   "Left_rear_child1_joint": 2.0, "Right_rear_child1_joint": 2.0,
                   "Left_Wheel_joint": 0.2, "Right_Wheel_joint": 0.2}
    class commands(LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]    
            height = [0.30, 0.33]       
    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cod_V1/urdf/COD_2026_Balance_3_0.urdf"  # 19kg 版
        name = "CodBalance"
        l1 = 0.215                     # 髋→膝 0.215 m
        l2 = 0.254                     # 膝→轮心 0.254 m
        self_collisions = 1            # 关自碰撞
        flip_visual_attachments = False

    # ── 奖励：先用复旦那 3 个私有轮子奖励（函数在 base/legged_robot.py 里）──
    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            wheel_vel_abs_match    = 0.5   # 左右轮转速绝对值匹配
            wheel_torque_abs_match = 0.5   # 左右轮力矩绝对值匹配
            wheel_torque_smooth    = 0.5   # ★ 轮子力矩平滑（治抖）


class CodBalanceCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "cod_balance"
        max_iterations = 50000