/*
 * test_main.c —— 对拍测试: C 推理 vs torch 期望值
 * 编译: gcc -O2 -I.. rl_policy.c test_main.c -lm -o test_main
 * 运行: ./test_main scenario.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "rl_policy.h"
#include "policy_weights.h"

static float rd(FILE *f) { float v; if (fread(&v, 4, 1, f) != 1) exit(1); return v; }

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "scenario.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); return 1; }

    int steps;
    if (fread(&steps, 4, 1, f) != 1) { perror("read steps"); return 1; }

    float gyro[3], quat[4], dof_pos[6], dof_vel[6], cmd[3];
    rl_policy_init();
    float max_err = 0.0f;

    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < 3; i++) gyro[i] = rd(f);
        for (int i = 0; i < 4; i++) quat[i] = rd(f);
        for (int i = 0; i < 6; i++) dof_pos[i] = rd(f);
        for (int i = 0; i < 6; i++) dof_vel[i] = rd(f);
        for (int i = 0; i < 3; i++) cmd[i] = rd(f);

        float obs_raw[25], obs[25];
        rl_policy_build_obs(gyro, quat, dof_pos, dof_vel, cmd, obs_raw, obs);
        float actions[6];
        rl_policy_step(obs, actions);

        float exp_a[6];
        for (int i = 0; i < 6; i++) exp_a[i] = rd(f);

        float err = 0.0f;
        for (int i = 0; i < 6; i++) {
            float d = fabsf(actions[i] - exp_a[i]);
            if (d > err) err = d;
        }
        if (err > max_err) max_err = err;
        printf("step %d  C: [", s);
        for (int i = 0; i < 6; i++) printf("%.5f%s", actions[i], i < 5 ? ", " : "");
        printf("]\n        exp: [");
        for (int i = 0; i < 6; i++) printf("%.5f%s", exp_a[i], i < 5 ? ", " : "");
        printf("]  max_err=%.3e %s\n", err, err < 1e-3 ? "PASS" : "FAIL");
    }
    printf("=== 总体最大误差 %.3e ===\n", max_err);
    fclose(f);
    return max_err < 1e-3 ? 0 : 1;
}
