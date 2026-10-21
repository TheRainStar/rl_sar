# Q25 射击稳定性仿真场景

用于将 [robot_lab](https://github.com/fan-ziqi/robot_lab) 中 **Shooting Stability** 策略迁移到 `rl_sar`，在 **MuJoCo** / **Gazebo** 中做 sim2sim 验证。当前仅适配 **Q25**（`q20b-yuan-v2_load`，带 `Load_sh_Link` 前载荷）。

训练侧任务路径（参考）：

```text
robot_lab/.../locomotion/shooting_stability/
```

---

## 场景说明

靶场几何与 `robot_lab` 的 `TargetRangeSceneQ25Cfg` / `constants.py` 对齐：

| 项目 | 数值 |
|------|------|
| 靶板尺寸 | 厚度 20 mm × 宽 0.5 m × 高 1.0 m |
| 靶心位置 | 正前方约 100 m，高度 1.4 m（红靶，与训练 `TARGET_HEIGHT_M` 一致） |
| 偏置靶 | **蓝靶**，固定在方位角 **+2.5°**（约 y=+4.36 m），对应训练 `TARGET_AZIMUTH_RANGE` 边界「不正对」工况 |
| 命中半径（横向） | 0.25 m |
| 方位角包络（可视化） | 约 ±2.5°（±4.36 m） |
| 后坐力作用点 | `Load_sh_Link` |
| 默认后坐力 | -36 N，周期 0.08 s，脉宽 0.01 s |

场景中包含：射击中线 / 边轨、25/50/75/100 m 地面标线、正对红靶 + 偏置蓝靶、弹道落点标记。

正前方红靶与侧向蓝靶是**两套状态机、两套策略**：

| 按键 | 状态 | 瞄准 | 策略文件（`config.yaml`） |
|------|------|------|---------------------------|
| `[1]` | `RLFSMStateShootFrontal` | 红靶，az=0 | `shooting_frontal_model`（默认 `policy_stab0614.pt`，hold） |
| `[3]` | `RLFSMStateShootOffset` | 蓝靶，+2.5° | `shooting_offset_model`（默认 `policy_twist.pt`，twist） |

观测里的 `target_azimuth`、命中判定与权重在切换时一起换。站稳后直接按对应键即可，不必先切靶再加载同一份 actor。

---

## 相关文件

```text
policy/q25/
  base.yaml                         # shooting_stability: true，开环站起姿态等
  robot_lab/
    config.yaml                     # 观测、PD、action_scale、靶场/后坐力、正/侧向策略文件名
    policy_stab0614.pt              # 正前方 hold / shooting-stability
    policy_twist.pt                 # 侧向 twist / 不正对
    policy.pt                       # 缺省回退权重

src/rl_sar_zoo/q25_description/mjcf/
  shooting_stability.xml            # MuJoCo 靶场场景
  q20b-yuan-v2_load.xml             # 带载荷 mesh 机器人
  assets/*.obj

src/rl_sar/worlds/
  shooting_stability.world          # Gazebo 靶场

src/rl_sar/launch/
  q25_shooting_gazebo.launch        # Gazebo 一键启动（蹲姿 spawn + 暂停）

src/rl_sar/
  src/rl_sim_mujoco.cpp             # MuJoCo 靶场初始化、后坐力、枪口观测
  src/rl_sim.cpp                    # Gazebo 射击模式
  fsm_robot/fsm_q25.hpp             # 0 站起 / 1 正前方策略 / 3 侧向策略 / 2 后坐 / R 重置
```

---

## 编译

```bash
cd ~/workspace/rl_sar

# MuJoCo（推荐先做 sim2sim）
./build.sh -mj

# Gazebo（ROS Noetic）
./build.sh
source devel/setup.bash
```

---

## MuJoCo 运行（推荐）

单进程同时跑仿真与控制，**不要**再开 `rosrun rl_sar rl_sim`。

```bash
cd ~/workspace/rl_sar
# 默认：sim2sim，策略闭环（关伺服 / 钉地扭转 / 锚点混合）
./cmake_build/bin/rl_sim_mujoco q25 shooting_stability
# 或显式：
./cmake_build/bin/rl_sim_mujoco q25 shooting_stability sim2sim
# 旧部署演示（锚点 + 瞄准伺服 + 钉地扭转，不代表策略能用）：
./cmake_build/bin/rl_sim_mujoco q25 shooting_stability deploy
```

### 键盘（必须在启动终端按）

MuJoCo 窗口**不接收** `0/1/2/3/R`。请保持**启动该命令的终端**获得焦点，在终端里按键：

| 按键 | 作用 |
|------|------|
| `0` | 解除暂停 / 开环站起（不开瞄准、不开后坐力） |
| `1` | 进入正前方状态：红靶 + hold 策略 |
| `3` | 进入侧向状态：蓝靶 + twist 策略 |
| `2` | 在策略运行中开启周期后坐力 |
| `R` | 重置场景与 FSM（回到红靶 / Passive） |

策略运行中再按 `1` / `3` 会切换到另一套状态机（换靶 + 换权重，并按 Isaac 复位姿态重新交接）。

**sim2sim 判据（按 `1` 或 `3` 之后看终端 `verdict`）**：`stand=YES`（`z>0.5`、`|pitch|<20°`、后腿 calf`<1.65`），且 `lat<0.25 m`、`vert<0.5 m`、`hit=YES`。开后坐后再看这两项是否明显变差。  
这才是策略迁移结果；`deploy` 模式下命中只能说明部署补丁能打中。

按 `1`/`3` 后立刻前倾跪，优先怀疑**交接**而不是「策略学会了跪」：开环站立 PD 约 280，Isaac 训练是 80/2；若一步降到 80，前载荷会在 0.5 s 内把大腿压过目标角。导出的 twist actor 在 Isaac 复位观测上也会给出 |a|≫1 的原始输出（训练时 `clip_actions=1.0` 裁掉了）。当前 sim2sim 已做：`last_action` 按 Isaac reset 置 0、动作裁 ±1、PD 从 `stand_hold` 缓降到 80/2。可用 `python3 scripts/shooting_probe_policy.py` 对照同一份 `policy.pt` 在 Isaac 复位观测与 MuJoCo seed 观测上的首步动作。

推荐流程：`0` 站稳数秒 → `1` 正前方 hold 或 `3` 侧向 twist → 看 `verdict` → `2` 开后坐力。
两套策略可在站稳后或闭环中互相切换；每次切换都会重新交接。

若终端不是 TTY（例如部分 IDE 内嵌终端），会出现按键无响应。请改用系统终端（如 `gnome-terminal` / `xterm`）启动上述命令。

启动成功后应看到类似：

```text
SIM2SIM policy-in-the-loop: mix=1 servo=OFF twist=OFF stabilize=OFF pd=80/2 clip_actions=±1.
Shooting scene ready (SIM2SIM policy test). ...
```

---

## Gazebo 运行

需要**两个终端**，且先起仿真再起控制：

```bash
# 终端 1：仿真（蹲姿 spawn、初始暂停）
cd ~/workspace/rl_sar
source devel/setup.bash
roslaunch rl_sar q25_shooting_gazebo.launch

# 终端 2：控制（等 Gazebo / roscore 就绪后再开）
source devel/setup.bash
rosrun rl_sar rl_sim
```

控制按键同样在**终端 2**输入 `0/1/3/2/R`，含义与 MuJoCo 一致。

常见错误：

- `Connection refused`：未先 `roslaunch`，或 roscore 已退出。
- `policy//base.yaml` / `No FSM registered for robot:`：机器人名未传入（空 `rname`）。射击场景请用上面的 `q25_shooting_gazebo.launch`，不要只跑裸的 `rl_sim`。

---

## 策略与配置要点

`policy/q25/base.yaml`：

- `shooting_stability: true` 开启射击 FSM（禁用走跑命令，使用 `0/1/3/2/R`）。
- `shooting_stand_dof_pos`：按 `0` 后的开环站立姿态（前后腿不对称，用于抵消前载荷低头）。
- `default_dof_pos`：策略默认关节参考，与开环站姿分开配置。

`policy/q25/robot_lab/config.yaml`：

- 观测维数 **52**：`lin_vel, ang_vel, gravity_vec, commands, dof_pos, dof_vel, actions, muzzle_aim_error, shooting_countdown, target_azimuth`
- 靶场：`shooting_target_distance / height / hit_radius / target_azimuth`
- 正前方 / 侧向权重：`shooting_frontal_model`、`shooting_offset_model`
- 后坐力：`shooting_recoil_interval / duration / force`
- PD：`rl_kp / rl_kd`（部署默认 80 / 2）
- `action_scale`：与训练一致，**hip 0.125，thigh/calf 0.25**
- 观测缩放：`ang_vel_scale=0.25`，`dof_vel_scale=0.05`（与 `apply_q25_shooting_obs_scales` 对齐）

更换权重时，将 hold / twist 导出分别放到 `policy_stab0614.pt` / `policy_twist.pt`（或改 `config.yaml` 里的文件名），并核对观测与训练导出配置一致。`policy.pt` 仅在指定文件缺失时作为回退。

---

## 与训练环境的对齐检查

部署前建议逐项核对：

1. **动作缩放** `action_scale` 与 Isaac Lab `Q25ActionsCfg` 一致。
2. **关节顺序 / mapping** 与 `joint_names` 一致（4 髋、4 大腿、4 小腿）。Isaac 运行时 `robot.joint_names` 也是这个顺序。
3. **瞄准刚体**：Q25 用 `Load_sh_Link`（不是 `GUN_MUZZLE`）。
4. **靶距 / 靶高 / 命中半径 / 后坐力幅值与周期** 与训练 play 配置一致。
5. 先在 **Isaac play** 确认策略本身稳定，再判断 MuJoCo/Gazebo 是否为 sim2sim 差异。

若 MuJoCo 中出现明显俯仰抖动，优先检查：`action_scale` 是否偏大、瞄准观测是否对齐、是否过早按 `1`（开环未站稳）。

---

## 遥测输出

射击模式下默认每 **0.5 s** 在启动终端打印一次，并写入 CSV：

| 内容 | 说明 |
|------|------|
| 姿态 | roll / pitch / yaw（度），以及机身角速度 |
| 基座位置 | MuJoCo `frame_pos`（世界系 xyz） |
| 靶心 / 射线落点 | 枪口射线与靶平面交点，以及 `impact - target` |
| 脱靶量 | 横向 / 纵向 miss、归一化 aim_error、是否命中 |
| 关节力矩 | 各关节 `q`、传感器 `tau_est`、PD 指令 `tau_cmd` |

配置（`policy/q25/base.yaml`）：

```yaml
shooting_telemetry: true
shooting_telemetry_interval: 2.0   # 控制台摘要间隔；CSV 同期写入
shooting_telemetry_csv: true
```

CSV 路径：`policy/q25/shooting_telemetry.csv`。设 `shooting_telemetry: false` 可关闭。

---

## 与 robot_lab 联合调试

`rl_sar` 与 `~/robot_lab` 可按同一 CSV schema 做 sim2sim 对照（不能单步联仿，但是可对齐配置、观测与曲线）。

### 1) 配置同步 / 检查

```bash
cd ~/workspace/rl_sar
# 只检查 action_scale / obs scale / PD / 后坐力 / 力矩上限等
python3 scripts/shooting_sync_config.py --check
# 按 robot_lab 训练侧写回 config.yaml
python3 scripts/shooting_sync_config.py --apply
# 可选：拷贝 Isaac play 导出的权重
python3 scripts/shooting_sync_config.py --copy-policy /path/to/exported/policy.pt
```

### 2) Isaac play 写遥测 CSV

在 `robot_lab` 环境中：

```bash
cd ~/robot_lab
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-ShootingStability-Deeprobotics-Q25-v0 \
  --num_envs 1 --play_straight_ahead --play_no_co_rotate \
  --play_telemetry_csv /tmp/isaac_shooting_telemetry.csv \
  --play_telemetry_interval 0.5 \
  --play_telemetry_max_time 20
```

play 结束时会在 checkpoint 旁导出 `exported/policy.pt`，可用上面的 `--copy-policy` 同步到 `rl_sar`。

### 3) MuJoCo 写同格式 CSV

```bash
cd ~/workspace/rl_sar
./cmake_build/bin/rl_sim_mujoco q25 shooting_stability
# 终端按 0 → 1（正前方）或 3（侧向），可选 2，生成 policy/q25/shooting_telemetry.csv
python3 scripts/shooting_sim2sim_verdict.py policy/q25/shooting_telemetry.csv
```

### 4) 对比两边 CSV

```bash
python3 scripts/shooting_compare_telemetry.py \
  --isaac /tmp/isaac_shooting_telemetry.csv \
  --mujoco policy/q25/shooting_telemetry.csv \
  --out /tmp/shooting_compare
```

输出：

- `/tmp/shooting_compare/compare_report.txt`：pitch / 落点 / 力矩等误差统计
- `compare_attitude.png`、`compare_aim.png`、`compare_joints.png` 等曲线图

一键提示脚本：

```bash
./scripts/shooting_joint_debug.sh
APPLY_CONFIG=1 ./scripts/shooting_joint_debug.sh
COMPARE=1 ./scripts/shooting_joint_debug.sh
```

### 两种运行模式

| 模式 | 启动 | 按 `1` / `3` 之后实际在控什么 |
|------|------|-------------------------------|
| **SIM2SIM（默认）** | `rl_sim_mujoco q25 shooting_stability` | **对应策略闭环**：`mix=1`，关瞄准伺服 / 钉地扭转 / 锚点混合，PD 用 Isaac 的 80/2，动作裁剪 ±1 |
| **DEPLOY** | `... shooting_stability deploy` | 开环锚点 + 射线 PI 伺服 + 偏置靶钉地扭转。能打中 **不代表** 策略能用 |

sim2sim 在启动和按 `1`/`3` 时会打印 `SIM2SIM policy-in-the-loop`。遥测每行带 `mode=SIM2SIM`、`hold`/`twist` 阶段和 `verdict`。CSV 多了 `ctrl_mode`（1=sim2sim）和 `stand_ok`。

跑完可用：

```bash
python3 scripts/shooting_sim2sim_verdict.py policy/q25/shooting_telemetry.csv
```

### 按 1 / 3 后摔倒 / 后腿跪地

**先按 `0` 站稳，再按 `1`（正前方）或 `3`（侧向）。** 从 Passive / 蹲姿直接交接，策略会在软腿、瞄准误差饱和（`aim_error=1`）的状态下接手，几乎必然前倾跪下。这不是策略本身的第一结论。

sim2sim 按 `1`/`3` 时会：

1. 把关节拉到训练复位：`default_dof_pos`（髋 0、大腿 −0.67、小腿 1.3），机身偏航 0，足端贴地。侧向再按 `3.2 × 方位角` 做左右髋差动预扭。MuJoCo 枪口在这组关节上会偏低约 0.4°，所以只绕机身 Y 轴补一个俯仰，让射线打到 1.4 m 靶心；关节角和偏航不变
2. **立刻闭环**，PD 就是 Isaac 的 80/2，`last_action=0`，动作裁剪 ±1，不再做 1.5 s 开环缓降

`aim_obs` 与 `mdp._aim_lateral_miss_m` 相同：`距离 × sin(枪口射线与靶心方向的夹角) / 0.5`，封顶 1。俯仰和偏航都算进去。后坐力在 `Load_sh_Link` 坐标系：策略开始 80 ms 后保持恒定，正前方 22 N、侧向 40 N（课程末端），另有 0.167 倍向上分量。

在 `settling` / `policy OFF` 期间倒了是开环站姿问题；`settled` 之后 `roll` 到 ±90° 才是策略迁移问题。

对照实验（无界面 rollout）：

| 交接 | twist `policy.pt` |
|------|-------------------|
| 标定站姿 + `last_action=0` + clip±1 | 能站住 |
| Isaac 默认关节 + z=0.45（纯 play 复位） | 约 3 s 倒下 |
| Passive/软腿直接交策略 | 立刻跪 |

所以：Isaac play 成功 ≠ 从任意姿态按 `1`/`3` 都能站住。要验证策略，必须从站稳后的闭环看 `stand_ok` / 命中率。若站稳交接后仍跪，才说明策略没迁过来，需要 Isaac 侧 finetune，而不是再打开 DEPLOY 补丁当“验证通过”。

DEPLOY 模式仍保留这些补丁（仅演示）：

| 配置 | 作用 |
|------|------|
| `shooting_aim_anchor_dof_pos` | MuJoCo 标定的稳定瞄准开环姿 |
| `shooting_deploy_stabilize` | 策略输出与锚点混合 |
| `shooting_aim_servo` | 按 `1` 后按射线误差 PI 对准 |
| `shooting_planted_twist` | 偏置靶：副本上求钉地扭转关节，PD 扭身 |
| `shooting_leg_*_clamp` | 限制大腿/小腿，防止后跪 |

其它对齐项：

1. **冷启动**：`shooting_policy_blend_duration` + seed `last_action`
2. **线速度**：MuJoCo `framelinvel` 已转到机体系
3. **策略对照**：FSM 直接加载 `policy_stab0614.pt` / `policy_twist.pt`；`./scripts/shooting_switch_policy.sh` 只改回退用的 `policy.pt`
4. **非零方位角**：按 `3` 进入侧向状态（蓝靶 + twist 策略）。sim2sim 下由 twist 策略自己扭身找靶；deploy 下用钉地扭转。

---

## 已知限制

- 目前射击稳定性场景仅适配 **Q25**。
- MuJoCo 与 Isaac PhysX 有动力学差异。Isaac play 成功只说明训练域能用；MuJoCo 默认 sim2sim 才是迁移测试。
- 部分环境下 Ctrl+C 退出 MuJoCo 可能出现段错误，不影响日常验证；可直接关闭窗口或再次启动。
- Gazebo 开环站起若仍跪趴，可先按 `R` 重置，再按 `0` 站起。可用 `SHOOTING_MODE=sim2sim|deploy` 切换。
