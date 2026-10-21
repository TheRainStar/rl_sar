/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim_mujoco.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

RL_Sim* RL_Sim::instance = nullptr;

RL_Sim::RL_Sim(int argc, char **argv)
{
    // Set static instance pointer early for signal handler
    instance = this;

    if (argc < 3)
    {
        std::cout << LOGGER::ERROR
                  << "Usage: " << argv[0] << " robot_name scene_name [sim2sim|deploy]" << std::endl
                  << "  sim2sim (default for shooting_stability): policy-in-the-loop, no deploy cheats"
                  << std::endl
                  << "  deploy: anchor + aim servo + planted twist (demo stack)"
                  << std::endl;
        throw std::runtime_error("Invalid arguments");
    }
    else
    {
        this->robot_name = argv[1];
        this->scene_name = argv[2];
    }

    this->shooting_sim2sim_mode = true;
    if (argc >= 4)
    {
        const std::string mode(argv[3]);
        if (mode == "deploy")
        {
            this->shooting_sim2sim_mode = false;
        }
        else if (mode == "sim2sim")
        {
            this->shooting_sim2sim_mode = true;
        }
        else
        {
            std::cout << LOGGER::ERROR
                      << "Unknown mode '" << mode << "'. Use sim2sim or deploy." << std::endl;
            throw std::runtime_error("Invalid shooting mode");
        }
    }
    if (const char *env_mode = std::getenv("SHOOTING_MODE"))
    {
        const std::string mode(env_mode);
        if (mode == "deploy")
        {
            this->shooting_sim2sim_mode = false;
        }
        else if (mode == "sim2sim")
        {
            this->shooting_sim2sim_mode = true;
        }
    }

    this->ang_vel_axis = "body";

    // now launch mujoco
    std::cout << LOGGER::INFO << "[MuJoCo] Launching..." << std::endl;

    // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
    if (rosetta_error_msg)
    {
        DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
        std::exit(1);
    }
#endif

    // print version, check compatibility
    std::cout << LOGGER::INFO << "[MuJoCo] Version: " << mj_versionString() << std::endl;
    if (mjVERSION_HEADER != mj_version())
    {
        mju_error("Headers and library have different versions");
    }

    // scan for libraries in the plugin directory to load additional plugins
    scanPluginLibraries();

    mjvCamera cam;
    mjv_defaultCamera(&cam);

    mjvOption opt;
    mjv_defaultOption(&opt);

    mjvPerturb pert;
    mjv_defaultPerturb(&pert);

    // simulate object encapsulates the UI
    sim = std::make_unique<mj::Simulate>(
        std::make_unique<mj::GlfwAdapter>(),
        &cam, &opt, &pert, /* is_passive = */ false);

    std::string filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/../rl_sar_zoo/" + this->robot_name + "_description/mjcf/" + this->scene_name + ".xml";

    // start physics thread
    std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename.c_str());
    physicsthreadhandle.detach();

    while (1)
    {
        if (d)
        {
            std::cout << LOGGER::INFO << "[MuJoCo] Data prepared" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    this->mj_model = m;
    this->mj_data = d;
    this->SetupSysJoystick("/dev/input/js0", 16); // 16 bits joystick

    // read params from yaml
    this->ReadYaml(this->robot_name, "base.yaml");
    if (this->scene_name == "shooting_stability" || this->params.Get<bool>("shooting_stability", false))
    {
        this->ConfigureShootingStabilityMode(true);
        if (!isatty(STDIN_FILENO))
        {
            std::cout << LOGGER::WARNING
                      << "stdin is not a TTY. Use gnome-terminal/xterm and press 0/1/2/R there."
                      << std::endl;
        }
        else
        {
            std::cout << LOGGER::NOTE
                      << "Press 0/1/2/R in this terminal (keep terminal focused, not the MuJoCo window)."
                      << std::endl;
        }
    }
    if (this->scene_name == "shooting_stability")
    {
        this->ReadYaml(this->robot_name + "/robot_lab", "config.yaml");
        this->shooting_recoil_body_id = mj_name2id(this->mj_model, mjOBJ_BODY, "Load_sh_Link");
        if (this->shooting_recoil_body_id < 0)
        {
            this->shooting_recoil_body_id = mj_name2id(this->mj_model, mjOBJ_BODY, "base_link");
        }
        this->shooting_target_mocap_id = mj_name2id(this->mj_model, mjOBJ_BODY, "shooting_target");
        this->shooting_target_offset_mocap_id = mj_name2id(this->mj_model, mjOBJ_BODY, "shooting_target_offset");
        this->ballistic_impact_mocap_id = mj_name2id(this->mj_model, mjOBJ_BODY, "ballistic_impact");
        // Frontal red board is always at az=0; blue board at configured offset.
        this->shooting_frontal_azimuth = 0.0f;
        this->shooting_offset_azimuth = this->params.Get<float>("shooting_offset_target_azimuth", 0.0436332313f);
        const float active_az = this->params.Get<float>("shooting_target_azimuth", 0.0f);
        this->shooting_active_target_is_offset =
            std::fabs(active_az - this->shooting_offset_azimuth) < 1e-4f &&
            std::fabs(this->shooting_offset_azimuth) > 1e-4f;
        this->obs.shooting_target_azimuth =
            this->shooting_active_target_is_offset ? this->shooting_offset_azimuth : this->shooting_frontal_azimuth;
        this->shooting_target_distance = this->params.Get<float>("shooting_target_distance", 100.0f);
        this->shooting_target_height = this->params.Get<float>("shooting_target_height", 1.3f);
        this->shooting_hit_radius = this->params.Get<float>("shooting_hit_radius", 0.25f);
        this->shooting_recoil_interval = this->params.Get<float>("shooting_recoil_interval", 0.08f);
        {
            const std::lock_guard<std::recursive_mutex> lock(this->sim->mtx);
            this->CacheShootingKinematics();
            if (this->shooting_ik_scratch == nullptr)
            {
                this->shooting_ik_scratch = mj_makeData(this->mj_model);
            }
            this->SetupShootingRange();
        }
        this->shooting_scene_ready = true;
        this->ApplyShootingSim2SimOverrides();
        std::cout << LOGGER::INFO
                  << "Shooting scene ready ("
                  << (this->IsShootingSim2Sim() ? "SIM2SIM policy test" : "DEPLOY demo")
                  << "). Red=frontal hold policy, blue=offset twist policy. "
                  << "Press [0] stand | [1] frontal/hold | [3] offset/twist | [2] recoil | [R] reset."
                  << std::endl;
    }

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init robot
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard: 0/1/2/R in the launch terminal
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

    // joystick
    this->loop_joystick = std::make_shared<LoopFunc>("loop_joystick", 0.01, std::bind(&RL_Sim::GetSysJoystick, this));
    this->loop_joystick->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;

    // start simulation UI loop (blocking call)
    sim->RenderLoop();
}

RL_Sim::~RL_Sim()
{
    // Clear static instance pointer
    instance = nullptr;

    if (this->shooting_ik_scratch != nullptr)
    {
        mj_deleteData(this->shooting_ik_scratch);
        this->shooting_ik_scratch = nullptr;
    }
    if (this->loop_keyboard)
    {
        this->loop_keyboard->shutdown();
    }
    this->loop_joystick->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::GetState(RobotState<float> *state)
{
    if (mj_data)
    {
        state->imu.quaternion[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 0];
        state->imu.quaternion[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 1];
        state->imu.quaternion[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 2];
        state->imu.quaternion[3] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 3];

        state->imu.gyroscope[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 4];
        state->imu.gyroscope[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 5];
        state->imu.gyroscope[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 6];

        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            state->motor_state.q[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]];
            state->motor_state.dq[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")];
            state->motor_state.tau_est[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + 2 * this->params.Get<int>("num_of_dofs")];
        }
        const int frame_vel_sensor_id = mj_name2id(this->mj_model, mjOBJ_SENSOR, "frame_vel");
        if (frame_vel_sensor_id >= 0)
        {
            const int adr = this->mj_model->sensor_adr[frame_vel_sensor_id];
            // MuJoCo framelinvel is world-frame; Isaac base_lin_vel is body-frame.
            const std::vector<float> lin_vel_w = {
                static_cast<float>(this->mj_data->sensordata[adr + 0]),
                static_cast<float>(this->mj_data->sensordata[adr + 1]),
                static_cast<float>(this->mj_data->sensordata[adr + 2]),
            };
            this->obs.lin_vel = QuatRotateInverse(state->imu.quaternion, lin_vel_w);
        }
        const int frame_pos_sensor_id = mj_name2id(this->mj_model, mjOBJ_SENSOR, "frame_pos");
        if (frame_pos_sensor_id >= 0)
        {
            const int adr = this->mj_model->sensor_adr[frame_pos_sensor_id];
            this->obs.base_pos = {
                static_cast<float>(this->mj_data->sensordata[adr + 0]),
                static_cast<float>(this->mj_data->sensordata[adr + 1]),
                static_cast<float>(this->mj_data->sensordata[adr + 2]),
            };
        }
        this->UpdateShootingStateSensors();
    }
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    if (mj_data)
    {
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            mj_data->ctrl[this->params.Get<std::vector<int>>("joint_mapping")[i]] =
                command->motor_command.tau[i] +
                command->motor_command.kp[i] * (command->motor_command.q[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]]) +
                command->motor_command.kd[i] * (command->motor_command.dq[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")]);
        }
    }
}

void RL_Sim::RobotControl()
{
    // Lock the sim mutex once for the entire control cycle to prevent race conditions
    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);

    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    if (this->shooting_stability_mode && this->rl_init_done)
    {
        this->shooting_policy_time_s += this->params.Get<float>("dt", 0.005f);
    }

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        if (this->shooting_stability_mode)
        {
            this->HandleShootingSimulationReset();
            this->OnSimulationPhysicsReset();
            this->shooting_scene_ready = true;
            std::cout << LOGGER::INFO << "Shooting scene reset. Press [0] to stand up." << std::endl;
        }
        else if (this->mj_model && this->mj_data)
        {
            this->OnSimulationPhysicsReset();
        }
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (!this->shooting_stability_mode)
        {
            if (simulation_running)
            {
                sim->run = 0;
                std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
            }
            else
            {
                sim->run = 1;
                std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
            }
            simulation_running = !simulation_running;
        }
    }

    this->control.ClearInput();

    this->ApplyShootingRecoil();
    this->ApplyShootingAimBaseYawCorrection();
    this->SetCommand(&this->robot_command);
    this->MaybeLogShootingTelemetry();
}

void RL_Sim::SetupShootingRange()
{
    this->UpdateShootingTargetPose();
    this->UpdateShootingBallisticRay();
    if (this->mj_model && this->mj_data)
    {
        mj_forward(this->mj_model, this->mj_data);
    }
}

void RL_Sim::PlaceShootingMocapTarget(int body_id, float azimuth)
{
    if (!this->mj_model || !this->mj_data || body_id < 0)
    {
        return;
    }

    const int mocap_idx = this->mj_model->body_mocapid[body_id];
    if (mocap_idx < 0)
    {
        return;
    }

    const float target_x = this->shooting_target_distance * std::cos(azimuth);
    const float target_y = this->shooting_target_distance * std::sin(azimuth);
    const float target_z = this->shooting_target_height;

    mjtNum *mocap_pos = this->mj_data->mocap_pos + 3 * mocap_idx;
    mocap_pos[0] = target_x;
    mocap_pos[1] = target_y;
    mocap_pos[2] = target_z;

    // Face the target board toward the shooter (yaw = azimuth + pi).
    const float yaw = azimuth + static_cast<float>(M_PI);
    mjtNum *mocap_quat = this->mj_data->mocap_quat + 4 * mocap_idx;
    mocap_quat[0] = std::cos(yaw * 0.5f);
    mocap_quat[1] = 0.0;
    mocap_quat[2] = 0.0;
    mocap_quat[3] = std::sin(yaw * 0.5f);
}

void RL_Sim::UpdateShootingTargetPose()
{
    // Keep both boards fixed: frontal (red) at az=0, offset (blue) at non-zero azimuth.
    // Active aim / telemetry uses obs.shooting_target_azimuth (FSM [1]/[3]).
    this->PlaceShootingMocapTarget(this->shooting_target_mocap_id, this->shooting_frontal_azimuth);
    this->PlaceShootingMocapTarget(this->shooting_target_offset_mocap_id, this->shooting_offset_azimuth);
}

void RL_Sim::SetShootingActiveTarget(bool offset)
{
    RL::SetShootingActiveTarget(offset);
    this->UpdateShootingTargetPose();
    this->UpdateShootingBallisticRay();
}

void RL_Sim::UpdateShootingBallisticRay()
{
    if (!this->mj_model || !this->mj_data || this->ballistic_impact_mocap_id < 0)
    {
        return;
    }

    const int impact_mocap_idx = this->mj_model->body_mocapid[this->ballistic_impact_mocap_id];
    if (impact_mocap_idx < 0)
    {
        return;
    }

    const ShootingRayImpact ray = this->ComputeShootingRayImpact();
    if (!ray.valid)
    {
        return;
    }

    mjtNum *impact_pos = this->mj_data->mocap_pos + 3 * impact_mocap_idx;
    impact_pos[0] = ray.impact[0];
    impact_pos[1] = ray.impact[1];
    impact_pos[2] = ray.impact[2];
}

void RL_Sim::UpdateShootingStateSensors()
{
    if (this->scene_name != "shooting_stability" || !this->mj_model || !this->mj_data)
    {
        return;
    }

    const int muzzle_pos_sensor_id = mj_name2id(this->mj_model, mjOBJ_SENSOR, "muzzle_pos");
    const int muzzle_quat_sensor_id = mj_name2id(this->mj_model, mjOBJ_SENSOR, "muzzle_quat");
    if (muzzle_pos_sensor_id >= 0 && muzzle_quat_sensor_id >= 0)
    {
        const int muzzle_pos_offset = this->mj_model->sensor_adr[muzzle_pos_sensor_id];
        const int muzzle_quat_offset = this->mj_model->sensor_adr[muzzle_quat_sensor_id];
        this->obs.muzzle_pos = {
            static_cast<float>(this->mj_data->sensordata[muzzle_pos_offset + 0]),
            static_cast<float>(this->mj_data->sensordata[muzzle_pos_offset + 1]),
            static_cast<float>(this->mj_data->sensordata[muzzle_pos_offset + 2]),
        };
        this->obs.muzzle_quat = {
            static_cast<float>(this->mj_data->sensordata[muzzle_quat_offset + 0]),
            static_cast<float>(this->mj_data->sensordata[muzzle_quat_offset + 1]),
            static_cast<float>(this->mj_data->sensordata[muzzle_quat_offset + 2]),
            static_cast<float>(this->mj_data->sensordata[muzzle_quat_offset + 3]),
        };
    }
    this->UpdateShootingBallisticRay();
}

void RL_Sim::ApplyShootingRecoil()
{
    if (this->scene_name != "shooting_stability" || !this->mj_model || !this->mj_data || this->shooting_recoil_body_id < 0)
    {
        return;
    }

    mjtNum *xfrc = this->mj_data->xfrc_applied + 6 * this->shooting_recoil_body_id;
    for (int i = 0; i < 6; ++i)
    {
        xfrc[i] = 0.0;
    }

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;
    this->ComputeShootingRecoilWrench(fx, fy, fz);
    // Isaac applies the wrench in the Load_sh_Link frame. xfrc_applied is world frame.
    const mjtNum *rot = this->mj_data->xmat + 9 * this->shooting_recoil_body_id;
    xfrc[0] = rot[0] * fx + rot[1] * fy + rot[2] * fz;
    xfrc[1] = rot[3] * fx + rot[4] * fy + rot[5] * fz;
    xfrc[2] = rot[6] * fx + rot[7] * fy + rot[8] * fz;
}

void RL_Sim::PlaceShootingStandFeetOnGround()
{
    if (!this->shooting_kinematics_ready || this->mj_model == nullptr || this->mj_data == nullptr)
    {
        return;
    }
    mj_forward(this->mj_model, this->mj_data);
    // Foot collision is a horizontal cylinder (radius 42 mm). Placing the foot
    // body origin on the plane buries the sole and the contact shove pitches
    // the base over before PD 80 can settle.
    const char *foot_geoms[4] = {
        "FL_foot_collision", "FR_foot_collision", "RL_foot_collision", "RR_foot_collision"};
    mjtNum min_sole_z = 1.0e6;
    int found = 0;
    for (const char *geom_name : foot_geoms)
    {
        const int gid = mj_name2id(this->mj_model, mjOBJ_GEOM, geom_name);
        if (gid < 0)
        {
            continue;
        }
        const mjtNum sole_z = this->mj_data->geom_xpos[3 * gid + 2] - this->mj_model->geom_size[3 * gid];
        min_sole_z = std::min(min_sole_z, sole_z);
        ++found;
    }
    if (found == 0)
    {
        mjtNum feet[4][3];
        if (!this->CaptureShootingFootPositions(this->mj_data, feet))
        {
            return;
        }
        min_sole_z = feet[0][2];
        for (int i = 1; i < 4; ++i)
        {
            min_sole_z = std::min(min_sole_z, feet[i][2]);
        }
    }
    this->mj_data->qpos[2] += (0.001 - min_sole_z);
    mj_forward(this->mj_model, this->mj_data);
}

void RL_Sim::ApplyShootingSim2SimHandoffPose()
{
    if (!this->IsShootingSim2Sim() || this->mj_model == nullptr || this->mj_data == nullptr)
    {
        return;
    }

    // Isaac reset_shooting_ready_state: default joint angles, root yaw 0, z = 0.45.
    // Twist stages also preshape hips: +gain*azimuth on FL/RL, -gain*azimuth on FR/RR.
    std::vector<float> pose = this->params.Get<std::vector<float>>("default_dof_pos");
    const float azimuth = this->obs.shooting_target_azimuth;
    const float hip_gain = this->params.Get<float>("shooting_hip_twist_gain", 3.2f);
    if (this->shooting_active_target_is_offset && std::fabs(azimuth) > 1.0e-6f && pose.size() >= 4)
    {
        const float delta = hip_gain * azimuth;
        const float side_sign[4] = {1.0f, -1.0f, 1.0f, -1.0f};  // FL, FR, RL, RR
        for (int i = 0; i < 4; ++i)
        {
            pose[i] += side_sign[i] * delta;
        }
    }

    const auto names = this->params.Get<std::vector<std::string>>("joint_names");
    const float reset_z = this->params.Get<float>("shooting_reset_height", 0.45f);
    this->mj_data->qpos[0] = 0.0;
    this->mj_data->qpos[1] = 0.0;
    this->mj_data->qpos[2] = reset_z;
    this->mj_data->qpos[3] = 1.0;
    this->mj_data->qpos[4] = 0.0;
    this->mj_data->qpos[5] = 0.0;
    this->mj_data->qpos[6] = 0.0;
    for (size_t i = 0; i < names.size() && i < pose.size(); ++i)
    {
        const int jid = mj_name2id(this->mj_model, mjOBJ_JOINT, names[i].c_str());
        if (jid < 0)
        {
            continue;
        }
        this->mj_data->qpos[this->mj_model->jnt_qposadr[jid]] = pose[i];
        this->mj_data->qvel[this->mj_model->jnt_dofadr[jid]] = 0.0;
    }
    for (int i = 0; i < 6; ++i)
    {
        this->mj_data->qvel[i] = 0.0;
    }
    this->PlaceShootingStandFeetOnGround();

    // Same joints as Isaac, but MuJoCo's muzzle sits ~0.7 m low at 100 m (~0.4 deg).
    // Pitch the base about Y until the ray meets the target height. Do not yaw:
    // an offset board is supposed to start off-axis so the twist policy sees that error.
    // Positive rotation about +Y noses down, so the applied angle is the negative of nose-up.
    float applied_pitch = 0.0f;
    for (int iter = 0; iter < 8; ++iter)
    {
        mj_forward(this->mj_model, this->mj_data);
        this->UpdateShootingStateSensors();
        const ShootingRayImpact ray = this->ComputeShootingRayImpact();
        if (!ray.valid || this->obs.muzzle_pos.size() < 3)
        {
            break;
        }
        const float signed_vert = ray.impact[2] - ray.target[2];
        if (std::fabs(signed_vert) < 0.02f)
        {
            break;
        }
        const float mx = this->obs.muzzle_pos[0];
        const float my = this->obs.muzzle_pos[1];
        const float mz = this->obs.muzzle_pos[2];
        const float target_horiz = std::hypot(ray.target[0] - mx, ray.target[1] - my);
        const float impact_horiz = std::hypot(ray.impact[0] - mx, ray.impact[1] - my);
        const float desired_elev = std::atan2(ray.target[2] - mz, std::max(target_horiz, 1.0e-3f));
        const float current_elev = std::atan2(ray.impact[2] - mz, std::max(impact_horiz, 1.0e-3f));
        const float nose_up = std::clamp(desired_elev - current_elev, -0.05f, 0.05f);
        const float pitch = -nose_up;
        applied_pitch += pitch;

        const float half = 0.5f * pitch;
        const float pw = std::cos(half);
        const float py = std::sin(half);
        const float w = static_cast<float>(this->mj_data->qpos[3]);
        const float x = static_cast<float>(this->mj_data->qpos[4]);
        const float y = static_cast<float>(this->mj_data->qpos[5]);
        const float z = static_cast<float>(this->mj_data->qpos[6]);
        this->mj_data->qpos[3] = pw * w - py * y;
        this->mj_data->qpos[4] = pw * x + py * z;
        this->mj_data->qpos[5] = pw * y + py * w;
        this->mj_data->qpos[6] = pw * z - py * x;
        const double qn = std::sqrt(this->mj_data->qpos[3] * this->mj_data->qpos[3] +
                                    this->mj_data->qpos[4] * this->mj_data->qpos[4] +
                                    this->mj_data->qpos[5] * this->mj_data->qpos[5] +
                                    this->mj_data->qpos[6] * this->mj_data->qpos[6]);
        if (qn > 1.0e-8)
        {
            for (int k = 3; k < 7; ++k)
            {
                this->mj_data->qpos[k] /= qn;
            }
        }
        this->PlaceShootingStandFeetOnGround();
    }

    mj_forward(this->mj_model, this->mj_data);
    this->UpdateShootingStateSensors();
    this->shooting_policy_time_s = 0.0f;
    const ShootingRayImpact ray = this->ComputeShootingRayImpact();
    const float aim_obs = this->ComputeShootingMuzzleAimError().empty()
                              ? 0.0f
                              : this->ComputeShootingMuzzleAimError()[0];
    std::cout << LOGGER::INFO
              << "SIM2SIM handoff: Isaac joint reset"
              << (this->shooting_active_target_is_offset ? " + hip preshape" : "")
              << " + base pitch " << (applied_pitch * 57.2958f) << " deg"
              << " z=" << this->mj_data->qpos[2]
              << " aim_obs=" << aim_obs
              << " lat=" << ray.lateral_miss_m
              << " vert=" << (ray.impact[2] - ray.target[2])
              << ". Joints stay at the training default. "
              << "Hips use PD 80/2, thighs and calves use PD 80/8."
              << std::endl;
}

void RL_Sim::CacheShootingKinematics()
{
    this->shooting_kinematics_ready = false;
    if (!this->mj_model)
    {
        return;
    }

    const char *foot_names[4] = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
    for (int i = 0; i < 4; ++i)
    {
        this->shooting_foot_body_id[i] = mj_name2id(this->mj_model, mjOBJ_BODY, foot_names[i]);
        if (this->shooting_foot_body_id[i] < 0)
        {
            return;
        }
    }
    this->shooting_kinematics_ready = true;
}

bool RL_Sim::CaptureShootingFootPositions(const mjData *d, mjtNum feet[4][3]) const
{
    if (!this->shooting_kinematics_ready || d == nullptr)
    {
        return false;
    }
    for (int i = 0; i < 4; ++i)
    {
        const int bid = this->shooting_foot_body_id[i];
        feet[i][0] = d->xpos[3 * bid + 0];
        feet[i][1] = d->xpos[3 * bid + 1];
        feet[i][2] = d->xpos[3 * bid + 2];
    }
    return true;
}

void RL_Sim::ApplyPlantedBaseYaw(mjData *d, float dyaw, const mjtNum feet[4][3]) const
{
    mjtNum cx = 0.0;
    mjtNum cy = 0.0;
    for (int i = 0; i < 4; ++i)
    {
        cx += feet[i][0];
        cy += feet[i][1];
    }
    cx *= 0.25;
    cy *= 0.25;

    const mjtNum c = std::cos(static_cast<mjtNum>(dyaw));
    const mjtNum s = std::sin(static_cast<mjtNum>(dyaw));
    const mjtNum dx = d->qpos[0] - cx;
    const mjtNum dy = d->qpos[1] - cy;
    d->qpos[0] = cx + c * dx - s * dy;
    d->qpos[1] = cy + s * dx + c * dy;

    const float w = static_cast<float>(d->qpos[3]);
    const float x = static_cast<float>(d->qpos[4]);
    const float y = static_cast<float>(d->qpos[5]);
    const float z = static_cast<float>(d->qpos[6]);
    float yaw = std::atan2(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z));
    const float roll = std::atan2(2.0f * (w * x + y * z), 1.0f - 2.0f * (x * x + y * y));
    const float sinp = 2.0f * (w * y - z * x);
    const float pitch = (std::fabs(sinp) >= 1.0f)
                            ? std::copysign(static_cast<float>(M_PI) / 2.0f, sinp)
                            : std::asin(sinp);
    yaw += dyaw;

    const float cr = std::cos(roll * 0.5f);
    const float sr = std::sin(roll * 0.5f);
    const float cp = std::cos(pitch * 0.5f);
    const float sp = std::sin(pitch * 0.5f);
    const float cyaw = std::cos(yaw * 0.5f);
    const float syaw = std::sin(yaw * 0.5f);
    d->qpos[3] = cr * cp * cyaw + sr * sp * syaw;
    d->qpos[4] = sr * cp * cyaw - cr * sp * syaw;
    d->qpos[5] = cr * sp * cyaw + sr * cp * syaw;
    d->qpos[6] = cr * cp * syaw - sr * sp * cyaw;
}

void RL_Sim::RestoreShootingFootIK(mjData *d, const mjtNum feet[4][3]) const
{
    const char *leg_joints[4][3] = {
        {"FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"},
        {"FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"},
        {"RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"},
        {"RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"},
    };
    const int nv = this->mj_model->nv;
    std::vector<mjtNum> jacp(static_cast<size_t>(3 * nv), 0.0);
    for (int iter = 0; iter < 12; ++iter)
    {
        mj_forward(this->mj_model, d);
        for (int f = 0; f < 4; ++f)
        {
            const int bid = this->shooting_foot_body_id[f];
            // XY only: leave Z to contact so the live robot does not kneel.
            const mjtNum err[3] = {
                feet[f][0] - d->xpos[3 * bid + 0],
                feet[f][1] - d->xpos[3 * bid + 1],
                0.0,
            };
            if (err[0] * err[0] + err[1] * err[1] < 1e-12)
            {
                continue;
            }
            mj_jacBody(this->mj_model, d, jacp.data(), nullptr, bid);
            mjtNum J[9];
            int jnt_id[3];
            int dof[3];
            for (int j = 0; j < 3; ++j)
            {
                jnt_id[j] = mj_name2id(this->mj_model, mjOBJ_JOINT, leg_joints[f][j]);
                dof[j] = this->mj_model->jnt_dofadr[jnt_id[j]];
                J[0 * 3 + j] = jacp[0 * nv + dof[j]];
                J[1 * 3 + j] = jacp[1 * nv + dof[j]];
                J[2 * 3 + j] = jacp[2 * nv + dof[j]];
            }
            mjtNum A[9];
            mjtNum b[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    mjtNum s = (i == j) ? 1e-3 : 0.0;
                    for (int k = 0; k < 3; ++k)
                    {
                        s += J[k * 3 + i] * J[k * 3 + j];
                    }
                    A[i * 3 + j] = s;
                }
                for (int k = 0; k < 3; ++k)
                {
                    b[i] += J[k * 3 + i] * err[k];
                }
            }
            mjtNum dq[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < 3; ++i)
            {
                int pivot = i;
                mjtNum best = std::fabs(A[i * 3 + i]);
                for (int r = i + 1; r < 3; ++r)
                {
                    if (std::fabs(A[r * 3 + i]) > best)
                    {
                        best = std::fabs(A[r * 3 + i]);
                        pivot = r;
                    }
                }
                if (best < 1e-10)
                {
                    dq[0] = dq[1] = dq[2] = 0.0;
                    break;
                }
                if (pivot != i)
                {
                    for (int c = i; c < 3; ++c)
                    {
                        std::swap(A[i * 3 + c], A[pivot * 3 + c]);
                    }
                    std::swap(b[i], b[pivot]);
                }
                const mjtNum diag = A[i * 3 + i];
                for (int r = i + 1; r < 3; ++r)
                {
                    const mjtNum fac = A[r * 3 + i] / diag;
                    for (int c = i; c < 3; ++c)
                    {
                        A[r * 3 + c] -= fac * A[i * 3 + c];
                    }
                    b[r] -= fac * b[i];
                }
            }
            if (std::fabs(A[0]) > 1e-10 && std::fabs(A[4]) > 1e-10 && std::fabs(A[8]) > 1e-10)
            {
                for (int i = 2; i >= 0; --i)
                {
                    mjtNum s = b[i];
                    for (int c = i + 1; c < 3; ++c)
                    {
                        s -= A[i * 3 + c] * dq[c];
                    }
                    dq[i] = s / A[i * 3 + i];
                }
            }
            for (int j = 0; j < 3; ++j)
            {
                const int qadr = this->mj_model->jnt_qposadr[jnt_id[j]];
                mjtNum q = d->qpos[qadr] + std::clamp(dq[j], -0.15, 0.15);
                if (this->mj_model->jnt_limited[jnt_id[j]])
                {
                    q = std::clamp(q, this->mj_model->jnt_range[2 * jnt_id[j]],
                                   this->mj_model->jnt_range[2 * jnt_id[j] + 1]);
                }
                d->qpos[qadr] = q;
            }
        }
    }
    mj_forward(this->mj_model, d);
}

void RL_Sim::SolveShootingPlantedTwist(float desired_yaw)
{
    if (!this->mj_model || !this->mj_data || this->shooting_ik_scratch == nullptr)
    {
        return;
    }
    if (!this->shooting_kinematics_ready)
    {
        this->CacheShootingKinematics();
    }
    if (!this->shooting_kinematics_ready || !this->params.Has("joint_names"))
    {
        return;
    }

    mj_copyData(this->shooting_ik_scratch, this->mj_model, this->mj_data);
    mj_forward(this->mj_model, this->shooting_ik_scratch);

    mjtNum feet[4][3];
    if (!this->CaptureShootingFootPositions(this->shooting_ik_scratch, feet))
    {
        return;
    }

    const float w = static_cast<float>(this->shooting_ik_scratch->qpos[3]);
    const float x = static_cast<float>(this->shooting_ik_scratch->qpos[4]);
    const float y = static_cast<float>(this->shooting_ik_scratch->qpos[5]);
    const float z = static_cast<float>(this->shooting_ik_scratch->qpos[6]);
    const float yaw = std::atan2(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z));
    float dyaw = desired_yaw - yaw;
    while (dyaw > static_cast<float>(M_PI))
    {
        dyaw -= 2.0f * static_cast<float>(M_PI);
    }
    while (dyaw < -static_cast<float>(M_PI))
    {
        dyaw += 2.0f * static_cast<float>(M_PI);
    }

    this->ApplyPlantedBaseYaw(this->shooting_ik_scratch, dyaw, feet);
    this->RestoreShootingFootIK(this->shooting_ik_scratch, feet);

    const auto names = this->params.Get<std::vector<std::string>>("joint_names");
    const int n = this->params.Get<int>("num_of_dofs");
    this->shooting_twist_q.resize(static_cast<size_t>(n), 0.0f);
    for (int i = 0; i < n && i < static_cast<int>(names.size()); ++i)
    {
        const int jid = mj_name2id(this->mj_model, mjOBJ_JOINT, names[i].c_str());
        if (jid < 0)
        {
            continue;
        }
        this->shooting_twist_q[static_cast<size_t>(i)] =
            static_cast<float>(this->shooting_ik_scratch->qpos[this->mj_model->jnt_qposadr[jid]]);
    }
    this->shooting_twist_valid = true;
    this->shooting_twist_last_azimuth = desired_yaw;
}

void RL_Sim::ApplyShootingAimBaseYawCorrection()
{
    // Never write live qpos. Solve planted-twist joints on a scratch copy and
    // let PD twist the body while contact keeps the feet in place.
    if (this->scene_name != "shooting_stability" || !this->mj_model || !this->mj_data)
    {
        return;
    }
    if (!this->shooting_aim_enabled || this->params.Get<int>("shooting_aim_servo", 1) == 0)
    {
        return;
    }
    if (this->params.Get<int>("shooting_planted_twist", 1) == 0)
    {
        return;
    }

    const float azimuth = this->obs.shooting_target_azimuth;
    if (std::fabs(azimuth) < 0.005f)
    {
        this->shooting_twist_valid = false;
        this->shooting_twist_blend = 0.0f;
        return;
    }
    const bool azimuth_changed =
        std::fabs(azimuth - this->shooting_twist_last_azimuth) > 1e-4f;
    if (this->shooting_twist_need_solve || !this->shooting_twist_valid || azimuth_changed)
    {
        this->SolveShootingPlantedTwist(azimuth);
        this->shooting_twist_blend = 0.0f;
        this->shooting_twist_need_solve = false;
    }

    if (this->shooting_twist_valid)
    {
        const float dt = std::max(this->params.Get<float>("dt", 0.005f), 1e-6f);
        const float dur = std::max(this->params.Get<float>("shooting_planted_twist_duration", 1.0f), dt);
        this->shooting_twist_blend = std::min(1.0f, this->shooting_twist_blend + dt / dur);
    }
}

void RL_Sim::SetupSysJoystick(const std::string& device, int bits)
{
    this->sys_js = std::make_unique<Joystick>(device);
    this->sys_js_active = this->sys_js && this->sys_js->isFound();
    if (!this->sys_js_active)
    {
        std::cout << LOGGER::ERROR << "Joystick [" << device << "] open failed." << std::endl;
        // exit(1);
    }

    this->sys_js_max_value = (1 << (bits - 1));
}

void RL_Sim::GetSysJoystick()
{
    // Clear all button event states
    for (int i = 0; i < 20; ++i)
    {
        this->sys_js_button[i].on_press = false;
        this->sys_js_button[i].on_release = false;
    }

    // Check if joystick is valid before using
    if (!this->sys_js_active)
    {
        return;
    }

    while (this->sys_js->sample(&this->sys_js_event))
    {
        if (this->sys_js_event.isButton())
        {
            this->sys_js_button[this->sys_js_event.number].update(this->sys_js_event.value);
        }
        else if (this->sys_js_event.isAxis())
        {
            double normalized = double(this->sys_js_event.value) / this->sys_js_max_value;
            if (std::abs(normalized) < this->axis_deadzone)
            {
                this->sys_js_axis[this->sys_js_event.number] = 0;
            }
            else
            {
                this->sys_js_axis[this->sys_js_event.number] = this->sys_js_event.value;
            }
        }
    }

    if (this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::A);
    if (this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::B);
    if (this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::X);
    if (this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->sys_js_button[4].on_press) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->sys_js_button[4].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->sys_js_button[4].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->sys_js_button[4].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->sys_js_button[4].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->sys_js_button[4].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->sys_js_button[5].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->sys_js_button[5].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->sys_js_button[5].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->sys_js_button[5].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->sys_js_button[5].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->sys_js_button[5].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->sys_js_button[4].pressed && this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::LB_RB);

    float ly = -float(this->sys_js_axis[1]) / float(this->sys_js_max_value);
    float lx = -float(this->sys_js_axis[0]) / float(this->sys_js_max_value);
    float rx = -float(this->sys_js_axis[3]) / float(this->sys_js_max_value);

    bool has_input = (ly != 0.0f || lx != 0.0f || rx != 0.0f);

    if (has_input)
    {
        this->control.x = ly;
        this->control.y = lx;
        this->control.yaw = rx;
        this->sys_js_active = true;
    }
    else if (this->sys_js_active)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
        this->sys_js_active = false;
    }
}

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        this->obs.commands = {0.0f, 0.0f, 0.0f};
        if (!this->shooting_stability_mode)
        {
            this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        }
        //not currently available for non-ros mujoco version
        // if (this->control.navigation_mode)
        // {
        //     this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};
        // }
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        this->obs.actions = this->Forward();
        if (this->shooting_stability_mode &&
            (this->episode_length_buf == 1 || this->episode_length_buf == 10 || this->episode_length_buf == 50))
        {
            this->DumpShootingPolicyDebugObs("step_" + std::to_string(this->episode_length_buf));
        }
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (!this->output_dof_pos.empty())
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (!this->output_dof_vel.empty())
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (!this->output_dof_tau.empty())
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        std::vector<float> tau_est(this->params.Get<int>("num_of_dofs"), 0.0f);
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]];
        }
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Sim::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (this->params.Get<std::vector<int>>("observations_history").size() != 0)
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    else
    {
        return actions;
    }
}

void RL_Sim::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(mj_data->sensordata[i]);
        // this->plot_target_joint_pos[i].push_back();  // TODO
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

void RL_Sim::SetSimulationPaused(bool paused)
{
    this->simulation_running = !paused;
    if (this->sim)
    {
        this->sim->run = this->simulation_running ? 1 : 0;
    }
}

void RL_Sim::OnSimulationPhysicsReset()
{
    if (!this->mj_model || !this->mj_data)
    {
        return;
    }
    mj_resetData(this->mj_model, this->mj_data);
    // Prefer crouch keyframe so spawn matches Gazebo/open-loop get-up start.
    const int crouch_key = mj_name2id(this->mj_model, mjOBJ_KEY, "crouch");
    if (crouch_key >= 0)
    {
        mj_resetDataKeyframe(this->mj_model, this->mj_data, crouch_key);
    }
    mj_forward(this->mj_model, this->mj_data);
    if (this->scene_name == "shooting_stability")
    {
        this->SetupShootingRange();
    }
}

// Signal handler for Ctrl+C
void signalHandler(int signum)
{
    std::cout << LOGGER::INFO << "Received signal " << signum << ", exiting..." << std::endl;
    if (RL_Sim::instance && RL_Sim::instance->sim)
    {
        RL_Sim::instance->sim->exitrequest.store(1);
    }
}

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);
    RL_Sim rl_sar(argc, argv);
    return 0;
}
