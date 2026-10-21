/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sdk.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

void RL::StateController(const RobotState<float>* state, RobotCommand<float>* command)
{
    auto updateState = [&](std::shared_ptr<FSMState> statePtr)
    {
        if (auto rl_fsm_state = std::dynamic_pointer_cast<RLFSMState>(statePtr))
        {
            rl_fsm_state->fsm_state = state;
            rl_fsm_state->fsm_command = command;
        }
    };
    for (auto& pair : fsm.states_)
    {
        updateState(pair.second);
    }

    fsm.Run();

    this->motiontime++;

    if (this->shooting_stability_mode)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
    }
    else if (this->control.current_keyboard == Input::Keyboard::W)
    {
        this->control.x += 0.1f;
    }
    else if (this->control.current_keyboard == Input::Keyboard::S)
    {
        this->control.x -= 0.1f;
    }
    else if (this->control.current_keyboard == Input::Keyboard::A)
    {
        this->control.y += 0.1f;
    }
    else if (this->control.current_keyboard == Input::Keyboard::D)
    {
        this->control.y -= 0.1f;
    }
    else if (this->control.current_keyboard == Input::Keyboard::Q)
    {
        this->control.yaw += 0.1f;
    }
    else if (this->control.current_keyboard == Input::Keyboard::E)
    {
        this->control.yaw -= 0.1f;
    }
    else if (this->control.current_keyboard == Input::Keyboard::Space)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
    }
    if (this->control.current_keyboard == Input::Keyboard::N || this->control.current_gamepad == Input::Gamepad::X)
    {
        if (!this->shooting_stability_mode)
        {
            this->control.navigation_mode = !this->control.navigation_mode;
            std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (this->control.navigation_mode ? "ON" : "OFF") << std::endl;
        }
    }
}

std::vector<float> RL::ComputeObservation()
{
    std::vector<std::vector<float>> obs_list;

    for (const std::string &observation : this->params.Get<std::vector<std::string>>("observations"))
    {
        // ============= Base Observations =============
        if (observation == "lin_vel")
        {
            obs_list.push_back(this->obs.lin_vel * this->params.Get<float>("lin_vel_scale"));
        }
        else if (observation == "ang_vel")
        {
            // In ROS1 Gazebo, the coordinate system for angular velocity is in the world coordinate system.
            // In ROS2 Gazebo, mujoco and real robot, the coordinate system for angular velocity is in the body coordinate system.
            if (this->ang_vel_axis == "body")
            {
                obs_list.push_back(this->obs.ang_vel * this->params.Get<float>("ang_vel_scale"));
            }
            else if (this->ang_vel_axis == "world")
            {
                obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.ang_vel) * this->params.Get<float>("ang_vel_scale"));
            }
        }
        else if (observation == "gravity_vec")
        {
            obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec));
        }
        else if (observation == "commands")
        {
            obs_list.push_back(this->obs.commands * this->params.Get<std::vector<float>>("commands_scale"));
        }
        else if (observation == "dof_pos")
        {
            std::vector<float> dof_pos_rel = this->obs.dof_pos - this->params.Get<std::vector<float>>("default_dof_pos");
            for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
            {
                dof_pos_rel[i] = 0.0f;
            }
            obs_list.push_back(dof_pos_rel * this->params.Get<float>("dof_pos_scale"));
        }
        else if (observation == "dof_vel")
        {
            obs_list.push_back(this->obs.dof_vel * this->params.Get<float>("dof_vel_scale"));
        }
        else if (observation == "actions")
        {
            obs_list.push_back(this->obs.actions);
        }
        else if (observation == "muzzle_aim_error")
        {
            if (this->shooting_stability_mode && !this->shooting_aim_enabled)
            {
                obs_list.push_back(this->NeutralShootingMuzzleAimError());
            }
            else
            {
                obs_list.push_back(this->ComputeShootingMuzzleAimError());
            }
        }
        else if (observation == "shooting_countdown")
        {
            if (this->shooting_stability_mode && !this->shooting_aim_enabled)
            {
                obs_list.push_back(this->NeutralShootingCountdown());
            }
            else
            {
                obs_list.push_back(this->ComputeShootingCountdown());
            }
        }
        else if (observation == "target_azimuth")
        {
            if (this->shooting_stability_mode && !this->shooting_aim_enabled)
            {
                obs_list.push_back(this->NeutralShootingTargetAzimuth());
            }
            else
            {
                obs_list.push_back(this->ComputeShootingTargetAzimuth());
            }
        }
        // ============= Other Observations =============
        else if (observation == "whole_body_tracking/motion_command")
        {
            std::vector<float> motion_cmd;
            if (this->motion_loader)
            {
                auto joint_pos_sdk = this->motion_loader->GetJointPos();
                auto joint_vel_sdk = this->motion_loader->GetJointVel();
                auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
                std::vector<float> joint_pos_training(joint_mapping.size());
                std::vector<float> joint_vel_training(joint_mapping.size());
                for (size_t i = 0; i < joint_mapping.size(); ++i)
                {
                    joint_pos_training[i] = joint_pos_sdk[joint_mapping[i]];
                    joint_vel_training[i] = joint_vel_sdk[joint_mapping[i]];
                }
                motion_cmd.insert(motion_cmd.end(), joint_pos_training.begin(), joint_pos_training.end());
                motion_cmd.insert(motion_cmd.end(), joint_vel_training.begin(), joint_vel_training.end());
            }
            else
            {
                motion_cmd.resize(this->params.Get<int>("num_of_dofs") * 2, 0.0f);
            }
            obs_list.push_back(motion_cmd);
        }
        else if (observation == "whole_body_tracking/motion_anchor_ori_b")
        {
            std::vector<float> anchor_ori(6, 0.0f);
            if (this->motion_loader)
            {
                auto waist_sdk_indices = this->params.Get<std::vector<int>>("waist_joint_indices");
                std::vector<float> waist_angles = {
                    this->obs.dof_pos[InverseJointMapping(waist_sdk_indices[0])],
                    this->obs.dof_pos[InverseJointMapping(waist_sdk_indices[1])],
                    this->obs.dof_pos[InverseJointMapping(waist_sdk_indices[2])]
                };
                std::vector<float> robot_torso_quat_w = MotionLoader::ComputeTorsoQuat(this->obs.base_quat, waist_angles);
                std::vector<float> ref_torso_quat_w = this->motion_loader->GetAnchorQuat();
                std::vector<float> init_quat = this->motion_loader->GetInitQuat();
                std::vector<float> motion_anchor_quat_w = QuaternionMultiply(init_quat, ref_torso_quat_w);
                std::vector<float> robot_quat_inv = QuaternionConjugate(robot_torso_quat_w);
                std::vector<float> relative_quat = QuaternionMultiply(robot_quat_inv, motion_anchor_quat_w);
                std::vector<float> rot_matrix = QuaternionToRotationMatrix(relative_quat);
                anchor_ori = MatrixFirstTwoColumns(rot_matrix);
            }
            obs_list.push_back(anchor_ori);
        }
        else if (observation == "RoboMimic_Deploy/phase")
        {
            float motion_time = this->episode_length_buf * this->params.Get<float>("dt") * this->params.Get<int>("decimation");
            float count = motion_time;
            float phase = count / this->motion_length;
            std::vector<float> phase_vec = {phase};
            obs_list.push_back(phase_vec);
        }
    }

    this->obs_dims.clear();
    for (const auto& obs : obs_list)
    {
       this->obs_dims.push_back(obs.size());
    }

    std::vector<float> obs;
    for (const auto& obs_vec : obs_list)
    {
        obs.insert(obs.end(), obs_vec.begin(), obs_vec.end());
    }
    std::vector<float> clamped_obs = clamp(obs, -this->params.Get<float>("clip_obs"), this->params.Get<float>("clip_obs"));
    return clamped_obs;
}

void RL::InitObservations()
{
    this->obs.lin_vel = {0.0f, 0.0f, 0.0f};
    this->obs.ang_vel = {0.0f, 0.0f, 0.0f};
    this->obs.gravity_vec = {0.0f, 0.0f, -1.0f};
    this->obs.commands = {0.0f, 0.0f, 0.0f};
    this->obs.base_quat = {1.0f, 0.0f, 0.0f, 0.0f};  // wxyz
    this->obs.dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    this->obs.dof_vel.clear();
    this->obs.dof_vel.resize(this->params.Get<int>("num_of_dofs"), 0.0f);
    this->obs.actions.clear();
    this->obs.actions.resize(this->params.Get<int>("num_of_dofs"), 0.0f);
    this->obs.muzzle_pos = {0.4f, 0.0f, 0.672f};
    this->obs.muzzle_quat = {1.0f, 0.0f, 0.0f, 0.0f};
    this->obs.base_pos = {0.0f, 0.0f, 0.0f};
    this->ComputeObservation();
}

ShootingRayImpact RL::ComputeShootingRayImpact() const
{
    ShootingRayImpact out;
    if (this->obs.muzzle_pos.size() < 3 || this->obs.muzzle_quat.size() < 4)
    {
        return out;
    }

    const float azimuth = this->obs.shooting_target_azimuth;
    out.target[0] = this->shooting_target_distance * std::cos(azimuth);
    out.target[1] = this->shooting_target_distance * std::sin(azimuth);
    out.target[2] = this->shooting_target_height;

    const std::vector<float> muzzle_forward = QuatRotate(
        this->obs.muzzle_quat,
        {this->shooting_muzzle_forward_axis[0], this->shooting_muzzle_forward_axis[1], this->shooting_muzzle_forward_axis[2]});
    const float forward_norm = std::max(Norm(muzzle_forward), 1e-6f);
    const std::vector<float> ray_dir = {
        muzzle_forward[0] / forward_norm,
        muzzle_forward[1] / forward_norm,
        muzzle_forward[2] / forward_norm,
    };
    const float muzzle_x = this->obs.muzzle_pos[0];
    const float muzzle_y = this->obs.muzzle_pos[1];
    const float muzzle_z = this->obs.muzzle_pos[2];

    // Target plane normal points back toward the shooter.
    const float nx = -std::cos(azimuth);
    const float ny = -std::sin(azimuth);
    const float nz = 0.0f;
    const float plane_offset = nx * out.target[0] + ny * out.target[1] + nz * out.target[2];
    const float denom = nx * ray_dir[0] + ny * ray_dir[1] + nz * ray_dir[2];

    out.impact[0] = out.target[0];
    out.impact[1] = out.target[1];
    out.impact[2] = out.target[2];
    if (std::fabs(denom) > 1e-6f)
    {
        const float plane_dist = plane_offset - (nx * muzzle_x + ny * muzzle_y + nz * muzzle_z);
        const float t = plane_dist / denom;
        if (t > 0.0f)
        {
            out.impact[0] = muzzle_x + t * ray_dir[0];
            out.impact[1] = muzzle_y + t * ray_dir[1];
            out.impact[2] = muzzle_z + t * ray_dir[2];
        }
    }

    out.delta[0] = out.impact[0] - out.target[0];
    out.delta[1] = out.impact[1] - out.target[1];
    out.delta[2] = out.impact[2] - out.target[2];
    // Target board frame: lateral along world Y when azimuth~0; keep simple abs for telemetry.
    out.lateral_miss_m = std::fabs(out.delta[1] * std::cos(azimuth) - out.delta[0] * std::sin(azimuth));
    out.vertical_miss_m = std::fabs(out.delta[2]);
    out.aim_error = std::clamp(out.lateral_miss_m / (2.0f * this->shooting_hit_radius), 0.0f, 1.0f);
    // Require both axes near the board: height is 1.0 m so use 0.5 m half-height as vertical budget.
    out.hit = (out.lateral_miss_m < this->shooting_hit_radius) && (out.vertical_miss_m < 0.5f);
    out.valid = true;
    return out;
}

std::vector<float> RL::ComputeShootingMuzzleAimError() const
{
    // robot_lab mdp._aim_lateral_miss_m: dist * ||ray × dir_to_target||.
    // The name says lateral; the value includes pitch and yaw, then / (2 * 0.25 m).
    if (this->obs.muzzle_pos.size() < 3 || this->obs.muzzle_quat.size() < 4)
    {
        return {0.0f};
    }

    const float azimuth = this->obs.shooting_target_azimuth;
    const float target[3] = {
        this->shooting_target_distance * std::cos(azimuth),
        this->shooting_target_distance * std::sin(azimuth),
        this->shooting_target_height,
    };
    const float to_target[3] = {
        target[0] - this->obs.muzzle_pos[0],
        target[1] - this->obs.muzzle_pos[1],
        target[2] - this->obs.muzzle_pos[2],
    };
    const float dist = std::sqrt(to_target[0] * to_target[0] + to_target[1] * to_target[1] + to_target[2] * to_target[2]);
    if (dist < 1.0e-6f)
    {
        return {0.0f};
    }
    const float dir[3] = {to_target[0] / dist, to_target[1] / dist, to_target[2] / dist};
    const std::vector<float> muzzle_forward = QuatRotate(
        this->obs.muzzle_quat,
        {this->shooting_muzzle_forward_axis[0], this->shooting_muzzle_forward_axis[1], this->shooting_muzzle_forward_axis[2]});
    const float forward_norm = std::max(Norm(muzzle_forward), 1.0e-6f);
    const float ray[3] = {
        muzzle_forward[0] / forward_norm,
        muzzle_forward[1] / forward_norm,
        muzzle_forward[2] / forward_norm,
    };
    const float cross[3] = {
        ray[1] * dir[2] - ray[2] * dir[1],
        ray[2] * dir[0] - ray[0] * dir[2],
        ray[0] * dir[1] - ray[1] * dir[0],
    };
    const float sin_theta = std::min(
        1.0f, std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]));
    const float miss_m = dist * sin_theta;
    const float aim_error = std::clamp(miss_m / (2.0f * this->shooting_hit_radius), 0.0f, 1.0f);
    return {aim_error};
}

void RL::MaybeLogShootingTelemetry()
{
    if (!this->shooting_stability_mode || !this->params.Get<bool>("shooting_telemetry", true))
    {
        return;
    }

    const float dt = std::max(this->params.Get<float>("dt", 0.005f), 1e-6f);
    const float interval_s = std::max(this->params.Get<float>("shooting_telemetry_interval", 2.0f), dt);
    const int period = std::max(static_cast<int>(std::lround(interval_s / dt)), 1);
    if (this->motiontime <= 0 || (this->motiontime % period) != 0)
    {
        return;
    }
    if (this->shooting_telemetry_last_motiontime == this->motiontime)
    {
        return;
    }
    this->shooting_telemetry_last_motiontime = this->motiontime;

    // Prefer live IMU / joint state; fall back to last policy obs.
    std::vector<float> quat = this->robot_state.imu.quaternion;
    if (quat.size() < 4)
    {
        quat = this->obs.base_quat;
    }
    std::vector<float> gyro = this->robot_state.imu.gyroscope;
    if (gyro.size() < 3)
    {
        gyro = this->obs.ang_vel;
    }
    const std::vector<float> euler = QuaternionToEuler(quat);
    constexpr float rad2deg = 57.2957795f;
    const float roll_deg = euler[0] * rad2deg;
    const float pitch_deg = euler[1] * rad2deg;
    const float yaw_deg = euler[2] * rad2deg;

    const ShootingRayImpact ray = this->ComputeShootingRayImpact();
    const int num_dofs = this->params.Get<int>("num_of_dofs");
    const auto joint_names = this->params.Has("joint_names")
                                 ? this->params.Get<std::vector<std::string>>("joint_names")
                                 : std::vector<std::string>{};
    const auto &q = this->robot_state.motor_state.q;
    const auto &dq = this->robot_state.motor_state.dq;
    const auto &tau_est = this->robot_state.motor_state.tau_est;
    const auto &cmd = this->robot_command.motor_command;

    std::vector<float> tau_cmd(num_dofs, 0.0f);
    for (int i = 0; i < num_dofs; ++i)
    {
        const float qi = (static_cast<int>(q.size()) > i) ? q[i] : 0.0f;
        const float dqi = (static_cast<int>(dq.size()) > i) ? dq[i] : 0.0f;
        const float q_des = (static_cast<int>(cmd.q.size()) > i) ? cmd.q[i] : qi;
        const float dq_des = (static_cast<int>(cmd.dq.size()) > i) ? cmd.dq[i] : 0.0f;
        const float tau_ff = (static_cast<int>(cmd.tau.size()) > i) ? cmd.tau[i] : 0.0f;
        const float kp = (static_cast<int>(cmd.kp.size()) > i) ? cmd.kp[i] : 0.0f;
        const float kd = (static_cast<int>(cmd.kd.size()) > i) ? cmd.kd[i] : 0.0f;
        tau_cmd[i] = tau_ff + kp * (q_des - qi) + kd * (dq_des - dqi);
    }

    const float t_s = static_cast<float>(this->motiontime) * dt;
    const float base_x = (this->obs.base_pos.size() > 0) ? this->obs.base_pos[0] : 0.0f;
    const float base_y = (this->obs.base_pos.size() > 1) ? this->obs.base_pos[1] : 0.0f;
    const float base_z = (this->obs.base_pos.size() > 2) ? this->obs.base_pos[2] : 0.0f;

    const bool stand_ok = this->ComputeShootingStandOk(pitch_deg, base_z);
    float hind_calf = 0.0f;
    if (q.size() >= 12)
    {
        hind_calf = std::max(q[10], q[11]);
    }
    const char *phase = "deploy";
    if (this->IsShootingSim2Sim())
    {
        if (this->shooting_sim2sim_settling)
        {
            phase = "settle";
        }
        else
        {
            phase = this->shooting_active_target_is_offset ? "twist" : "hold";
        }
    }
    const float policy_aim = this->ComputeShootingMuzzleAimError().empty()
                                 ? 0.0f
                                 : this->ComputeShootingMuzzleAimError()[0];
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "[Shooting] t=" << t_s << "s " << (this->IsShootingSim2Sim() ? "SIM2SIM" : "DEPLOY")
        << " " << phase
        << (this->shooting_loaded_model_name.empty() ? "" : (" " + this->shooting_loaded_model_name))
        << " stand=" << (stand_ok ? "YES" : "NO")
        << " rpy=" << roll_deg << "," << pitch_deg << "," << yaw_deg
        << " z=" << base_z
        << " lat=" << ray.lateral_miss_m
        << " vert=" << ray.vertical_miss_m
        << " aim=" << policy_aim
        << " hit=" << (ray.hit ? "YES" : "NO")
        << " hind_calf=" << hind_calf;
    std::cout << std::endl << LOGGER::INFO << oss.str() << std::endl;

    // Optional CSV (one row per print).
    if (this->params.Get<bool>("shooting_telemetry_csv", true))
    {
        if (!this->shooting_telemetry_csv_ready)
        {
            this->shooting_telemetry_csv_filename =
                std::string(POLICY_DIR) + "/" + this->robot_name + "/shooting_telemetry.csv";
            std::ofstream header(this->shooting_telemetry_csv_filename.c_str());
            header << "t,roll_deg,pitch_deg,yaw_deg,gyro_x,gyro_y,gyro_z,base_x,base_y,base_z,"
                   << "target_x,target_y,target_z,impact_x,impact_y,impact_z,"
                   << "delta_x,delta_y,delta_z,lateral_miss_m,vertical_miss_m,aim_error,hit,"
                   << "aim_on,recoil_on,ctrl_mode,stand_ok";
            for (int i = 0; i < num_dofs; ++i)
            {
                const std::string name = (static_cast<int>(joint_names.size()) > i) ? joint_names[i] : ("joint_" + std::to_string(i));
                header << ",q_" << name << ",tau_est_" << name << ",tau_cmd_" << name;
            }
            header << "\n";
            header.close();
            this->shooting_telemetry_csv_ready = true;
            std::cout << LOGGER::INFO << "Shooting telemetry CSV: " << this->shooting_telemetry_csv_filename << std::endl;
        }

        std::ofstream file(this->shooting_telemetry_csv_filename.c_str(), std::ios_base::app);
        file << t_s << ","
             << roll_deg << "," << pitch_deg << "," << yaw_deg << ","
             << (gyro.size() > 0 ? gyro[0] : 0.0f) << ","
             << (gyro.size() > 1 ? gyro[1] : 0.0f) << ","
             << (gyro.size() > 2 ? gyro[2] : 0.0f) << ","
             << base_x << "," << base_y << "," << base_z << ","
             << ray.target[0] << "," << ray.target[1] << "," << ray.target[2] << ","
             << ray.impact[0] << "," << ray.impact[1] << "," << ray.impact[2] << ","
             << ray.delta[0] << "," << ray.delta[1] << "," << ray.delta[2] << ","
             << ray.lateral_miss_m << "," << ray.vertical_miss_m << "," << policy_aim << ","
             << (ray.hit ? 1 : 0) << ","
             << (this->shooting_aim_enabled ? 1 : 0) << ","
             << (this->shooting_recoil_enabled ? 1 : 0) << ","
             << (this->IsShootingSim2Sim() ? 1 : 0) << ","
             << (this->ComputeShootingStandOk(pitch_deg, base_z) ? 1 : 0);
        for (int i = 0; i < num_dofs; ++i)
        {
            const float qi = (static_cast<int>(q.size()) > i) ? q[i] : 0.0f;
            const float tau_e = (static_cast<int>(tau_est.size()) > i) ? tau_est[i] : 0.0f;
            file << "," << qi << "," << tau_e << "," << tau_cmd[i];
        }
        file << "\n";
        file.close();
    }
}

std::vector<float> RL::ComputeShootingCountdown() const
{
    // RunModel increments episode_length_buf before the forward. Isaac's first
    // (reset) observation uses common_step_counter == 0, so subtract one.
    const float policy_dt = this->params.Get<float>("dt") * this->params.Get<int>("decimation");
    const int cycle = std::max(static_cast<int>(this->shooting_recoil_interval / std::max(policy_dt, 1e-6f)), 1);
    const unsigned long long step = this->episode_length_buf > 0 ? this->episode_length_buf - 1 : 0;
    const float phase = static_cast<float>(step % static_cast<unsigned long long>(cycle)) / static_cast<float>(cycle);
    return {2.0f * phase - 1.0f};
}

std::vector<float> RL::ComputeShootingTargetAzimuth() const
{
    const float azimuth = this->obs.shooting_target_azimuth;
    return {std::sin(azimuth), std::cos(azimuth)};
}

void RL::InitOutputs()
{
    int num_of_dofs = this->params.Get<int>("num_of_dofs");
    this->output_dof_tau.clear();
    this->output_dof_tau.resize(num_of_dofs, 0.0f);
    this->output_dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    this->output_dof_vel.clear();
    this->output_dof_vel.resize(num_of_dofs, 0.0f);
}

void RL::InitControl()
{
    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;
}

void RL::InitJointNum(size_t num_joints)
{
    this->robot_state.motor_state.resize(num_joints);
    this->start_state.motor_state.resize(num_joints);
    this->now_state.motor_state.resize(num_joints);
    this->robot_command.motor_command.resize(num_joints);
}

void RL::ConfigureShootingStabilityMode(bool enabled)
{
    this->shooting_stability_mode = enabled;
    if (enabled)
    {
        this->shooting_scene_ready = false;
        this->SetSimulationPaused(true);
        this->ResetShootingControlState();
        std::cout << LOGGER::NOTE
                  << "Shooting stability controls: [0] stand | [1] frontal/hold (red) | [3] offset/twist (blue) | [2] recoil | [R] reset"
                  << std::endl;
    }
}

void RL::EnsureShootingSimulationRunning()
{
    if (!this->shooting_stability_mode)
    {
        return;
    }
    this->SetSimulationPaused(false);
}

void RL::ResetShootingControlState()
{
    this->shooting_aim_enabled = false;
    this->shooting_recoil_enabled = false;
    this->episode_length_buf = 0;
    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;
    this->shooting_policy_blend_active = false;
    this->shooting_policy_blend_percent = 1.0f;
    this->shooting_policy_blend_start_q.clear();
    this->shooting_stabilize_logged = false;
    this->shooting_aim_pitch_trim = 0.0f;
    this->shooting_aim_yaw_trim = 0.0f;
    this->shooting_aim_pitch_cmd = 0.0f;
    this->shooting_aim_yaw_cmd = 0.0f;
    this->shooting_twist_q.clear();
    this->shooting_twist_blend = 0.0f;
    this->shooting_twist_valid = false;
    this->shooting_twist_need_solve = false;
    this->shooting_sim2sim_settling = false;
    this->shooting_sim2sim_settle_s = 0.0f;
    this->shooting_policy_time_s = 0.0f;
}

void RL::SeedShootingPolicyFromCurrentState(const RobotState<float> *state)
{
    if (state == nullptr)
    {
        return;
    }
    const int num_dofs = this->params.Get<int>("num_of_dofs");
    const auto default_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    const auto action_scale = this->params.Get<std::vector<float>>("action_scale");

    this->obs.dof_pos = state->motor_state.q;
    this->obs.dof_vel = state->motor_state.dq;
    this->obs.base_quat = state->imu.quaternion;
    this->obs.ang_vel = state->imu.gyroscope;
    this->obs.actions.assign(num_dofs, 0.0f);
    this->output_dof_pos.resize(num_dofs);
    this->output_dof_vel.assign(num_dofs, 0.0f);

    // Deploy (mix≈0): seed targets to the aim anchor. Sim2sim: seed from the
    // measured pose so last_action / first command stay consistent with the policy.
    const bool stabilize_on = !this->IsShootingSim2Sim() &&
                              (this->params.Get<int>("shooting_deploy_stabilize", 0) != 0);
    std::vector<float> seed_q = state->motor_state.q;
    if (stabilize_on && this->params.Get<float>("shooting_policy_mix", 0.0f) <= 1e-3f)
    {
        if (this->params.Has("shooting_aim_anchor_dof_pos"))
        {
            seed_q = this->params.Get<std::vector<float>>("shooting_aim_anchor_dof_pos");
        }
        else if (this->params.Has("shooting_stand_dof_pos"))
        {
            seed_q = this->params.Get<std::vector<float>>("shooting_stand_dof_pos");
        }
    }

    for (int i = 0; i < num_dofs; ++i)
    {
        const float q_meas = (static_cast<int>(state->motor_state.q.size()) > i) ? state->motor_state.q[i] : default_pos[i];
        const float q_cmd = (static_cast<int>(seed_q.size()) > i) ? seed_q[i] : q_meas;
        const float scale = std::max(action_scale[i], 1e-6f);
        // Isaac reset last_action is 0. Pose-derived last_action makes the twist
        // actor emit saturated first actions and is only used for the deploy stack.
        if (this->IsShootingSim2Sim())
        {
            this->obs.actions[i] = 0.0f;
        }
        else
        {
            this->obs.actions[i] = (q_meas - default_pos[i]) / scale;
        }
        this->output_dof_pos[i] = q_cmd;
    }

    this->shooting_policy_blend_start_q = state->motor_state.q;
    this->shooting_policy_blend_percent = 0.0f;
    this->shooting_policy_blend_active = this->params.Get<float>("shooting_policy_blend_duration", 1.0f) > 1e-4f;

    // Clear stale command queues and publish a hold pose immediately.
    std::vector<float> unused;
    while (this->output_dof_pos_queue.try_pop(unused))
    {
    }
    while (this->output_dof_vel_queue.try_pop(unused))
    {
    }
    while (this->output_dof_tau_queue.try_pop(unused))
    {
    }
    this->output_dof_pos_queue.push(this->output_dof_pos);
    this->output_dof_vel_queue.push(this->output_dof_vel);

    const ShootingRayImpact ray = this->ComputeShootingRayImpact();
    const float seed_aim = this->ComputeShootingMuzzleAimError().empty()
                               ? 0.0f
                               : this->ComputeShootingMuzzleAimError()[0];
    std::cout << LOGGER::INFO
              << "Seeded policy from current pose; blend_duration="
              << this->params.Get<float>("shooting_policy_blend_duration", 1.0f) << "s"
                  << " last_action=" << (this->IsShootingSim2Sim() ? "0" : "from_pose")
                  << " aim_obs=" << seed_aim
                  << " (lateral " << ray.lateral_miss_m << "m / 0.5)"
                  << " lat=" << ray.lateral_miss_m
                  << " vert=" << ray.vertical_miss_m
                  << std::endl;
    this->DumpShootingPolicyDebugObs("seed");
}

void RL::DumpShootingPolicyDebugObs(const std::string &tag)
{
    if (!this->params.Get<bool>("shooting_policy_debug_obs", true))
    {
        return;
    }
    std::vector<float> obs = this->ComputeObservation();
    const std::string path = std::string(POLICY_DIR) + "/" + this->robot_name + "/shooting_obs_dump.csv";
    const bool write_header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
    std::ofstream file(path, std::ios_base::app);
    if (write_header)
    {
        file << "tag,t";
        for (size_t i = 0; i < obs.size(); ++i)
        {
            file << ",o" << i;
        }
        file << "\n";
    }
    const float t_s = static_cast<float>(this->motiontime) * this->params.Get<float>("dt", 0.005f);
    file << tag << "," << t_s;
    for (float v : obs)
    {
        file << "," << v;
    }
    file << "\n";
}

void RL::HandleShootingSimulationReset()
{
    this->ResetShootingControlState();
    this->SetSimulationPaused(true);
    this->rl_init_done = false;
    this->SetShootingActiveTarget(false);
    this->ApplyShootingRecoil();
    this->fsm.RequestStateChange("RLFSMStatePassive");
}

void RL::SetShootingActiveTarget(bool offset)
{
    this->shooting_active_target_is_offset = offset;
    this->obs.shooting_target_azimuth = offset
                                            ? this->shooting_offset_azimuth
                                            : this->shooting_frontal_azimuth;
    this->shooting_aim_pitch_trim = 0.0f;
    this->shooting_aim_yaw_trim = 0.0f;
    this->shooting_aim_pitch_cmd = 0.0f;
    this->shooting_aim_yaw_cmd = 0.0f;
    this->shooting_twist_valid = false;
    this->shooting_twist_blend = 0.0f;
    this->shooting_twist_need_solve = true;
}

std::string RL::ResolveShootingPolicyModelName(bool offset_target) const
{
    if (offset_target)
    {
        return this->params.Get<std::string>("shooting_offset_model", "policy_twist.pt");
    }
    return this->params.Get<std::string>("shooting_frontal_model", "policy_stab0614.pt");
}

void RL::ReloadShootingPolicyModel(const std::string &model_name)
{
    const std::string robot_config_path = this->robot_name + "/" + this->config_name;
    const std::string dir = std::string(POLICY_DIR) + "/" + robot_config_path + "/";
    std::string chosen = model_name.empty() ? "policy.pt" : model_name;
    std::string model_path = dir + chosen;
    if (!std::filesystem::exists(model_path))
    {
        std::cout << LOGGER::WARNING << "Shooting policy not found: " << model_path
                  << ", falling back to policy.pt" << std::endl;
        chosen = "policy.pt";
        model_path = dir + chosen;
    }
    if (this->model && this->shooting_loaded_model_name == chosen)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(this->model_mutex);
    this->model = InferenceRuntime::ModelFactory::load_model(model_path);
    if (!this->model)
    {
        throw std::runtime_error("Failed to load shooting policy from: " + model_path);
    }
    this->shooting_loaded_model_name = chosen;
    this->params.Set<std::string>("model_name", chosen);
    std::cout << LOGGER::INFO << "Loaded shooting policy: " << chosen << std::endl;
}

void RL::LoadShootingAimPolicy(bool offset_target)
{
    this->config_name = "robot_lab";
    const std::string robot_config_path = this->robot_name + "/" + this->config_name;
    if (!this->shooting_rl_yaml_loaded)
    {
        this->InitRL(robot_config_path);
        this->shooting_rl_yaml_loaded = true;
        this->shooting_frontal_azimuth = 0.0f;
        this->shooting_offset_azimuth = this->params.Get<float>("shooting_offset_target_azimuth", 0.0436332313f);
    }
    this->ReloadShootingPolicyModel(this->ResolveShootingPolicyModelName(offset_target));
    this->SetShootingActiveTarget(offset_target);
}

void RL::ComputeShootingRecoilWrench(float &fx, float &fy, float &fz) const
{
    fx = 0.0f;
    fy = 0.0f;
    fz = 0.0f;
    if (!this->shooting_stability_mode || !this->shooting_recoil_enabled || !this->rl_init_done)
    {
        return;
    }

    // Interval event time_left is resampled to 0.08 s on reset and the wrench
    // buffer is not cleared, so the link-frame force turns on at 0.08 s and stays.
    const float interval = std::max(this->shooting_recoil_interval, 1.0e-6f);
    if (this->shooting_policy_time_s + 1.0e-4f < interval)
    {
        return;
    }

    const char *force_key = this->shooting_active_target_is_offset
                                ? "shooting_offset_recoil_force"
                                : "shooting_frontal_recoil_force";
    const float configured = this->params.Has(force_key)
                                 ? this->params.Get<float>(force_key)
                                 : this->params.Get<float>("shooting_recoil_force", -36.0f);
    const float magnitude = std::fabs(configured);
    fx = -magnitude;
    fz = magnitude * 0.167f;
}

void RL::InitRL(std::string robot_config_path)
{
    std::lock_guard<std::mutex> lock(this->model_mutex);

    this->ReadYaml(robot_config_path, "config.yaml");
    // InitRL reloads robot_lab/config.yaml; re-apply the sim2sim overlay so
    // deploy mix/servo/clamp values cannot sneak back in.
    this->ApplyShootingSim2SimOverrides();

    // init joint num first
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));

    // init rl
    this->InitObservations();
    this->InitOutputs();
    this->InitControl();

    // init obs history
    const auto& observations_history = this->params.Get<std::vector<int>>("observations_history");  // avoid dangling reference
    if (!observations_history.empty())
    {
        int history_length = *std::max_element(observations_history.begin(), observations_history.end()) + 1;
        this->history_obs_buf = ObservationBuffer(1, this->obs_dims, history_length, this->params.Get<std::string>("observations_history_priority"));
    }

    // init model
    std::string model_path = std::string(POLICY_DIR) + "/" + robot_config_path + "/" + this->params.Get<std::string>("model_name");
    this->model = InferenceRuntime::ModelFactory::load_model(model_path);
    if (!this->model)
    {
        throw std::runtime_error("Failed to load model from: " + model_path);
    }
    this->shooting_loaded_model_name = this->params.Get<std::string>("model_name");
}

bool RL::IsShootingSim2Sim() const
{
    return this->shooting_stability_mode && this->shooting_sim2sim_mode;
}

bool RL::ComputeShootingStandOk(float pitch_deg, float base_z) const
{
    const auto &q = this->robot_state.motor_state.q;
    float hind_calf = 0.0f;
    if (q.size() >= 12)
    {
        hind_calf = std::max(q[10], q[11]);
    }
    return (base_z > 0.50f) && (std::fabs(pitch_deg) < 20.0f) && (hind_calf < 1.65f);
}

void RL::ApplyShootingSim2SimOverrides()
{
    if (!this->IsShootingSim2Sim())
    {
        return;
    }

    this->params.Set<int>("shooting_sim2sim", 1);
    this->params.Set<int>("shooting_deploy_stabilize", 0);
    this->params.Set<float>("shooting_policy_mix", 1.0f);
    this->params.Set<int>("shooting_aim_servo", 0);
    this->params.Set<int>("shooting_planted_twist", 0);
    this->params.Set<int>("shooting_aim_base_yaw_correct", 0);

    const int num_dofs = std::max(this->params.Get<int>("num_of_dofs", 12), 1);
    // Hips stay at Isaac 80/2 so saturated hip actions do not yaw the body.
    // Thigh and calf stiffness also stay at 80: 140/160 locked a -3 deg yaw
    // and a 5 m lateral miss. Extra damping on those joints shrinks the
    // vertical hunt without holding that offset pose.
    std::vector<float> rl_kp(static_cast<size_t>(num_dofs), 80.0f);
    std::vector<float> rl_kd(static_cast<size_t>(num_dofs), 2.0f);
    for (int i = 4; i < std::min(num_dofs, 12); ++i)
    {
        rl_kd[static_cast<size_t>(i)] = 8.0f;
    }
    this->params.Set<std::vector<float>>("rl_kp", rl_kp);
    this->params.Set<std::vector<float>>("rl_kd", rl_kd);
    // Match train_shooting_stability.py clip_actions=1.0 (not the deploy kneel clips).
    this->params.Set<std::vector<float>>("clip_actions_lower", std::vector<float>(static_cast<size_t>(num_dofs), -1.0f));
    this->params.Set<std::vector<float>>("clip_actions_upper", std::vector<float>(static_cast<size_t>(num_dofs), 1.0f));
    // Training applies 80/2 and the raw action from the reset pose. No pose/PD blend.
    this->params.Set<float>("shooting_policy_blend_duration", 0.0f);

    if (!this->shooting_sim2sim_logged)
    {
        this->shooting_sim2sim_logged = true;
        std::cout << LOGGER::NOTE
                  << "SIM2SIM: joint order matches Isaac (4 hips, 4 thighs, 4 calves). "
                  << "Hips track policy targets at PD 80/2. "
                  << "Thighs and calves use PD 80/8."
                  << std::endl;
    }
}

void RL::ApplyShootingDeployStabilize(std::vector<float> &dof_pos, bool update_aim_servo)
{
    // Prefer int flag (1/0). Do NOT call Get<bool> on the same key — yaml-cpp will
    // throw TypedBadConversion if the node is an integer.
    if (this->IsShootingSim2Sim())
    {
        return;
    }
    const bool enabled = this->params.Get<int>("shooting_deploy_stabilize", 0) != 0;
    if (!this->shooting_stability_mode || !enabled || dof_pos.empty())
    {
        return;
    }

    const int num_dofs = this->params.Get<int>("num_of_dofs");
    std::vector<float> anchor;
    if (this->params.Has("shooting_aim_anchor_dof_pos"))
    {
        anchor = this->params.Get<std::vector<float>>("shooting_aim_anchor_dof_pos");
    }
    else if (this->params.Has("shooting_stand_dof_pos"))
    {
        anchor = this->params.Get<std::vector<float>>("shooting_stand_dof_pos");
    }
    else
    {
        anchor = this->params.Get<std::vector<float>>("default_dof_pos");
    }

    // After [1]: closed-loop aim servo. Static anchor alone drifts and does not look like aiming.
    const bool aim_servo = this->shooting_aim_enabled &&
                           (this->params.Get<int>("shooting_aim_servo", 1) != 0);
    if (aim_servo && update_aim_servo && this->obs.muzzle_pos.size() >= 3 && this->obs.muzzle_quat.size() >= 4)
    {
        const ShootingRayImpact ray = this->ComputeShootingRayImpact();
        if (ray.valid)
        {
            const float dt = std::max(this->params.Get<float>("dt", 0.005f), 1e-6f);
            // Signed errors: +vert_err => need to raise impact (pitch muzzle up).
            const float vert_err = ray.target[2] - ray.impact[2];
            const float azimuth = this->obs.shooting_target_azimuth;
            const float lat_err =
                -(ray.delta[1] * std::cos(azimuth) - ray.delta[0] * std::sin(azimuth));

            const float pitch_kp = this->params.Get<float>("shooting_aim_pitch_kp", 0.012f);
            const float pitch_ki = this->params.Get<float>("shooting_aim_pitch_ki", 0.004f);
            const float yaw_kp = this->params.Get<float>("shooting_aim_yaw_kp", 0.010f);
            const float yaw_ki = this->params.Get<float>("shooting_aim_yaw_ki", 0.003f);
            const float pitch_lim = this->params.Get<float>("shooting_aim_pitch_trim_limit", 0.35f);
            const float yaw_lim = this->params.Get<float>("shooting_aim_yaw_trim_limit", 0.25f);

            this->shooting_aim_pitch_trim =
                std::clamp(this->shooting_aim_pitch_trim + pitch_ki * vert_err * dt, -pitch_lim, pitch_lim);
            this->shooting_aim_yaw_trim =
                std::clamp(this->shooting_aim_yaw_trim + yaw_ki * lat_err * dt, -yaw_lim, yaw_lim);

            this->shooting_aim_pitch_cmd =
                std::clamp(pitch_kp * vert_err + this->shooting_aim_pitch_trim, -pitch_lim, pitch_lim);
            this->shooting_aim_yaw_cmd =
                std::clamp(yaw_kp * lat_err + this->shooting_aim_yaw_trim, -yaw_lim, yaw_lim);

            const float pitch_cmd = this->shooting_aim_pitch_cmd;
            const float yaw_cmd = this->shooting_aim_yaw_cmd;

            // Pitch up: extend front thighs / flex rear thighs (Isaac order).
            if (anchor.size() >= 8)
            {
                anchor[4] += pitch_cmd;
                anchor[5] += pitch_cmd;
                anchor[6] -= pitch_cmd;
                anchor[7] -= pitch_cmd;
            }
            // Residual hip yaw only when not using planted-twist IK (az≈0 fine aim).
            if (anchor.size() >= 4 && !this->shooting_twist_valid)
            {
                anchor[0] += yaw_cmd;
                anchor[1] -= yaw_cmd;
                anchor[2] += yaw_cmd;
                anchor[3] -= yaw_cmd;
            }
        }
    }
    else if (aim_servo)
    {
        // Re-apply last P+I command without integrating (control-thread path).
        const float pitch_cmd = this->shooting_aim_pitch_cmd;
        const float yaw_cmd = this->shooting_aim_yaw_cmd;
        if (anchor.size() >= 8)
        {
            anchor[4] += pitch_cmd;
            anchor[5] += pitch_cmd;
            anchor[6] -= pitch_cmd;
            anchor[7] -= pitch_cmd;
        }
        if (anchor.size() >= 4 && !this->shooting_twist_valid)
        {
            anchor[0] += yaw_cmd;
            anchor[1] -= yaw_cmd;
            anchor[2] += yaw_cmd;
            anchor[3] -= yaw_cmd;
        }
    }
    else if (!this->shooting_aim_enabled)
    {
        this->shooting_aim_pitch_trim = 0.0f;
        this->shooting_aim_yaw_trim = 0.0f;
    }

    if (this->shooting_twist_valid &&
        this->shooting_twist_q.size() >= static_cast<size_t>(num_dofs) &&
        anchor.size() >= static_cast<size_t>(num_dofs))
    {
        const float b = std::clamp(this->shooting_twist_blend, 0.0f, 1.0f);
        std::vector<float> twisted = this->shooting_twist_q;
        if (twisted.size() >= 8)
        {
            const float pitch_cmd = this->shooting_aim_pitch_cmd;
            twisted[4] += pitch_cmd;
            twisted[5] += pitch_cmd;
            twisted[6] -= pitch_cmd;
            twisted[7] -= pitch_cmd;
        }
        for (int i = 0; i < num_dofs; ++i)
        {
            anchor[i] = (1.0f - b) * anchor[i] + b * twisted[i];
        }
    }

    float mix = this->params.Get<float>("shooting_policy_mix", 0.0f);
    mix = std::clamp(mix, 0.0f, 1.0f);

    for (int i = 0; i < num_dofs && i < static_cast<int>(dof_pos.size()) && i < static_cast<int>(anchor.size()); ++i)
    {
        dof_pos[i] = (1.0f - mix) * anchor[i] + mix * dof_pos[i];
    }

    if (this->params.Has("shooting_leg_thigh_clamp") && num_dofs >= 8)
    {
        const auto thigh_clamp = this->params.Get<std::vector<float>>("shooting_leg_thigh_clamp");
        if (thigh_clamp.size() >= 2)
        {
            for (int i = 4; i < 8 && i < static_cast<int>(dof_pos.size()); ++i)
            {
                dof_pos[i] = std::clamp(dof_pos[i], thigh_clamp[0], thigh_clamp[1]);
            }
        }
    }
    if (this->params.Has("shooting_leg_calf_clamp") && num_dofs >= 12)
    {
        const auto calf_clamp = this->params.Get<std::vector<float>>("shooting_leg_calf_clamp");
        if (calf_clamp.size() >= 2)
        {
            for (int i = 8; i < 12 && i < static_cast<int>(dof_pos.size()); ++i)
            {
                dof_pos[i] = std::clamp(dof_pos[i], calf_clamp[0], calf_clamp[1]);
            }
        }
    }

    if (!this->shooting_stabilize_logged)
    {
        this->shooting_stabilize_logged = true;
        std::cout << LOGGER::INFO << "Deploy stabilize ON mix=" << mix
                  << " aim_servo=" << (this->params.Get<int>("shooting_aim_servo", 1) != 0 ? "ON" : "OFF")
                  << " (press [1] starts closed-loop aim at target)" << std::endl;
    }
}

void RL::ComputeOutput(const std::vector<float> &actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau)
{
    std::vector<float> actions_scaled = actions * this->params.Get<std::vector<float>>("action_scale");
    std::vector<float> pos_actions_scaled = actions_scaled;
    std::vector<float> vel_actions_scaled(actions.size(), 0.0f);
    for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
    {
        pos_actions_scaled[i] = 0.0f;
        vel_actions_scaled[i] = actions_scaled[i];
    }
    std::vector<float> all_actions_scaled = pos_actions_scaled + vel_actions_scaled;
    output_dof_pos = pos_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos");
    // Apply before queue push so RLControl always receives stabilized targets.
    this->ApplyShootingDeployStabilize(output_dof_pos);
    output_dof_vel = vel_actions_scaled;
    output_dof_tau = this->params.Get<std::vector<float>>("rl_kp") * (output_dof_pos - this->obs.dof_pos) - this->params.Get<std::vector<float>>("rl_kd") * this->obs.dof_vel;
    output_dof_tau = clamp(output_dof_tau, -this->params.Get<std::vector<float>>("torque_limits"), this->params.Get<std::vector<float>>("torque_limits"));
}

int RL::InverseJointMapping(int idx) const
{
    auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
    for (size_t i = 0; i < joint_mapping.size(); ++i) {
        if (joint_mapping[i] == idx) return (int)i;
    }
    return -1;
}

void RL::TorqueProtect(const std::vector<float>& origin_output_dof_tau)
{
    std::vector<int> out_of_range_indices;
    std::vector<float> out_of_range_values;
    for (size_t i = 0; i < origin_output_dof_tau.size(); ++i)
    {
        float torque_value = origin_output_dof_tau[i];
        float limit_lower = -this->params.Get<std::vector<float>>("torque_limits")[i];
        float limit_upper = this->params.Get<std::vector<float>>("torque_limits")[i];

        if (torque_value < limit_lower || torque_value > limit_upper)
        {
            out_of_range_indices.push_back(i);
            out_of_range_values.push_back(torque_value);
        }
    }
    if (!out_of_range_indices.empty())
    {
        for (size_t i = 0; i < out_of_range_indices.size(); ++i)
        {
            int index = out_of_range_indices[i];
            float value = out_of_range_values[i];
            float limit_lower = -this->params.Get<std::vector<float>>("torque_limits")[index];
            float limit_upper = this->params.Get<std::vector<float>>("torque_limits")[index];

            std::cout << LOGGER::WARNING << "Torque(" << index + 1 << ")=" << value << " out of range(" << limit_lower << ", " << limit_upper << ")" << std::endl;
        }
        // Just a reminder, no protection
        // this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::INFO << "Switching to STATE_POS_GETDOWN"<< std::endl;
    }
}

void RL::AttitudeProtect(const std::vector<float> &quaternion, float pitch_threshold, float roll_threshold)
{
    // Use QuaternionToEuler from vector_math.hpp
    std::vector<float> euler = QuaternionToEuler(quaternion);
    float roll = euler[0] * 57.2958f;   // Convert to degrees
    float pitch = euler[1] * 57.2958f;

    if (std::fabs(roll) > roll_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Roll exceeds " << roll_threshold << " degrees. Current: " << roll << " degrees." << std::endl;
    }
    if (std::fabs(pitch) > pitch_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Pitch exceeds " << pitch_threshold << " degrees. Current: " << pitch << " degrees." << std::endl;
    }
}

#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

static int kbhit()
{
    static bool initialized = false;
    static termios original_term;

    // Initialize terminal to non-canonical mode on first call
    if (!initialized)
    {
        tcgetattr(STDIN_FILENO, &original_term);

        termios new_term = original_term;
        new_term.c_lflag &= ~(ICANON | ECHO);  // Disable canonical mode and echo
        new_term.c_cc[VMIN] = 0;   // Non-blocking read
        new_term.c_cc[VTIME] = 0;  // No timeout

        tcsetattr(STDIN_FILENO, TCSANOW, &new_term);

        // Register cleanup function to restore terminal on exit
        static bool cleanup_registered = false;
        if (!cleanup_registered)
        {
            std::atexit([]() {
                tcsetattr(STDIN_FILENO, TCSANOW, &original_term);
            });
            cleanup_registered = true;
        }

        initialized = true;
    }

    // Non-blocking read of a single character
    char c;
    int result = read(STDIN_FILENO, &c, 1);

    return (result == 1) ? (unsigned char)c : -1;
}

void RL::KeyboardInterface()
{
    int c = kbhit();
    if (c > 0)
    {
        switch (c)
        {
        case '0': this->control.SetKeyboard(Input::Keyboard::Num0); break;
        case '1': this->control.SetKeyboard(Input::Keyboard::Num1); break;
        case '2': this->control.SetKeyboard(Input::Keyboard::Num2); break;
        case '3': this->control.SetKeyboard(Input::Keyboard::Num3); break;
        case '4': this->control.SetKeyboard(Input::Keyboard::Num4); break;
        case '5': this->control.SetKeyboard(Input::Keyboard::Num5); break;
        case '6': this->control.SetKeyboard(Input::Keyboard::Num6); break;
        case '7': this->control.SetKeyboard(Input::Keyboard::Num7); break;
        case '8': this->control.SetKeyboard(Input::Keyboard::Num8); break;
        case '9': this->control.SetKeyboard(Input::Keyboard::Num9); break;
        case 'a': case 'A': this->control.SetKeyboard(Input::Keyboard::A); break;
        case 'b': case 'B': this->control.SetKeyboard(Input::Keyboard::B); break;
        case 'c': case 'C': this->control.SetKeyboard(Input::Keyboard::C); break;
        case 'd': case 'D': this->control.SetKeyboard(Input::Keyboard::D); break;
        case 'e': case 'E': this->control.SetKeyboard(Input::Keyboard::E); break;
        case 'f': case 'F': this->control.SetKeyboard(Input::Keyboard::F); break;
        case 'g': case 'G': this->control.SetKeyboard(Input::Keyboard::G); break;
        case 'h': case 'H': this->control.SetKeyboard(Input::Keyboard::H); break;
        case 'i': case 'I': this->control.SetKeyboard(Input::Keyboard::I); break;
        case 'j': case 'J': this->control.SetKeyboard(Input::Keyboard::J); break;
        case 'k': case 'K': this->control.SetKeyboard(Input::Keyboard::K); break;
        case 'l': case 'L': this->control.SetKeyboard(Input::Keyboard::L); break;
        case 'm': case 'M': this->control.SetKeyboard(Input::Keyboard::M); break;
        case 'n': case 'N': this->control.SetKeyboard(Input::Keyboard::N); break;
        case 'o': case 'O': this->control.SetKeyboard(Input::Keyboard::O); break;
        case 'p': case 'P': this->control.SetKeyboard(Input::Keyboard::P); break;
        case 'q': case 'Q': this->control.SetKeyboard(Input::Keyboard::Q); break;
        case 'r': case 'R': this->control.SetKeyboard(Input::Keyboard::R); break;
        case 's': case 'S': this->control.SetKeyboard(Input::Keyboard::S); break;
        case 't': case 'T': this->control.SetKeyboard(Input::Keyboard::T); break;
        case 'u': case 'U': this->control.SetKeyboard(Input::Keyboard::U); break;
        case 'v': case 'V': this->control.SetKeyboard(Input::Keyboard::V); break;
        case 'w': case 'W': this->control.SetKeyboard(Input::Keyboard::W); break;
        case 'x': case 'X': this->control.SetKeyboard(Input::Keyboard::X); break;
        case 'y': case 'Y': this->control.SetKeyboard(Input::Keyboard::Y); break;
        case 'z': case 'Z': this->control.SetKeyboard(Input::Keyboard::Z); break;
        case ' ': this->control.SetKeyboard(Input::Keyboard::Space); break;
        case '\n': case '\r': this->control.SetKeyboard(Input::Keyboard::Enter); break;
        case 27:  // Escape sequence (for arrow keys on Unix/Linux/macOS)
        {
            char seq[2];
            // Try to read escape sequence non-blockingly
            if (read(STDIN_FILENO, &seq[0], 1) == 1)
            {
                if (seq[0] == '[')
                {
                    if (read(STDIN_FILENO, &seq[1], 1) == 1)
                    {
                        switch (seq[1])
                        {
                        case 'A': this->control.SetKeyboard(Input::Keyboard::Up); break;
                        case 'B': this->control.SetKeyboard(Input::Keyboard::Down); break;
                        case 'C': this->control.SetKeyboard(Input::Keyboard::Right); break;
                        case 'D': this->control.SetKeyboard(Input::Keyboard::Left); break;
                        default: break;
                        }
                    }
                }
                else
                {
                    // Plain escape key
                    this->control.SetKeyboard(Input::Keyboard::Escape);
                }
            }
            else
            {
                // Plain escape key
                this->control.SetKeyboard(Input::Keyboard::Escape);
            }
        } break;
        default:  break;
        }
        if (this->shooting_stability_mode &&
            (c == '0' || c == '1' || c == '2' || c == '3' || c == 'r' || c == 'R'))
        {
            std::cout << LOGGER::INFO << "Key received: " << static_cast<char>(c) << std::endl;
        }
    }
}

template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }
    return values;
}

void RL::ReadYaml(const std::string& file_path, const std::string& file_name)
{
    std::string config_path = std::string(POLICY_DIR) + "/" + file_path + "/" + file_name;
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path)[file_path];
    }
    catch (YAML::BadFile &e)
    {
        std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }

    for (auto it = config.begin(); it != config.end(); ++it)
    {
        std::string key = it->first.as<std::string>();
        this->params.config_node[key] = it->second;
    }
}

void RL::CSVInit(std::string robot_path)
{
    csv_filename = std::string(POLICY_DIR) + "/" + robot_path + "/motor";

    // Uncomment these lines if need timestamp for file name
    // auto now = std::chrono::system_clock::now();
    // std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    // std::stringstream ss;
    // ss << std::put_time(std::localtime(&now_c), "%Y%m%d%H%M%S");
    // std::string timestamp = ss.str();
    // csv_filename += "_" + timestamp;

    csv_filename += ".csv";
    std::ofstream file(csv_filename.c_str());

    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "tau_cal_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "tau_est_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_pos_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_pos_target_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_vel_" << i << ","; }

    file << std::endl;

    file.close();
}

void RL::CSVLogger(const std::vector<float>& torque, const std::vector<float>& tau_est, const std::vector<float>& joint_pos, const std::vector<float>& joint_pos_target, const std::vector<float>& joint_vel)
{
    std::ofstream file(csv_filename.c_str(), std::ios_base::app);

    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << torque[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << tau_est[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_pos[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_pos_target[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_vel[i] << ","; }

    file << std::endl;

    file.close();
}

bool RLFSMState::Interpolate(
    float& percent,
    const std::vector<float>& start_pos,
    const std::vector<float>& target_pos,
    float duration_seconds,
    const std::string& description,
    bool use_fixed_gains)
{
    if (percent >= 1.0f)
    {
        return false;
    }

    if (percent == 0.0f)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < start_pos.size() && i < target_pos.size(); ++i)
        {
            max_diff = std::max(max_diff, std::abs(start_pos[i] - target_pos[i]));
        }

        if (max_diff < 0.1f)
        {
            percent = 1.0f;
        }
    }

    int required_frames = std::max(1, static_cast<int>(std::ceil(duration_seconds / rl.params.Get<float>("dt"))));
    float step = 1.0f / required_frames;

    percent += step;
    percent = std::min(percent, 1.0f);

    auto kp = use_fixed_gains ? rl.params.Get<std::vector<float>>("fixed_kp") : rl.params.Get<std::vector<float>>("rl_kp");
    auto kd = use_fixed_gains ? rl.params.Get<std::vector<float>>("fixed_kd") : rl.params.Get<std::vector<float>>("rl_kd");

    for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
    {
        fsm_command->motor_command.q[i] = (1 - percent) * start_pos[i] + percent * target_pos[i];
        fsm_command->motor_command.dq[i] = 0;
        fsm_command->motor_command.kp[i] = kp[i];
        fsm_command->motor_command.kd[i] = kd[i];
        fsm_command->motor_command.tau[i] = 0;
    }

    if (!description.empty())
    {
        LOGGER::PrintProgress(percent, description);
    }

    if (percent >= 1.0f)
    {
        return false;
    }

    return true;
}

void RLFSMState::RLControl()
{
    std::vector<float> _output_dof_pos, _output_dof_vel;
    if (rl.output_dof_pos_queue.try_pop(_output_dof_pos) && rl.output_dof_vel_queue.try_pop(_output_dof_vel))
    {
        if (rl.shooting_stability_mode && rl.shooting_policy_blend_active &&
            !rl.shooting_policy_blend_start_q.empty() && !_output_dof_pos.empty())
        {
            const float duration = std::max(rl.params.Get<float>("shooting_policy_blend_duration", 1.0f), 1e-4f);
            const float dt = std::max(rl.params.Get<float>("dt", 0.005f), 1e-6f);
            rl.shooting_policy_blend_percent =
                std::min(1.0f, rl.shooting_policy_blend_percent + dt / duration);
            const float a = rl.shooting_policy_blend_percent;
            for (size_t i = 0; i < _output_dof_pos.size() && i < rl.shooting_policy_blend_start_q.size(); ++i)
            {
                _output_dof_pos[i] =
                    (1.0f - a) * rl.shooting_policy_blend_start_q[i] + a * _output_dof_pos[i];
            }
            if (a >= 1.0f)
            {
                rl.shooting_policy_blend_active = false;
            }
        }

        // Re-apply anchor+trim without double-integrating the aim servo.
        // Sim2sim skips this: targets stay the raw policy joint positions.
        if (!rl.IsShootingSim2Sim())
        {
            rl.ApplyShootingDeployStabilize(_output_dof_pos, /*update_aim_servo=*/false);
        }

        // Deploy keeps stand-hold gains on an anchor pose.
        // Sim2sim tracks the policy joint target with the per-joint gains from
        // ApplyShootingSim2SimOverrides: hips 80/2, thighs and calves 80/8.
        const bool stabilize_on = !rl.IsShootingSim2Sim() &&
                                  rl.shooting_stability_mode &&
                                  (rl.params.Get<int>("shooting_deploy_stabilize", 0) != 0);
        const auto rl_kp = rl.params.Get<std::vector<float>>("rl_kp");
        const auto rl_kd = rl.params.Get<std::vector<float>>("rl_kd");
        const bool use_hold = stabilize_on && rl.params.Has("stand_hold_kp");
        auto kp_vec = use_hold ? rl.params.Get<std::vector<float>>("stand_hold_kp") : rl_kp;
        auto kd_vec = use_hold && rl.params.Has("stand_hold_kd")
                          ? rl.params.Get<std::vector<float>>("stand_hold_kd")
                          : rl_kd;

        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            if (!_output_dof_pos.empty())
            {
                fsm_command->motor_command.q[i] = _output_dof_pos[i];
            }
            if (!_output_dof_vel.empty())
            {
                fsm_command->motor_command.dq[i] = _output_dof_vel[i];
            }
            fsm_command->motor_command.kp[i] = kp_vec[i];
            fsm_command->motor_command.kd[i] = kd_vec[i];
            fsm_command->motor_command.tau[i] = 0;
        }
    }
}
