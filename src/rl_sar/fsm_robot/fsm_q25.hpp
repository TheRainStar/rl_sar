/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef Q25_FSM_HPP
#define Q25_FSM_HPP

#include "fsm.hpp"
#include "rl_sdk.hpp"

#include <algorithm>
#include <cmath>
#include <exception>

namespace q25_fsm
{

namespace
{

void HoldJointPositions(
    RLFSMState &state,
    const std::vector<float> &target_pos,
    const std::vector<float> &kp,
    const std::vector<float> &kd)
{
    for (int i = 0; i < state.rl.params.Get<int>("num_of_dofs"); ++i)
    {
        state.fsm_command->motor_command.q[i] = target_pos[i];
        state.fsm_command->motor_command.dq[i] = 0;
        state.fsm_command->motor_command.kp[i] = kp[i];
        state.fsm_command->motor_command.kd[i] = kd[i];
        state.fsm_command->motor_command.tau[i] = 0;
    }
}

bool BodyAttitudeWithinLimits(
    const RobotState<float> *state,
    float max_roll_pitch_deg,
    float max_gyro_deg_s)
{
    if (state == nullptr || state->imu.quaternion.size() < 4)
    {
        return false;
    }

    const std::vector<float> euler = QuaternionToEuler(state->imu.quaternion);
    const float roll_deg = std::fabs(euler[0] * 57.2958f);
    const float pitch_deg = std::fabs(euler[1] * 57.2958f);

    float gyro_mag_deg_s = 0.0f;
    if (state->imu.gyroscope.size() >= 3)
    {
        const float wx = state->imu.gyroscope[0];
        const float wy = state->imu.gyroscope[1];
        const float wz = state->imu.gyroscope[2];
        gyro_mag_deg_s = std::sqrt(wx * wx + wy * wy + wz * wz) * 57.2958f;
    }

    return roll_deg <= max_roll_pitch_deg && pitch_deg <= max_roll_pitch_deg && gyro_mag_deg_s <= max_gyro_deg_s;
}

std::vector<float> GetPreGetUpPos(const RL &rl)
{
    if (rl.params.Has("pre_getup_dof_pos"))
    {
        return rl.params.Get<std::vector<float>>("pre_getup_dof_pos");
    }
    return {
        0.0f, -0.95f, 1.85f,
        0.0f, -0.95f, 1.85f,
        0.0f, -0.95f, 1.85f,
        0.0f, -0.95f, 1.85f,
    };
}

std::vector<float> GetStandHoldPos(const RL &rl)
{
    if (rl.shooting_stability_mode && rl.params.Has("shooting_stand_dof_pos"))
    {
        return rl.params.Get<std::vector<float>>("shooting_stand_dof_pos");
    }
    return rl.params.Get<std::vector<float>>("default_dof_pos");
}

std::vector<float> GetStandHoldKp(const RL &rl)
{
    if (rl.shooting_stability_mode && rl.params.Has("stand_hold_kp"))
    {
        return rl.params.Get<std::vector<float>>("stand_hold_kp");
    }
    if (rl.params.Has("getup_kp"))
    {
        return rl.params.Get<std::vector<float>>("getup_kp");
    }
    return rl.params.Get<std::vector<float>>("fixed_kp");
}

std::vector<float> GetStandHoldKd(const RL &rl)
{
    if (rl.shooting_stability_mode && rl.params.Has("stand_hold_kd"))
    {
        return rl.params.Get<std::vector<float>>("stand_hold_kd");
    }
    if (rl.params.Has("getup_kd"))
    {
        return rl.params.Get<std::vector<float>>("getup_kd");
    }
    return rl.params.Get<std::vector<float>>("fixed_kd");
}

bool JointsNearTarget(
    const RobotState<float> *state,
    const std::vector<float> &target_pos,
    float tolerance)
{
    if (state == nullptr || state->motor_state.q.size() < target_pos.size())
    {
        return false;
    }
    for (size_t i = 0; i < target_pos.size(); ++i)
    {
        if (std::fabs(state->motor_state.q[i] - target_pos[i]) > tolerance)
        {
            return false;
        }
    }
    return true;
}

bool InterpolateWithGains(
    RLFSMState &state,
    float &percent,
    const std::vector<float> &start_pos,
    const std::vector<float> &target_pos,
    float duration_seconds,
    const std::vector<float> &kp,
    const std::vector<float> &kd,
    const std::string &description)
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

    const float dt = state.rl.params.Get<float>("dt");
    const int required_frames = std::max(1, static_cast<int>(std::ceil(duration_seconds / dt)));
    const float step = 1.0f / required_frames;

    percent += step;
    percent = std::min(percent, 1.0f);

    for (int i = 0; i < state.rl.params.Get<int>("num_of_dofs"); ++i)
    {
        state.fsm_command->motor_command.q[i] = (1 - percent) * start_pos[i] + percent * target_pos[i];
        state.fsm_command->motor_command.dq[i] = 0;
        state.fsm_command->motor_command.kp[i] = kp[i];
        state.fsm_command->motor_command.kd[i] = kd[i];
        state.fsm_command->motor_command.tau[i] = 0;
    }

    if (!description.empty())
    {
        LOGGER::PrintProgress(percent, description);
    }

    return percent < 1.0f;
}

bool SettleAttitudeWithinLimits(
    const RL &rl,
    const RobotState<float> *state,
    float max_roll_deg,
    float max_gyro_deg_s)
{
    if (state == nullptr || state->imu.quaternion.size() < 4)
    {
        return false;
    }

    const std::vector<float> euler = QuaternionToEuler(state->imu.quaternion);
    const float roll_deg = std::fabs(euler[0] * 57.2958f);
    const float pitch_deg = std::fabs(euler[1] * 57.2958f);

    float gyro_mag_deg_s = 0.0f;
    if (state->imu.gyroscope.size() >= 3)
    {
        const float wx = state->imu.gyroscope[0];
        const float wy = state->imu.gyroscope[1];
        const float wz = state->imu.gyroscope[2];
        gyro_mag_deg_s = std::sqrt(wx * wx + wy * wy + wz * wz) * 57.2958f;
    }

    if (rl.shooting_stability_mode)
    {
        const float max_pitch_deg = rl.params.Get<float>("shooting_stand_attitude_max_pitch_deg", 25.0f);
        return roll_deg <= max_roll_deg && pitch_deg <= max_pitch_deg && gyro_mag_deg_s <= max_gyro_deg_s;
    }

    return roll_deg <= max_roll_deg && pitch_deg <= max_roll_deg && gyro_mag_deg_s <= max_gyro_deg_s;
}

void LogJointSettleErrors(
    const RL &rl,
    const RobotState<float> *state,
    const std::vector<float> &target_pos,
    float tolerance)
{
    if (state == nullptr)
    {
        return;
    }

    const auto joint_names = rl.params.Get<std::vector<std::string>>("joint_names");
    for (size_t i = 0; i < target_pos.size() && i < state->motor_state.q.size(); ++i)
    {
        const float err = state->motor_state.q[i] - target_pos[i];
        if (std::fabs(err) > tolerance)
        {
            const std::string name = i < joint_names.size() ? joint_names[i] : "joint";
            std::cout << LOGGER::WARNING << "  " << name << ": actual="
                      << state->motor_state.q[i] << " target=" << target_pos[i]
                      << " err=" << err << std::endl;
        }
    }
}

} // namespace

class RLFSMStatePassive : public RLFSMState
{
public:
    RLFSMStatePassive(RL *rl) : RLFSMState(*rl, "RLFSMStatePassive") {}

    void Enter() override
    {
        if (rl.shooting_stability_mode)
        {
            const bool from_getup = rl.fsm.previous_state_ &&
                                    rl.fsm.previous_state_->GetStateName() == "RLFSMStateGetUp";
            if (!from_getup)
            {
                rl.SetSimulationPaused(true);
            }
            std::cout << LOGGER::NOTE
                      << "Passive. [0] stand | [1] frontal/hold (red) | [3] offset/twist (blue) | [2] recoil | [R] reset"
                      << std::endl;
        }
        else
        {
            std::cout << LOGGER::NOTE << "Entered passive mode. Press '0' to switch to RLFSMStateGetUp." << std::endl;
        }
    }

    void Run() override
    {
        if (rl.shooting_stability_mode)
        {
            const auto passive_kp = rl.params.Has("passive_kp")
                                        ? rl.params.Get<std::vector<float>>("passive_kp")
                                        : rl.params.Get<std::vector<float>>("fixed_kp");
            const auto passive_kd = rl.params.Get<std::vector<float>>("fixed_kd");
            HoldJointPositions(*this, GetPreGetUpPos(rl), passive_kp, passive_kd);
            return;
        }

        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            fsm_command->motor_command.dq[i] = 0;
            fsm_command->motor_command.kp[i] = 0;
            fsm_command->motor_command.kd[i] = 8;
            fsm_command->motor_command.tau[i] = 0;
        }
    }

    void Exit() override
    {
        if (rl.shooting_stability_mode &&
            (rl.control.current_keyboard == Input::Keyboard::Num0 ||
             rl.control.current_keyboard == Input::Keyboard::Num1 ||
             rl.control.current_keyboard == Input::Keyboard::Num2 ||
             rl.control.current_keyboard == Input::Keyboard::Num3))
        {
            rl.control.ClearInput();
        }
    }

    std::string CheckChange() override
    {
        if (!rl.shooting_scene_ready)
        {
            return state_name_;
        }
        if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        if (rl.shooting_stability_mode &&
            (rl.control.current_keyboard == Input::Keyboard::Num1 ||
             rl.control.current_keyboard == Input::Keyboard::Num3 ||
             rl.control.current_gamepad == Input::Gamepad::RB_DPadUp ||
             rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft))
        {
            std::cout << LOGGER::WARNING
                      << "Press [0] first and wait for standing hold, then [1] (frontal/hold) or [3] (offset/twist). "
                      << "Policy handoff from Passive/crouch kneels immediately."
                      << std::endl;
            rl.control.ClearInput();
            return state_name_;
        }
        if (rl.shooting_stability_mode &&
            (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadDown))
        {
            std::cout << LOGGER::NOTE << "Press [1] or [3] first to load a shooting policy." << std::endl;
            rl.control.ClearInput();
        }
        return state_name_;
    }
};

class RLFSMStateGetUp : public RLFSMState
{
public:
    RLFSMStateGetUp(RL *rl) : RLFSMState(*rl, "RLFSMStateGetUp") {}

    float percent_pre_getup = 0.0f;
    float percent_getup = 0.0f;
    float settle_elapsed = 0.0f;
    int settle_stable_frames = 0;
    bool stand_from_passive = true;
    bool getup_finished = false;
    bool stand_succeeded = false;
    bool in_settle_phase = false;
    std::vector<float> pre_getup_pos;

    void Enter() override
    {
        percent_pre_getup = 0.0f;
        percent_getup = 0.0f;
        settle_elapsed = 0.0f;
        settle_stable_frames = 0;
        getup_finished = false;
        stand_succeeded = false;
        in_settle_phase = false;
        pre_getup_pos = GetPreGetUpPos(rl);
        stand_from_passive = rl.fsm.previous_state_->GetStateName() == "RLFSMStatePassive";
        rl.now_state = *fsm_state;
        rl.start_state = rl.now_state;
        if (stand_from_passive && JointsNearTarget(&rl.now_state, pre_getup_pos, 0.15f))
        {
            percent_pre_getup = 1.0f;
        }
        if (rl.shooting_stability_mode)
        {
            rl.ResetShootingControlState();
            rl.EnsureShootingSimulationRunning();
            if (rl.control.current_keyboard == Input::Keyboard::Num0)
            {
                rl.control.ClearInput();
            }
            std::cout << LOGGER::INFO << "Simulation started. Standing up (no aim, no recoil)..." << std::endl;
        }
    }

    void Run() override
    {
        const auto stand_pos = GetStandHoldPos(rl);
        const auto getup_kp = rl.params.Has("getup_kp") ? rl.params.Get<std::vector<float>>("getup_kp") : rl.params.Get<std::vector<float>>("fixed_kp");
        const auto getup_kd = rl.params.Has("getup_kd") ? rl.params.Get<std::vector<float>>("getup_kd") : rl.params.Get<std::vector<float>>("fixed_kd");
        const auto hold_kp = GetStandHoldKp(rl);
        const auto hold_kd = GetStandHoldKd(rl);
        const float pre_duration = rl.params.Get<float>("getup_pre_duration", 0.8f);
        const float extend_duration = stand_from_passive ? rl.params.Get<float>("getup_duration", 3.5f) : 1.0f;

        if (stand_from_passive)
        {
            if (!rl.shooting_stability_mode &&
                InterpolateWithGains(*this, percent_pre_getup, rl.now_state.motor_state.q, pre_getup_pos, pre_duration, getup_kp, getup_kd, "Pre Getting up"))
            {
                return;
            }
            if (rl.shooting_stability_mode)
            {
                percent_pre_getup = 1.0f;
            }
            if (InterpolateWithGains(*this, percent_getup, pre_getup_pos, stand_pos, extend_duration, getup_kp, getup_kd, "Getting up"))
            {
                return;
            }
        }
        else if (InterpolateWithGains(*this, percent_getup, rl.now_state.motor_state.q, stand_pos, extend_duration, getup_kp, getup_kd, "Getting up"))
        {
            return;
        }

        if (rl.shooting_stability_mode)
        {
            HoldJointPositions(*this, stand_pos, hold_kp, hold_kd);
            if (!getup_finished)
            {
                const std::vector<float> euler = QuaternionToEuler(fsm_state->imu.quaternion);
                const float joint_tolerance = rl.params.Get<float>("getup_joint_tolerance", 0.15f);
                const bool joints_ok = JointsNearTarget(fsm_state, stand_pos, joint_tolerance);
                std::cout << std::endl
                          << LOGGER::INFO << "Shooting stand hold (open-loop). roll="
                          << euler[0] * 57.2958f << " deg, pitch="
                          << euler[1] * 57.2958f << " deg"
                          << (joints_ok ? "" : " — press [1] frontal/hold or [3] offset/twist")
                          << std::endl;
                if (!joints_ok)
                {
                    LogJointSettleErrors(rl, fsm_state, stand_pos, joint_tolerance);
                }
            }
            stand_succeeded = true;
            getup_finished = true;
            return;
        }

        in_settle_phase = true;
        HoldJointPositions(*this, stand_pos, hold_kp, hold_kd);

        const float dt = rl.params.Get<float>("dt");
        const float max_roll_deg = rl.params.Get<float>("stand_attitude_max_deg", 12.0f);
        const float max_gyro_deg_s = rl.params.Get<float>("stand_gyro_max_deg_s", 25.0f);
        const float settle_max_duration = rl.params.Get<float>("getup_settle_max_duration", 8.0f);
        const float settle_stable_duration = rl.params.Get<float>("getup_settle_stable_duration", 0.5f);
        const float joint_tolerance = rl.params.Get<float>("getup_joint_tolerance", 0.15f);
        const int settle_stable_frames_required = std::max(1, static_cast<int>(std::ceil(settle_stable_duration / dt)));

        settle_elapsed += dt;
        const bool attitude_ok = SettleAttitudeWithinLimits(rl, fsm_state, max_roll_deg, max_gyro_deg_s);
        const bool joints_ok = JointsNearTarget(fsm_state, stand_pos, joint_tolerance);
        if (attitude_ok && joints_ok)
        {
            ++settle_stable_frames;
        }
        else
        {
            settle_stable_frames = 0;
        }

        if (!getup_finished)
        {
            LOGGER::PrintProgress(
                std::min(1.0f, settle_elapsed / settle_max_duration),
                "Stabilizing pose");
        }

        if (settle_stable_frames >= settle_stable_frames_required)
        {
            if (!getup_finished)
            {
                const std::vector<float> euler = QuaternionToEuler(fsm_state->imu.quaternion);
                std::cout << std::endl
                          << LOGGER::INFO << "Stand up complete. roll="
                          << euler[0] * 57.2958f << " deg, pitch="
                          << euler[1] * 57.2958f << " deg" << std::endl;
            }
            stand_succeeded = true;
            getup_finished = true;
        }
        else if (settle_elapsed >= settle_max_duration)
        {
            if (!getup_finished)
            {
                const std::vector<float> euler = QuaternionToEuler(fsm_state->imu.quaternion);
                std::cout << std::endl
                          << LOGGER::WARNING << "Stand up settle timeout. roll="
                          << euler[0] * 57.2958f << " deg, pitch="
                          << euler[1] * 57.2958f << " deg, joints_ok="
                          << (joints_ok ? "yes" : "no") << std::endl;
                if (!joints_ok)
                {
                    LogJointSettleErrors(rl, fsm_state, stand_pos, joint_tolerance);
                }
            }
            stand_succeeded = true;
            getup_finished = true;
        }
    }

    void Exit() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num0)
        {
            rl.control.ClearInput();
        }
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        if (!getup_finished)
        {
            return state_name_;
        }
        if (rl.shooting_stability_mode)
        {
            return "RLFSMStateShootStandHold";
        }
        if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLLocomotion";
        }
        if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        return state_name_;
    }
};

class RLFSMStateShootStandHold : public RLFSMState
{
public:
    RLFSMStateShootStandHold(RL *rl) : RLFSMState(*rl, "RLFSMStateShootStandHold") {}

    void Enter() override
    {
        std::cout << LOGGER::INFO
                  << "Standing hold. When stable: [1] frontal/hold (red) | [3] offset/twist (blue)."
                  << std::endl;
    }

    void Run() override
    {
        const auto stand_pos = GetStandHoldPos(rl);
        const auto hold_kp = GetStandHoldKp(rl);
        const auto hold_kd = GetStandHoldKd(rl);
        HoldJointPositions(*this, stand_pos, hold_kp, hold_kd);
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            rl.shooting_aim_enabled = true;
            rl.shooting_recoil_enabled = false;
            return "RLFSMStateShootFrontal";
        }
        if (rl.control.current_keyboard == Input::Keyboard::Num3 || rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft)
        {
            rl.shooting_aim_enabled = true;
            rl.shooting_recoil_enabled = false;
            return "RLFSMStateShootOffset";
        }
        return state_name_;
    }
};

class RLFSMStateGetDown : public RLFSMState
{
public:
    RLFSMStateGetDown(RL *rl) : RLFSMState(*rl, "RLFSMStateGetDown") {}

    float percent_getdown = 0.0f;

    void Enter() override
    {
        percent_getdown = 0.0f;
        rl.now_state = *fsm_state;
    }

    void Run() override
    {
        Interpolate(percent_getdown, rl.now_state.motor_state.q, rl.start_state.motor_state.q, 2.0f, "Getting down", true);
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X || percent_getdown >= 1.0f)
        {
            return "RLFSMStatePassive";
        }
        if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateShootPolicy : public RLFSMState
{
public:
    RLFSMStateShootPolicy(RL *rl, const std::string &name, bool offset_target)
        : RLFSMState(*rl, name), offset_target_(offset_target) {}

    void Enter() override
    {
        rl.episode_length_buf = 0;
        rl.EnsureShootingSimulationRunning();
        rl.shooting_aim_enabled = true;
        try
        {
            rl.LoadShootingAimPolicy(offset_target_);
            rl.now_state = *fsm_state;
        }
        catch (const std::exception &e)
        {
            std::cout << LOGGER::ERROR << "LoadShootingAimPolicy() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStateShootStandHold");
            return;
        }

        const RobotState<float> *seed_state = fsm_state;
        if (rl.IsShootingSim2Sim())
        {
            // Snap to the Isaac reset pose, then close the loop on the next
            // physics steps. Do not mj_step here; that races loop_rl.
            rl.rl_init_done = false;
            rl.shooting_sim2sim_settling = false;
            rl.shooting_policy_time_s = 0.0f;
            rl.ApplyShootingSim2SimHandoffPose();
            rl.GetState(&rl.robot_state);
            seed_state = &rl.robot_state;
            rl.rl_init_done = true;
        }
        else
        {
            rl.rl_init_done = true;
        }
        rl.SeedShootingPolicyFromCurrentState(seed_state);

        const float az_deg = rl.obs.shooting_target_azimuth * 57.2958f;
        const char *board = offset_target_ ? "OFFSET (blue, non-facing)" : "FRONTAL (red)";
        const char *policy_kind = offset_target_ ? "twist" : "hold";
        if (rl.IsShootingSim2Sim())
        {
            const float aim_obs = rl.ComputeShootingMuzzleAimError().empty()
                                      ? 0.0f
                                      : rl.ComputeShootingMuzzleAimError()[0];
            std::cout << LOGGER::INFO
                      << "SIM2SIM " << board << " " << policy_kind << " policy="
                      << rl.shooting_loaded_model_name
                      << " az=" << az_deg << " deg aim_obs=" << aim_obs
                      << ". Closed-loop: hips 80/2, thighs and calves 80/8. "
                      << "[1] frontal | [3] offset | [2] recoil | [0] stand."
                      << std::endl;
        }
        else
        {
            std::cout << LOGGER::INFO
                      << "DEPLOY " << board << " " << policy_kind << " policy="
                      << rl.shooting_loaded_model_name
                      << " az=" << az_deg << " deg. Recoil "
                      << (rl.shooting_recoil_enabled ? "ON" : "OFF")
                      << ". Press [2] to enable recoil."
                      << std::endl;
        }
    }

    void Run() override
    {
        if (rl.shooting_sim2sim_settling)
        {
            const auto stand_pos = GetStandHoldPos(rl);
            HoldJointPositions(*this, stand_pos, GetStandHoldKp(rl), GetStandHoldKd(rl));
            const float dt = rl.params.Get<float>("dt", 0.005f);
            rl.shooting_sim2sim_settle_s -= dt;
            const int settle_print_period = std::max(1, static_cast<int>(std::lround(0.5f / std::max(dt, 1e-6f))));
            if (rl.motiontime % settle_print_period == 0)
            {
                std::cout << LOGGER::INFO
                          << "SIM2SIM settling (policy OFF) "
                          << std::max(0.0f, rl.shooting_sim2sim_settle_s) << "s"
                          << std::endl;
            }
            if (rl.shooting_sim2sim_settle_s > 0.0f)
            {
                return;
            }
            rl.shooting_sim2sim_settling = false;
            rl.episode_length_buf = 0;
            rl.GetState(&rl.robot_state);
            rl.SeedShootingPolicyFromCurrentState(&rl.robot_state);
            rl.rl_init_done = true;
            const ShootingRayImpact ray = rl.ComputeShootingRayImpact();
            const float aim_obs = rl.ComputeShootingMuzzleAimError().empty()
                                      ? 0.0f
                                      : rl.ComputeShootingMuzzleAimError()[0];
            std::cout << std::endl
                      << LOGGER::INFO
                      << "SIM2SIM handoff settled ("
                      << (offset_target_ ? "twist/blue" : "hold/red")
                      << "): z=" << (rl.obs.base_pos.size() > 2 ? rl.obs.base_pos[2] : 0.0f)
                      << " aim_obs=" << aim_obs
                      << " lat=" << ray.lateral_miss_m
                      << " vert=" << ray.vertical_miss_m
                      << ". Policy closed-loop now. [1]/[3] switch policy, [2] recoil."
                      << std::endl;
        }

        if (!rl.rl_init_done)
        {
            rl.rl_init_done = true;
        }
        RLControl();
    }

    void Exit() override
    {
        rl.shooting_sim2sim_settling = false;
        rl.shooting_sim2sim_settle_s = 0.0f;
        rl.rl_init_done = false;
        if (rl.control.current_keyboard == Input::Keyboard::Num0 ||
            rl.control.current_keyboard == Input::Keyboard::Num1 ||
            rl.control.current_keyboard == Input::Keyboard::Num2 ||
            rl.control.current_keyboard == Input::Keyboard::Num3)
        {
            rl.control.ClearInput();
        }
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            rl.rl_init_done = false;
            rl.shooting_aim_enabled = false;
            rl.shooting_recoil_enabled = false;
            return "RLFSMStatePassive";
        }
        if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
        {
            if (!rl.shooting_recoil_enabled)
            {
                rl.shooting_recoil_enabled = true;
                std::cout << std::endl << LOGGER::INFO << "Recoil enabled." << std::endl;
            }
            return state_name_;
        }
        if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            rl.shooting_recoil_enabled = false;
            rl.shooting_aim_enabled = false;
            rl.rl_init_done = false;
            return "RLFSMStateGetUp";
        }
        if (!offset_target_ &&
            (rl.control.current_keyboard == Input::Keyboard::Num3 ||
             rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft))
        {
            return "RLFSMStateShootOffset";
        }
        if (offset_target_ &&
            (rl.control.current_keyboard == Input::Keyboard::Num1 ||
             rl.control.current_gamepad == Input::Gamepad::RB_DPadUp))
        {
            return "RLFSMStateShootFrontal";
        }
        return state_name_;
    }

protected:
    bool offset_target_;
};

class RLFSMStateShootFrontal : public RLFSMStateShootPolicy
{
public:
    RLFSMStateShootFrontal(RL *rl) : RLFSMStateShootPolicy(rl, "RLFSMStateShootFrontal", false) {}
};

class RLFSMStateShootOffset : public RLFSMStateShootPolicy
{
public:
    RLFSMStateShootOffset(RL *rl) : RLFSMStateShootPolicy(rl, "RLFSMStateShootOffset", true) {}
};

class RLFSMStateRLLocomotion : public RLFSMState
{
public:
    RLFSMStateRLLocomotion(RL *rl) : RLFSMState(*rl, "RLFSMStateRLLocomotion") {}

    void Enter() override
    {
        rl.episode_length_buf = 0;
        rl.config_name = "robot_lab";
        const std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_config_path);
            rl.now_state = *fsm_state;
        }
        catch (const std::exception &e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        if (!rl.rl_init_done)
        {
            rl.rl_init_done = true;
        }
        const float dt = rl.params.Get<float>("dt", 0.005f);
        const int status_period = std::max(1, static_cast<int>(std::lround(2.0f / std::max(dt, 1e-6f))));
        if (rl.motiontime % status_period == 0)
        {
            std::cout << LOGGER::INFO << "RL Controller [" << rl.config_name << "] x:" << rl.control.x
                      << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::endl;
        }
        RLControl();
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            rl.rl_init_done = false;
            return "RLFSMStatePassive";
        }
        if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLLocomotion";
        }
        return state_name_;
    }
};

} // namespace q25_fsm

class Q25FSMFactory : public FSMFactory
{
public:
    Q25FSMFactory(const std::string &initial) : initial_state_(initial) {}
    std::shared_ptr<FSMState> CreateState(void *context, const std::string &state_name) override
    {
        RL *rl = static_cast<RL *>(context);
        if (state_name == "RLFSMStatePassive")
            return std::make_shared<q25_fsm::RLFSMStatePassive>(rl);
        else if (state_name == "RLFSMStateGetUp")
            return std::make_shared<q25_fsm::RLFSMStateGetUp>(rl);
        else if (state_name == "RLFSMStateShootStandHold")
            return std::make_shared<q25_fsm::RLFSMStateShootStandHold>(rl);
        else if (state_name == "RLFSMStateGetDown")
            return std::make_shared<q25_fsm::RLFSMStateGetDown>(rl);
        else if (state_name == "RLFSMStateShootFrontal")
            return std::make_shared<q25_fsm::RLFSMStateShootFrontal>(rl);
        else if (state_name == "RLFSMStateShootOffset")
            return std::make_shared<q25_fsm::RLFSMStateShootOffset>(rl);
        else if (state_name == "RLFSMStateRLLocomotion")
            return std::make_shared<q25_fsm::RLFSMStateRLLocomotion>(rl);
        return nullptr;
    }
    std::string GetType() const override { return "q25"; }
    std::vector<std::string> GetSupportedStates() const override
    {
        return {
            "RLFSMStatePassive",
            "RLFSMStateGetUp",
            "RLFSMStateShootStandHold",
            "RLFSMStateGetDown",
            "RLFSMStateShootFrontal",
            "RLFSMStateShootOffset",
            "RLFSMStateRLLocomotion",
        };
    }
    std::string GetInitialState() const override { return initial_state_; }

private:
    std::string initial_state_;
};

REGISTER_FSM_FACTORY(Q25FSMFactory, "RLFSMStatePassive")

#endif // Q25_FSM_HPP
