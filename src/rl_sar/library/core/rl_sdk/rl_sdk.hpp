/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_SDK_HPP
#define RL_SDK_HPP

#include <iostream>
#include <string>
#include <exception>
#include <unistd.h>
#include <algorithm>
#include <tbb/concurrent_queue.h>
#include <vector>
#include <memory>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <array>

#include <yaml-cpp/yaml.h>
#include "fsm.hpp"
#include "observation_buffer.hpp"
#include "vector_math.hpp"
#include "inference_runtime.hpp"
#include "logger.hpp"
#include "motion_loader.hpp"

template <typename T>
struct RobotCommand
{
    struct MotorCommand
    {
        std::vector<int> mode;
        std::vector<T> q;
        std::vector<T> dq;
        std::vector<T> tau;
        std::vector<T> kp;
        std::vector<T> kd;

        void resize(size_t num_joints)
        {
            mode.resize(num_joints, 0);
            q.resize(num_joints, 0.0f);
            dq.resize(num_joints, 0.0f);
            tau.resize(num_joints, 0.0f);
            kp.resize(num_joints, 0.0f);
            kd.resize(num_joints, 0.0f);
        }
    } motor_command;
};

template <typename T>
struct RobotState
{
    struct IMU
    {
        std::vector<T> quaternion = {1.0f, 0.0f, 0.0f, 0.0f}; // w, x, y, z
        std::vector<T> gyroscope = {0.0f, 0.0f, 0.0f};
        std::vector<T> accelerometer = {0.0f, 0.0f, 0.0f};
    } imu;

    struct MotorState
    {
        std::vector<T> q;
        std::vector<T> dq;
        std::vector<T> ddq;
        std::vector<T> tau_est;
        std::vector<T> cur;

        void resize(size_t num_joints)
        {
            q.resize(num_joints, 0.0f);
            dq.resize(num_joints, 0.0f);
            ddq.resize(num_joints, 0.0f);
            tau_est.resize(num_joints, 0.0f);
            cur.resize(num_joints, 0.0f);
        }
    } motor_state;
};

namespace Input
{
    // Recommend: Num0-GetUp Num9-GetDown N-ToggleNavMode
    //            R-SimReset Enter-SimToggle
    //            M-MotorEnable K-MotorDisable P-MotorPassive
    //            Num1-BaseLocomotion Num2-Num8-Skills(7)
    //            WS-AxisX AD-AxisY QE-AxisYaw Space-AxisClear
    enum class Keyboard
    {
        None = 0,
        A, B, C, D, E, F, G, H, I, J, K, L, M,
        N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
        Space, Enter, Escape,
        Up, Down, Left, Right
    };

    // Recommend: A-GetUp B-GetDown X-ToggleNavMode Y-None
    //            RB_Y-SimReset RB_X-SimToggle
    //            LB_A-MotorEnable LB_B-MotorDisable LB_X-MotorPassive
    //            RB_DPadUp-BaseLocomotion RB_DPadOthers/LB_DPadOthers-Skills(7)
    //            LY-AxisX LX-AxisY RX-AxisYaw
    enum class Gamepad
    {
        None = 0,
        A, B, X, Y, LB, RB, LStick, RStick, DPadUp, DPadDown, DPadLeft, DPadRight,
        LB_A, LB_B, LB_X, LB_Y, LB_LStick, LB_RStick, LB_DPadUp, LB_DPadDown, LB_DPadLeft, LB_DPadRight,
        RB_A, RB_B, RB_X, RB_Y, RB_LStick, RB_RStick, RB_DPadUp, RB_DPadDown, RB_DPadLeft, RB_DPadRight,
        LB_RB
    };
}

struct Control
{
    Input::Keyboard current_keyboard = Input::Keyboard::None, last_keyboard = Input::Keyboard::None;
    Input::Gamepad current_gamepad = Input::Gamepad::None, last_gamepad = Input::Gamepad::None;

    float x = 0.0f;
    float y = 0.0f;
    float yaw = 0.0f;
    bool navigation_mode = false;

    void SetKeyboard(Input::Keyboard keyboad)
    {
        if (current_keyboard != keyboad)
        {
            last_keyboard = current_keyboard;
            current_keyboard = keyboad;
        }
    }

    void SetGamepad(Input::Gamepad gamepad)
    {
        if (current_gamepad != gamepad)
        {
            last_gamepad = current_gamepad;
            current_gamepad = gamepad;
        }
    }

    void ClearInput()
    {
        current_keyboard = last_keyboard;
        current_gamepad = Input::Gamepad::None;
    }
};

struct YamlParams
{
    YAML::Node config_node;

    // Get config value by key
    // WARNING: For vectors/containers, store result in a variable before using iterators/references:
    //   ✓ auto vec = params.Get<std::vector<int>>("key"); vec.begin()
    //   ✗ params.Get<std::vector<int>>("key").begin()  // dangling reference!
    template<typename T>
    T Get(const std::string& key, const T& default_value = T()) const
    {
        if (config_node[key])
        {
            return config_node[key].as<T>();
        }
        return default_value;
    }

    bool Has(const std::string& key) const
    {
        return config_node[key].IsDefined();
    }

    template<typename T>
    void Set(const std::string& key, const T& value)
    {
        config_node[key] = value;
    }
};

template <typename T>
struct Observations
{
    std::vector<T> lin_vel;
    std::vector<T> ang_vel;
    std::vector<T> gravity_vec;
    std::vector<T> commands;
    std::vector<T> base_quat;
    std::vector<T> dof_pos;
    std::vector<T> dof_vel;
    std::vector<T> actions;
    std::vector<T> muzzle_pos;
    std::vector<T> muzzle_quat;
    std::vector<T> base_pos;  // world xyz (optional; filled by sim when available)
    T shooting_target_azimuth = 0.0f;
};

struct ShootingRayImpact
{
    float target[3] = {0.0f, 0.0f, 0.0f};
    float impact[3] = {0.0f, 0.0f, 0.0f};
    float delta[3] = {0.0f, 0.0f, 0.0f};  // impact - target
    float lateral_miss_m = 0.0f;
    float vertical_miss_m = 0.0f;
    float aim_error = 0.0f;
    bool valid = false;
    bool hit = false;
};

class RL
{
public:
    RL() {};
    ~RL() {};

    YamlParams params;
    Observations<float> obs;
    std::vector<int> obs_dims;

    RobotState<float> robot_state;
    RobotCommand<float> robot_command;
    tbb::concurrent_queue<std::vector<float>> output_dof_pos_queue;
    tbb::concurrent_queue<std::vector<float>> output_dof_vel_queue;
    tbb::concurrent_queue<std::vector<float>> output_dof_tau_queue;

    FSM fsm;
    RobotState<float> start_state;
    RobotState<float> now_state;
    bool rl_init_done = false;

    // init
    void InitObservations();
    void InitOutputs();
    void InitControl();
    void InitRL(std::string robot_config_path);
    void InitJointNum(size_t num_joints);

    // rl functions
    virtual std::vector<float> Forward() = 0;
    std::vector<float> ComputeObservation();
    virtual void GetState(RobotState<float> *state) = 0;
    virtual void SetCommand(const RobotCommand<float> *command) = 0;
    void StateController(const RobotState<float> *state, RobotCommand<float> *command);
    void ComputeOutput(const std::vector<float> &actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau);

    // yaml params
    void ReadYaml(const std::string& file_path, const std::string& file_name);

    // csv logger
    std::string csv_filename;
    void CSVInit(std::string robot_name);
    void CSVLogger(const std::vector<float> &torque, const std::vector<float> &tau_est, const std::vector<float> &joint_pos, const std::vector<float> &joint_pos_target, const std::vector<float> &joint_vel);

    // control
    Control control;
    void KeyboardInterface();

    // history buffer
    ObservationBuffer history_obs_buf;
    std::vector<float> history_obs;

    // others
    int motiontime = 0;
    std::string robot_name, config_name;
    bool simulation_running = true;
    std::string ang_vel_axis = "body";  // "world" or "body"
    unsigned long long episode_length_buf = 0;
    float motion_length = 0.0;
    int InverseJointMapping(int idx) const;

    // Shooting-stability deployment (Q25): staged keyboard control, no locomotion commands.
    bool shooting_stability_mode = false;
    bool shooting_aim_enabled = false;
    bool shooting_recoil_enabled = false;
    float shooting_target_distance = 100.0f;
    float shooting_target_height = 1.3f;
    float shooting_hit_radius = 0.25f;
    float shooting_recoil_interval = 0.08f;
    std::array<float, 3> shooting_muzzle_forward_axis = {1.0f, 0.0f, 0.0f};
    void ConfigureShootingStabilityMode(bool enabled);
    void EnsureShootingSimulationRunning();
    bool shooting_scene_ready = true;
    void ResetShootingControlState();
    void HandleShootingSimulationReset();
    virtual void OnSimulationPhysicsReset() {}
    virtual void SetSimulationPaused(bool paused) { this->simulation_running = !paused; }
    std::vector<float> ComputeShootingMuzzleAimError() const;
    std::vector<float> ComputeShootingCountdown() const;
    std::vector<float> ComputeShootingTargetAzimuth() const;
    std::vector<float> NeutralShootingMuzzleAimError() const { return {0.0f}; }
    std::vector<float> NeutralShootingCountdown() const { return {0.0f}; }
    std::vector<float> NeutralShootingTargetAzimuth() const { return {0.0f, 1.0f}; }
    ShootingRayImpact ComputeShootingRayImpact() const;
    void MaybeLogShootingTelemetry();
    void SeedShootingPolicyFromCurrentState(const RobotState<float> *state);
    void DumpShootingPolicyDebugObs(const std::string &tag = "policy_step");
    void ApplyShootingDeployStabilize(std::vector<float> &dof_pos, bool update_aim_servo = true);
    void ApplyShootingSim2SimOverrides();
    bool IsShootingSim2Sim() const;
    // Frontal (red, az=0, hold policy) vs offset (blue, twist policy).
    float shooting_frontal_azimuth = 0.0f;
    float shooting_offset_azimuth = 0.0436332313f;
    bool shooting_active_target_is_offset = false;
    bool shooting_rl_yaml_loaded = false;
    std::string shooting_loaded_model_name;
    virtual void SetShootingActiveTarget(bool offset);
    void LoadShootingAimPolicy(bool offset_target);
    void ReloadShootingPolicyModel(const std::string &model_name);
    std::string ResolveShootingPolicyModelName(bool offset_target) const;
    // Snap MuJoCo qpos to the open-loop stand used for a fair policy handoff.
    virtual void ApplyShootingSim2SimHandoffPose() {}
    bool ComputeShootingStandOk(float pitch_deg, float base_z) const;
    // Link-frame wrench. Matches Isaac set_external_force_and_torque(is_global=False):
    // after the first 80 ms of the policy episode the force stays on.
    void ComputeShootingRecoilWrench(float &fx, float &fy, float &fz) const;
    float shooting_policy_time_s = 0.0f;
    virtual void ApplyShootingRecoil() {}
    bool shooting_stabilize_logged = false;
    // Runtime overlay: policy-in-the-loop (true) vs deploy cheats (false).
    bool shooting_sim2sim_mode = true;
    bool shooting_sim2sim_logged = false;
    // After [1], hold the open-loop stand in the normal physics thread before
    // enabling loop_rl. A blocking mj_step settle races the policy thread.
    bool shooting_sim2sim_settling = false;
    float shooting_sim2sim_settle_s = 0.0f;
    // Closed-loop aim trim (deploy): nudges stand pose until muzzle ray hits target.
    float shooting_aim_pitch_trim = 0.0f;
    float shooting_aim_yaw_trim = 0.0f;
    float shooting_aim_pitch_cmd = 0.0f;
    float shooting_aim_yaw_cmd = 0.0f;
    // Planted-twist joint target (Isaac order). PD tracks this; live qpos is never overwritten.
    std::vector<float> shooting_twist_q;
    float shooting_twist_blend = 0.0f;
    bool shooting_twist_valid = false;
    bool shooting_twist_need_solve = false;

    // Periodic shooting telemetry (console + optional CSV).
    bool shooting_telemetry_csv_ready = false;
    std::string shooting_telemetry_csv_filename;
    int shooting_telemetry_last_motiontime = -1;

    // Soft handoff from open-loop stand to policy (avoids one-frame joint jumps).
    bool shooting_policy_blend_active = false;
    float shooting_policy_blend_percent = 1.0f;
    std::vector<float> shooting_policy_blend_start_q;

    // Motion tracking (for mimic/dance tasks)
    std::unique_ptr<MotionLoader> motion_loader;

    // protect func
    void TorqueProtect(const std::vector<float> &origin_output_dof_tau);
    void AttitudeProtect(const std::vector<float> &quaternion, float pitch_threshold, float roll_threshold);

    // rl module
    std::unique_ptr<InferenceRuntime::Model> model;
    // output buffer
    std::vector<float> output_dof_tau;
    std::vector<float> output_dof_pos;
    std::vector<float> output_dof_vel;

    // thread safety
    std::mutex model_mutex;
};

class RLFSMState : public FSMState
{
public:
    RLFSMState(RL& rl, const std::string& name)
        : FSMState(name), rl(rl), fsm_state(nullptr), fsm_command(nullptr) {}

    RL& rl;
    const RobotState<float> *fsm_state;
    RobotCommand<float> *fsm_command;

    bool Interpolate(
        float& percent,
        const std::vector<float>& start_pos,
        const std::vector<float>& target_pos,
        float duration_seconds,
        const std::string& description = "",
        bool use_fixed_gains = true
    );

    void RLControl();
};

#endif // RL_SDK_HPP
