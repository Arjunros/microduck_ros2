# microduck_ros2

ROS 2 simulation and policy runtime for [Pollen Robotics' Microduck](https://github.com/pollen-robotics/microduck_rl),
the 14-servo RL biped. It runs the robot in MuJoCo with the same BAM XL330 actuator model the policies are
trained against, and drives it with the official ONNX policies through standard ROS 2 topics, so you can
teleop it with `/cmd_vel`, watch it in RViz, record it with rosbag, and plug it into the rest of a ROS stack.

> Community project. Not affiliated with or endorsed by Pollen Robotics.

```
            /cmd_vel  /microduck/head_cmd  /microduck/body_cmd
                              │
   ┌──────────────┐  /joint_states, /imu/data   ┌──────────────────┐
   │ microduck_sim │ ─────────────────────────▶ │ microduck_policy │
   │ MuJoCo + BAM  │ ◀───────────────────────── │  ONNX, 61 → 14   │
   └──────────────┘   /microduck/joint_targets  └──────────────────┘
      │ /clock /odom /tf /microduck/visual (RViz meshes)
```

## Packages

| Package | What it does |
|---|---|
| `microduck_sim` | MuJoCo sim at 200 Hz physics / 50 Hz control. BAM M6 XL330 actuators (voltage, back-EMF, load-dependent friction). Publishes state, ground-truth odometry, TF and mesh markers. |
| `microduck_policy` | Builds the 61D observation exactly like Pollen's `infer_policy.py`, runs the ONNX policy, publishes 14 joint targets. Optional standing policy with Pollen's switching rule. Reads `manifest.json` (action scale, compatibility checks). |
| `microduck_bringup` | Launch file, parameters, RViz config, optional gamepad teleop. |

## Install

The model and policies are **not** in this repo (the meshes are CC BY-NC-SA, see [NOTICE](NOTICE)). The
fetch script clones `microduck_rl` at the tested commit into `~/microduck_rl` and downloads
`pollen-robotics/microduck-policies` into `~/microduck-policies`:

```bash
./scripts/fetch_microduck_assets.sh
```

### Option A: ROS 2 Jazzy (recommended)

BAM needs Python 3.12, which is what Jazzy ships with.

```bash
cd ~/ros2_ws/src && git clone https://github.com/<you>/microduck_ros2
pip install --break-system-packages mujoco onnxruntime \
  "better-actuator-models @ git+https://github.com/Rhoban/bam.git@62bd8ce12154340be97e06f7f41a0ca8f116d967"
sudo apt install ros-jazzy-teleop-twist-keyboard ros-jazzy-teleop-twist-joy
cd ~/ros2_ws && colcon build --symlink-install && source install/setup.bash
```

### Option B: ROS 2 Humble

Everything works, but BAM cannot load on Python 3.10, so the sim falls back to the XML position actuators and
logs a warning. Policies are not trained on those, so walking quality drops. Fine for plumbing and RViz work;
use Option A or C for anything you want to trust.

```bash
pip install --user mujoco onnxruntime
```

### Option C: pixi (any Linux, including Ubuntu 22.04)

A self-contained Jazzy + Python 3.12 environment from RoboStack, BAM included:

```bash
pixi install && pixi run fetch && pixi run build
```

## Run

```bash
ros2 launch microduck_bringup sim.launch.py \
  walking_policy:=$HOME/microduck-policies/<walking>.onnx \
  standing_policy:=$HOME/microduck-policies/<standing>.onnx   # optional

ros2 run teleop_twist_keyboard teleop_twist_keyboard          # i/j/l/, to walk
```

The fetch script prints the files in the policy set with their `kind` and `slot`, so you can pick the walking
and standing ones. Launch arguments: `use_bam`, `sync_mode`, `rviz`, `joy` (gamepad: hold LB, left stick to
walk, right stick to turn), `microduck_rl_path`.

Reset: `ros2 service call /microduck/reset std_srvs/srv/Trigger`

Head and body pose commands (zero by default):

```bash
ros2 topic pub --once /microduck/head_cmd std_msgs/Float64MultiArray "{data: [0.0, 0.2, 0.3, 0.0]}"   # neck_pitch head_pitch head_yaw head_roll [rad]
ros2 topic pub --once /microduck/body_cmd std_msgs/Float64MultiArray "{data: [0, 0, -0.02, 0, 0.2, 0]}" # x y z [m] roll pitch yaw [rad]
```

## Topics

| Topic | Type | Dir |
|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | sim → |
| `/imu/data` | `sensor_msgs/Imu` (orientation = trunk orientation, gyro, accel) | sim → |
| `/odom`, `/tf` | ground-truth trunk pose, `world → trunk_base` | sim → |
| `/microduck/visual` | `visualization_msgs/MarkerArray` | sim → |
| `/clock` | sim time | sim → |
| `/microduck/joint_targets` | `sensor_msgs/JointState` (positions, rad) | policy → sim |
| `/cmd_vel` | `geometry_msgs/Twist` (linear.x, linear.y, angular.z) | → policy |
| `/microduck/head_cmd`, `/microduck/body_cmd` | `std_msgs/Float64MultiArray` (4, 6) | → policy |
| `/microduck/policy_obs` | `std_msgs/Float32MultiArray` (61D, when `publish_debug`) | policy → |

## How it stays faithful to training

- **Observation**: a test runs Pollen's own `PolicyInference.get_observations()` from `microduck_rl` on random
  states and checks ours matches to 1e-6. Same for the default pose against the `STAND` keyframe.
- **Actuators**: the BAM setup is ported from `infer_policy.py` (motor-mode actuators, voltage-bounded force
  range, zeroed joint damping/friction, stiff friction constraint, 7.4 V with 0.1 V/Nm sag).
- **Timing**: in `sync_mode` (default) the sim publishes state, waits for the targets computed from that exact
  state (matched on the header stamp), then steps. The ROS loop is bit-identical to running the sim and the
  policy in one Python process, which is tested. `sync_mode:=false` free-runs in real time, closer to hardware.

## Limitations (v0.1)

- Feed-forward policies only. Recurrent (`model_api: 2`) policies are refused with a clear error.
- Episodic skills (kicks, roulade, ground pick, sit-stand) aren't wired up yet; walking + standing are.
- No hardware driver yet. The topic interface is designed so a `microduck_hardware` node can replace the sim.
- No URDF yet: RViz shows the robot through mesh markers published by the sim.

## Tests

```bash
MICRODUCK_RL_PATH=~/microduck_rl PYTHONPATH=microduck_policy:microduck_sim python -m pytest microduck_policy/test microduck_sim/test
```

## License

Code: Apache-2.0. The Microduck model and policies you download stay under their own licenses. See [NOTICE](NOTICE).
