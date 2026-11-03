# microduck_ros2

[![CI](https://github.com/<you>/microduck_ros2/actions/workflows/ci.yml/badge.svg)](https://github.com/<you>/microduck_ros2/actions/workflows/ci.yml)

ROS 2 simulation and policy runtime for [Pollen Robotics' Microduck](https://github.com/pollen-robotics/microduck_rl),
the 14-servo RL biped. It runs the robot in MuJoCo with the same BAM XL330 actuator model the policies are
trained against, drives it with the official ONNX policies through standard ROS 2 topics, exposes its skills
(kicks, roulade, ground pick, sit/stand) as services, and navigates an apartment with Nav2.

> Community project. Not affiliated with or endorsed by Pollen Robotics.

```
   /cmd_vel  /microduck/head_cmd  /microduck/body_cmd   /microduck/skill/*  /microduck/sit|stand
                                  │                            │
   ┌───────────────┐  /joint_states, /imu/data   ┌──────────────────────────┐
   │ microduck_sim │ ─────────────────────────▶ │ microduck_policy         │
   │ MuJoCo + BAM  │ ◀───────────────────────── │ ONNX 61 → 14, skill FSM  │
   └───────────────┘   /microduck/joint_targets  └──────────────────────────┘
     │ /clock /odom /tf /scan /microduck/visual            ▲ /cmd_vel
     └──────────────────────────────▶ Nav2 + map_server / slam_toolbox
```

## Packages

| Package | What it does |
|---|---|
| `microduck_sim` | MuJoCo sim at 200 Hz physics / 50 Hz control with BAM M6 XL330 actuators (voltage, back-EMF, load-dependent friction). Publishes state, ground-truth odometry, Nav2-style TF, a virtual lidar, and RViz meshes. Scenes: `flat`, `ball`, `apartment`. Also `make_map`, which rasterizes a scene into a Nav2 map. |
| `microduck_policy` | Builds the 61D observation exactly like Pollen's `infer_policy.py`, runs the walking policy, and runs skills with the same rules as Pollen's on-robot daemon (`robotd`), read from the policy set's `manifest.json`. |
| `microduck_bringup` | Launch files (`sim`, `nav`), Nav2 and slam_toolbox params, RViz configs, gamepad skill mapping, integration tests. |

## Install

The model and policies are **not** in this repo (the meshes are CC BY-NC-SA, see [NOTICE](NOTICE)).

### Option A: pixi (recommended, any Linux including Ubuntu 22.04)

A self-contained ROS 2 Jazzy + Python 3.12 environment from RoboStack, with BAM, Nav2 and slam_toolbox:

```bash
curl -fsSL https://pixi.sh/install.sh | bash      # then open a new terminal
git clone https://github.com/<you>/microduck_ros2 && cd microduck_ros2
pixi install && pixi run fetch && pixi run build
```

`pixi run fetch` clones `microduck_rl` at the tested commit into `~/microduck_rl` and downloads
`pollen-robotics/microduck-policies` into `~/microduck-policies`.

If your `~/.bashrc` sources `/opt/ros/humble/setup.bash`, use a terminal where it isn't sourced; the two ROS
installs don't mix.

### Option B: Docker

Builds the same pixi environment, with the model, policies and map baked in:

```bash
docker build -t microduck_ros2 .
xhost +local:docker
docker run --rm -it --net=host -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix microduck_ros2              # sim
docker run --rm -it --net=host -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix microduck_ros2 pixi run nav # Nav2
```

The image contains Pollen's non-commercial meshes, so keep it local; don't push it to a public registry.

### Option C: system ROS 2 Jazzy

```bash
cd ~/ros2_ws/src && git clone https://github.com/<you>/microduck_ros2
pip install --break-system-packages mujoco onnxruntime huggingface_hub \
  "better-actuator-models @ git+https://github.com/Rhoban/bam.git@62bd8ce12154340be97e06f7f41a0ca8f116d967"
sudo apt install ros-jazzy-navigation2 ros-jazzy-slam-toolbox ros-jazzy-teleop-twist-keyboard ros-jazzy-teleop-twist-joy
./microduck_ros2/scripts/fetch_microduck_assets.sh
cd ~/ros2_ws && colcon build --symlink-install && source install/setup.bash
```

On Humble (Python 3.10) everything runs, but BAM can't load, so the sim falls back to plain position actuators
the policies weren't trained on. Use pixi instead.

## Walk

```bash
pixi run sim           # or: ros2 launch microduck_bringup sim.launch.py
pixi run keyboard      # second terminal: i/j/l/, to walk, k to stop
```

The walking policy defaults to the set's `slot: walk` entry (`velstand.onnx`, which also stands still at zero
command). Override it with `walking_policy:=alpha_walking.onnx standing_policy:=alpha_stand.onnx`.

## Skills

Every skill in the policy set gets a service, discovered from `manifest.json` the same way the robot's daemon
does it. Episodic policies with a constant command and a `duration_s` become skills; phase-encoded ones become
the ground pick; posture-flag ones become sit/stand.

```bash
pixi run sim scene:=ball          # a ball is placed in front of the foot when a kick starts
ros2 service call /microduck/skill/kick_left   std_srvs/srv/Trigger
ros2 service call /microduck/skill/kick_right  std_srvs/srv/Trigger
ros2 service call /microduck/skill/roulade     std_srvs/srv/Trigger
ros2 service call /microduck/skill/ground_pick std_srvs/srv/Trigger
ros2 service call /microduck/sit   std_srvs/srv/Trigger
ros2 service call /microduck/stand std_srvs/srv/Trigger
ros2 topic echo /microduck/policy_state      # walk, kick_left, ground_pick, sit, rise, ...
```

The rules follow Pollen's on-robot daemon (`robotd`): one move at a time, no moves while sitting, and skills
see zero head/body commands while sit/stand keeps them live. A stand requested before the seat settles
(`ramp_s`) waits for it, then the sitstand net rises for `unwind_s` before handing back to the gait.

**Gamepad:** `pixi run sim joy:=true`. Hold LB and use the sticks to walk; A ground pick, B kick right,
RB kick left, X roulade, Y sit/stand.

## Navigate (Nav2)

```bash
pixi run nav                 # known map of the apartment, generated from the scene on first run
pixi run nav mode:=slam      # build the map with slam_toolbox while it walks
```

Click **2D Goal Pose** in RViz. The apartment is Pollen's 6-room, ~7 × 6 m flat from `microduck_rl`. Nav2 uses
Regulated Pure Pursuit at 0.2 m/s (inside velstand's trained range), NavFn, and a velocity smoother.

Two simplifications, both sim-only: odometry is ground truth (so in map mode `map → odom` is identity), and
`/scan` comes from a **virtual planar lidar** (360 rays at 12 cm height, ray-cast against the scene and
ignoring the robot). The real Microduck has no lidar; a camera- or ToF-based scan would replace it.

## Topics and services

| Name | Type | |
|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | sim → |
| `/imu/data` | `sensor_msgs/Imu` (orientation = trunk orientation, gyro, accel) | sim → |
| `/odom` | ground truth, `odom → base_footprint` | sim → |
| `/tf` | `odom → base_footprint → trunk_base`, `base_footprint → laser_link` | sim → |
| `/scan` | `sensor_msgs/LaserScan` (virtual lidar) | sim → |
| `/microduck/visual` | `visualization_msgs/MarkerArray` (robot meshes + scene) | sim → |
| `/clock` | sim time | sim → |
| `/microduck/joint_targets` | `sensor_msgs/JointState` (positions, rad) | policy → sim |
| `/microduck/policy_state` | `std_msgs/String` | policy → |
| `/cmd_vel` | `geometry_msgs/Twist` (linear.x, linear.y, angular.z) | → policy |
| `/microduck/head_cmd`, `/microduck/body_cmd` | `std_msgs/Float64MultiArray` (4, 6) | → policy |
| `/microduck/skill/<name>`, `/microduck/sit`, `/microduck/stand` | `std_srvs/Trigger` | policy |
| `/microduck/reset`, `/microduck/place_ball_{left,right}` | `std_srvs/Trigger` | sim |

Head and body pose commands (zero by default):

```bash
ros2 topic pub --once /microduck/head_cmd std_msgs/Float64MultiArray "{data: [0.0, 0.2, 0.3, 0.0]}"   # neck_pitch head_pitch head_yaw head_roll [rad]
ros2 topic pub --once /microduck/body_cmd std_msgs/Float64MultiArray "{data: [0, 0, -0.02, 0, 0.2, 0]}" # x y z [m] roll pitch yaw [rad]
```

## How it stays faithful to training

- **Observation:** a test runs Pollen's own `PolicyInference.get_observations()` from `microduck_rl` on random
  states and checks ours matches to 1e-6. Same for the default pose against the `STAND` keyframe.
- **Actuators:** the BAM setup is ported from `infer_policy.py` (motor-mode actuators, voltage-bounded force
  range, zeroed joint damping/friction, stiff friction constraint, 7.4 V with 0.1 V/Nm sag).
- **Action scale:** 1.0 with no target filter, as in training and Pollen's CPU rehearsal. The physical robot's
  daemon runs 0.9 and a low-pass; that belongs in a future hardware bridge, not the sim.
- **Timing:** in `sync_mode` (default) the sim publishes state, waits for the targets computed from that exact
  state (matched on the header stamp), then steps. A CI test checks the ROS loop is bit-identical to running
  sim and policy in one process. `sync_mode:=false` free-runs in real time, closer to hardware.

## Tests

```bash
pixi run test                 # unit: obs parity with upstream, skills FSM, sim, lidar, map
pixi run test-integration     # real nodes over ROS 2: sync loop, every skill, Nav2 planning
```

CI runs both on every push. The tests use stand-in ONNX nets, so they check plumbing and timing, not gait quality.

## Limitations

- Feed-forward policies only. Recurrent (`model_api: 2`) policies are refused with a clear error.
- No hardware driver yet. The topic interface is designed so a `microduck_hardware` node can replace the sim.
- No URDF yet: RViz shows the robot through mesh markers published by the sim.
- Roller-skate mode (`roller.onnx`) needs the roller model and isn't wired up.

## License

Code: Apache-2.0. The Microduck model and policies you download stay under their own licenses. See [NOTICE](NOTICE).
