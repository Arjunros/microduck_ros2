# microduck_ros2

[![CI](https://github.com/<you>/microduck_ros2/actions/workflows/ci.yml/badge.svg)](https://github.com/<you>/microduck_ros2/actions/workflows/ci.yml)

ROS 2 simulation and policy runtime for [Pollen Robotics' Microduck](https://github.com/pollen-robotics/microduck_rl),
the 14-servo RL biped. It runs the robot in MuJoCo with the same BAM XL330 actuator model the policies are
trained against, drives it with the official ONNX policies through standard ROS 2 topics, exposes its skills
(kicks, roulade, ground pick, sit/stand) as services, sees through its head camera, navigates an apartment
with Nav2, and takes orders in plain English (typed or spoken) through an LLM agent.

> Community project. Not affiliated with or endorsed by Pollen Robotics.

```
  "go to the kitchen and sit" ─▶ microduck_agent (rule parser | Ollama | OpenAI | Claude)
                                    │ Nav2 goals, /cmd_vel, skills, head, camera
   /cmd_vel /microduck/head_cmd     ▼                  /microduck/skill/*  /microduck/sit|stand
   ┌───────────────┐  /joint_states, /imu/data   ┌──────────────────────────┐
   │ microduck_sim │ ─────────────────────────▶ │ microduck_policy         │
   │ MuJoCo + BAM  │ ◀───────────────────────── │ ONNX 61 → 14, skill FSM  │
   └───────────────┘   /microduck/joint_targets  └──────────────────────────┘
     │ /clock /odom /tf /scan /microduck/sim_qpos          ▲ /cmd_vel
     ├──▶ camera_node ──▶ /camera/image_raw, /camera/depth/image_raw
     ├──▶ robot_state_publisher (URDF generated from the MuJoCo model) ──▶ /tf
     └──▶ Nav2 + map_server / slam_toolbox
```

## Packages

| Package | What it does |
|---|---|
| `microduck_sim` | MuJoCo sim at 200 Hz physics / 50 Hz control with BAM M6 XL330 actuators (voltage, back-EMF, load-dependent friction). Publishes state, ground-truth odometry, Nav2-style TF and a virtual lidar. Scenes: `flat`, `ball`, `apartment` (+ `ball:=true` anywhere). Also `camera_node` (head camera, RGB + depth), `make_urdf` (URDF generated from the MuJoCo model) and `make_map` (Nav2 map + named rooms from a scene). |
| `microduck_policy` | Builds the 61D observation exactly like Pollen's `infer_policy.py`, runs the walking policy, and runs skills with the same rules as Pollen's on-robot daemon (`robotd`), read from the policy set's `manifest.json`. |
| `microduck_agent` | Natural-language control. An LLM (or an offline rule parser) turns requests into tool calls: go to a room, go to the ball, walk, turn, skills, look, take a picture, stop. Text chat CLI and push-to-talk voice. |
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

Builds the CI-tested pixi environment (with software OpenGL for the camera), with the model, policies and map baked in:

```bash
docker build -t microduck_ros2 .
xhost +local:docker
docker run --rm -it --net=host -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix microduck_ros2              # sim
docker run --rm -it --net=host -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix microduck_ros2 pixi run -e ci nav # Nav2
```

The image contains Pollen's non-commercial meshes, so keep it local; don't push it to a public registry.

### Option C: system ROS 2 Jazzy

```bash
cd ~/ros2_ws/src && git clone https://github.com/<you>/microduck_ros2
pip install --break-system-packages mujoco onnxruntime huggingface_hub pillow \
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

## Head camera

On by default (`camera:=false` to skip). `camera_node` renders MuJoCo's `head_camera` at the real sensor's
640×360 / 15 fps and publishes `/camera/image_raw` (rgb8), `/camera/depth/image_raw` (32FC1, metres) and
`/camera/camera_info`, in `head_camera_optical_frame`. It runs in its own process, so rendering never slows
the physics.

Two corrections, both applied to the shared model so the TF frame matches the image exactly:

- **The model's camera faces backwards.** Pollen's own simulator notes this and turns it 180°; we do the same.
- **It's mounted a quarter turn off**, like the real sensor. We turn it upright for ROS consumers, so the
  image's up is the world's up.

It picks an OpenGL backend automatically (EGL, then GLFW, then OSMesa). On a machine with no GPU, use the
pixi `ci` environment, which adds Mesa's software renderer: `pixi run -e ci sim`. It's slow, but it works.

## URDF and TF

`robot_state_publisher` gets a URDF generated at launch from the MuJoCo model (`ros2 run microduck_sim
make_urdf` writes it out). Kinematics come straight from MuJoCo, and a test checks every link, the camera and
the IMU match MuJoCo's forward kinematics to 1e-7 m. RViz shows the robot with `RobotModel`, and frames such as
`head_camera_optical_frame`, `imu`, `tof`, `left_foot` and `right_foot` are on `/tf` for other tools (Foxglove,
MoveIt, image_proc).

```
odom → base_footprint → trunk_base → hips, legs, feet, neck → head → head_camera_optical_frame
                     └→ laser_link
```

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

`make_map` also writes `apartment_places.yaml`: one reachable point per room (kitchen, living room,
bedroom, office, bathroom, the corridors, start), found from the scene's floor geometry, plus aliases
like "sofa" or "home". Edit it to add your own spots.

## Talk to it

```bash
pixi run nav agent:=true ball:=true      # sim + Nav2 + agent, with a ball in the living room
pixi run chat                            # second terminal
```
```
you> go to the ball and kick it
duck: Done: next to the ball, facing it; did kick right
you> go to the kitchen then sit
you> turn around and walk forward 30 cm
you> where are you?
you> stop                                # handled at once, even mid-walk
```

Pick the brain with `backend:=`:

| backend | needs | notes |
|---|---|---|
| `rule` (default) | nothing | Offline keyword parser: rooms, "then", turns, distances, skills, look, stop. |
| `ollama` | [Ollama](https://ollama.com) + a tool-calling model | `backend:=ollama model:=<model>`. Free and local. |
| `openai` | `OPENAI_API_KEY` | `backend:=openai model:=<model>`. Any OpenAI-compatible server works with `base_url:=`. |
| `anthropic` | `ANTHROPIC_API_KEY` | `backend:=anthropic` (default model `claude-haiku-5-5`, or `model:=`). |

With an LLM backend the agent plans multi-step requests itself, and with a vision-capable model it can
**look**: "what's in front of you?" calls `see`, which sends the head-camera image to the model.

**Voice (experimental):** `pixi run -e voice voice`. Press Enter, speak, press Enter. Speech-to-text is
faster-whisper running locally (it downloads its model on first use), and replies are spoken with `espeak-ng`
if installed (`sudo apt install espeak-ng`). Anything else that publishes text to `/microduck/command` works
too, e.g. your own assistant.

The agent publishes on `/microduck/agent/reply` and `/microduck/agent/events` (tool calls, for UIs). In the
apartment, `go_to_ball` reads the ball's ground-truth position from `/microduck/ball`, a sim-only shortcut
that ball detection on `/camera/image_raw` would replace on a real robot.

## Topics and services

| Name | Type | |
|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | sim → |
| `/imu/data` | `sensor_msgs/Imu` (orientation = trunk orientation, gyro, accel) | sim → |
| `/odom` | ground truth, `odom → base_footprint` | sim → |
| `/tf` | `odom → base_footprint → trunk_base`, `base_footprint → laser_link` | sim → |
| `/scan` | `sensor_msgs/LaserScan` (virtual lidar) | sim → |
| `/microduck/visual` | `visualization_msgs/MarkerArray` (scene geoms; namespaces `scene`, `objects`) | sim → |
| `/microduck/sim_qpos` | full state for the camera node | sim → |
| `/microduck/ball` | `geometry_msgs/PointStamped` (ball scenes) | sim → |
| `/camera/image_raw`, `/camera/depth/image_raw`, `/camera/camera_info` | head camera | camera → |
| `/robot_description` | URDF | robot_state_publisher → |
| `/microduck/command` → `/microduck/agent/reply`, `/microduck/agent/events` | `std_msgs/String` | agent |
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
pixi run test                 # unit: obs parity, skills FSM, sim, lidar, map, URDF kinematics, agent + backends
pixi run -e ci test-integration   # real nodes over ROS 2: sync loop, skills, Nav2, camera + TF, agent → Nav2
```

CI runs both on every push. The tests use stand-in ONNX nets, so they check plumbing and timing, not gait quality.

## Limitations

- Feed-forward policies only. Recurrent (`model_api: 2`) policies are refused with a clear error.
- No hardware driver yet. The topic interface is designed so a `microduck_hardware` node can replace the sim.
- The voice front-end hasn't been tested with a microphone in CI.
- Roller-skate mode (`roller.onnx`) needs the roller model and isn't wired up.

## License

Code: Apache-2.0. The Microduck model and policies you download stay under their own licenses. See [NOTICE](NOTICE).
