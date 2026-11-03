#!/usr/bin/env bash
# Fetch the Microduck model (from pollen-robotics/microduck_rl) and the official
# policy set (pollen-robotics/microduck-policies on Hugging Face).
#
# They are NOT vendored in this repo: the meshes are CC BY-NC-SA 4.0
# (non-commercial), while this repo is Apache-2.0. See NOTICE.
#
#   ./scripts/fetch_microduck_assets.sh                 # -> ~/microduck_rl, ~/microduck-policies
#   MICRODUCK_RL_PATH=/opt/microduck_rl ./scripts/fetch_microduck_assets.sh
set -euo pipefail

RL_PATH="${MICRODUCK_RL_PATH:-$HOME/microduck_rl}"
POLICIES_PATH="${MICRODUCK_POLICIES_PATH:-$HOME/microduck-policies}"
# Commit this ROS port was written and tested against.
RL_COMMIT="${MICRODUCK_RL_COMMIT:-bae39f5900b318f7e87fc485c30a31213cfa14eb}"

if [ ! -d "$RL_PATH/.git" ]; then
  echo ">> cloning microduck_rl into $RL_PATH"
  git clone https://github.com/pollen-robotics/microduck_rl "$RL_PATH"
fi
echo ">> checking out microduck_rl @ ${RL_COMMIT:0:7}"
git -C "$RL_PATH" fetch --quiet origin "$RL_COMMIT" 2>/dev/null || git -C "$RL_PATH" fetch --quiet origin
git -C "$RL_PATH" -c advice.detachedHead=false checkout --quiet "$RL_COMMIT"

if command -v hf >/dev/null 2>&1; then
  HF=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli download)
else
  echo ">> installing huggingface_hub CLI (pip --user)"
  python3 -m pip install --user --quiet "huggingface_hub[cli]" || \
    python3 -m pip install --user --quiet --break-system-packages "huggingface_hub[cli]"
  export PATH="$HOME/.local/bin:$PATH"
  HF=(hf download)
fi

echo ">> downloading pollen-robotics/microduck-policies into $POLICIES_PATH"
"${HF[@]}" pollen-robotics/microduck-policies --local-dir "$POLICIES_PATH"

echo
echo "Model:    $RL_PATH/src/mjlab_microduck/robot/microduck/scene.xml"
echo "Policies: $POLICIES_PATH"
if [ -f "$POLICIES_PATH/manifest.json" ]; then
  python3 - "$POLICIES_PATH/manifest.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
for p in m.get("policies", []):
    print(f"  {p.get('file',''):<28} kind={p.get('kind','?'):<10} slot={p.get('slot','-'):<8} {p.get('description','')[:60]}")
EOF
fi
echo
echo "Next: ros2 launch microduck_bringup sim.launch.py walking_policy:=$POLICIES_PATH/<walking file>.onnx"
