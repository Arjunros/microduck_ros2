"""Ground-truth occupancy map of a MuJoCo scene, for Nav2's map_server.

A cell is occupied if a collidable scene geom (not the robot) occupies any of a
few heights in the duck's body band, or if there is no floor under it.

  ros2 run microduck_sim make_map --scene apartment --out ~/microduck_maps/apartment
"""

import argparse
import os

import mujoco
import numpy as np

from .engine import SCENE_DIR_REL, SCENES


def _inside(points, geom_type, pos, R, size):
    """Boolean mask: which points lie inside a box / cylinder / sphere / capsule geom."""
    p = (points - pos) @ R  # into the geom frame (R columns are geom axes)
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return np.all(np.abs(p) <= size[:3], axis=1)
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return (np.hypot(p[:, 0], p[:, 1]) <= size[0]) & (np.abs(p[:, 2]) <= size[1])
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.linalg.norm(p, axis=1) <= size[0]
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        z = np.clip(p[:, 2], -size[1], size[1])
        return np.hypot(np.hypot(p[:, 0], p[:, 1]), p[:, 2] - z) <= size[0]
    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return np.sum((p / size[:3]) ** 2, axis=1) <= 1.0
    return np.zeros(len(points), dtype=bool)


def occupancy(model, resolution=0.05, heights=(0.03, 0.08, 0.15, 0.25), margin=0.25,
              robot_body="trunk_base"):
    """Returns (grid[ny, nx] bool occupied, origin_xy)."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_body)
    geoms = [g for g in range(model.ngeom)
             if model.body_rootid[model.geom_bodyid[g]] != robot
             and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
    solid = [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
    has_plane = any(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in range(model.ngeom))

    # Bounds from the scene geoms' bounding spheres.
    centers = data.geom_xpos[geoms][:, :2]
    rb = model.geom_rbound[geoms][:, None]
    lo = (centers - rb).min(axis=0) - margin
    hi = (centers + rb).max(axis=0) + margin
    nx, ny = (np.ceil((hi - lo) / resolution)).astype(int)
    xs = lo[0] + (np.arange(nx) + 0.5) * resolution
    ys = lo[1] + (np.arange(ny) + 0.5) * resolution
    X, Y = np.meshgrid(xs, ys)
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)

    occ = np.zeros(len(xy), dtype=bool)
    for h in heights:
        pts = np.column_stack([xy, np.full(len(xy), h)])
        for g in solid:
            occ |= _inside(pts, model.geom_type[g], data.geom_xpos[g],
                           data.geom_xmat[g].reshape(3, 3), model.geom_size[g])

    if not has_plane:  # no ground plane: only cells with a floor under them are walkable
        below = np.column_stack([xy, np.full(len(xy), -0.004)])
        floor = np.zeros(len(xy), dtype=bool)
        for g in solid:
            floor |= _inside(below, model.geom_type[g], data.geom_xpos[g],
                             data.geom_xmat[g].reshape(3, 3), model.geom_size[g])
        occ |= ~floor
    grid = occ.reshape(ny, nx)

    # Crop to the free area plus a margin (bounding spheres over-estimate extents).
    free_r, free_c = np.nonzero(~grid)
    if len(free_r):
        pad = int(np.ceil(margin / resolution))
        r0, r1 = max(free_r.min() - pad, 0), min(free_r.max() + pad + 1, ny)
        c0, c1 = max(free_c.min() - pad, 0), min(free_c.max() + pad + 1, nx)
        grid = grid[r0:r1, c0:c1]
        lo = lo + np.array([c0, r0]) * resolution
    return grid, lo


def write_map(grid, origin, resolution, out_prefix):
    """Nav2 map_server format: PGM (row 0 = top = max y) + YAML."""
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or ".", exist_ok=True)
    img = np.where(grid, 0, 254).astype(np.uint8)[::-1]
    pgm = out_prefix + ".pgm"
    with open(pgm, "wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
        f.write(img.tobytes())
    yml = out_prefix + ".yaml"
    with open(yml, "w") as f:
        f.write(f"image: {os.path.basename(pgm)}\nmode: trinary\nresolution: {resolution}\n"
                f"origin: [{origin[0]:.4f}, {origin[1]:.4f}, 0.0]\nnegate: 0\n"
                "occupied_thresh: 0.65\nfree_thresh: 0.25\n")
    return yml


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="apartment", help="flat | ball | apartment, or a path to a scene XML")
    ap.add_argument("--microduck-rl-path",
                    default=os.environ.get("MICRODUCK_RL_PATH", os.path.expanduser("~/microduck_rl")))
    ap.add_argument("--out", default=None, help="output prefix (default ~/microduck_maps/<scene>)")
    ap.add_argument("--resolution", type=float, default=0.05)
    args = ap.parse_args(argv)

    scene = args.scene
    if scene in SCENES:
        scene = os.path.join(os.path.expanduser(args.microduck_rl_path), SCENE_DIR_REL, SCENES[scene])
    out = args.out or os.path.expanduser(
        f"~/microduck_maps/{os.path.splitext(os.path.basename(scene))[0].replace('scene_', '')}")
    model = mujoco.MjModel.from_xml_path(scene)
    grid, origin = occupancy(model, args.resolution)
    yml = write_map(grid, origin, args.resolution, os.path.expanduser(out))
    print(f"{yml}: {grid.shape[1]}x{grid.shape[0]} cells at {args.resolution} m, "
          f"{100 * grid.mean():.0f}% occupied")


if __name__ == "__main__":
    main()
