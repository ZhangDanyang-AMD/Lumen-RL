"""Unitree G1 (29 DOF) model for the Isaac-Velocity-Rough-G1-v0 benchmark, derived from MuJoCo Menagerie's unitree_g1.

Source: mujoco_menagerie unitree_g1/g1.xml and its meshes (upstream license: see the Menagerie repository). The derived
XML is written to assets/menagerie_unitree_g1/g1_boxes.xml by ``python g1_model.py --src DIR --write``, where DIR holds
g1.xml and assets/ (downloaded once; not vendored). The derived XML has no mesh.

Isaac Lab v2.3.0 settings (isaaclab_assets/robots/unitree.py G1_CFG with G1_MINIMAL_CFG, and the rough velocity env):

  * every collision mesh is replaced by one box bounding its vertices, oriented by their principal axes (PCA of the
    compiled mesh's vertices; an axis-aligned box would reach the floor through a tilted link). The feet keep
    Menagerie's foot spheres. hipPhysics's lab build takes primitive collision only.
  * self collisions off (enabled_self_collisions=False): every non-adjacent robot body pair is excluded.
  * implicit PD actuators per group with Isaac's stiffness, damping and effort limit: legs (hip yaw and roll 150, hip
    pitch and knee 200; damping 5; effort 300), ankles (20, 2, 20), waist (200, 5, 300), arms and wrists (40, 10, 300).
    MuJoCo position actuator: force = kp (ctrl - q) - kv qdot, clipped to the effort limit (the implicit drive).
  * armature per group (legs, waist, ankles and shoulders/elbows 0.01, wrists 0.001); joint friction 0 (Isaac has none).
  * default pose G1_CFG.init_state.joint_pos. Isaac's `.*_elbow_pitch_joint` is Menagerie's `*_elbow_joint`; Isaac's
    `torso_joint` (G1_CFG) is applied to Menagerie's three waist joints.
  * gravity on for the robot.
"""

from __future__ import annotations

import argparse
import itertools
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets" / "menagerie_unitree_g1" / "g1_boxes.xml"
ROOT_POS = (0.0, 0.0, 0.79)  # standing height of this model (Menagerie keyframe "stand"); G1_CFG init is 0.74

# (joints, kp, kv, effort limit, armature)
GROUPS = (
    (("left_hip_yaw_joint", "right_hip_yaw_joint"), 150.0, 5.0, 300.0, 0.01),
    (("left_hip_roll_joint", "right_hip_roll_joint"), 150.0, 5.0, 300.0, 0.01),
    (("left_hip_pitch_joint", "right_hip_pitch_joint"), 200.0, 5.0, 300.0, 0.01),
    (("left_knee_joint", "right_knee_joint"), 200.0, 5.0, 300.0, 0.01),
    (("left_ankle_pitch_joint", "right_ankle_pitch_joint"), 20.0, 2.0, 20.0, 0.01),
    (("left_ankle_roll_joint", "right_ankle_roll_joint"), 20.0, 2.0, 20.0, 0.01),
    (("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"), 200.0, 5.0, 300.0, 0.01),
    (tuple(f"{s}_{j}_joint" for s in ("left", "right")
           for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")), 40.0, 10.0, 300.0, 0.01),
    (tuple(f"{s}_{j}_joint" for s in ("left", "right")
           for j in ("wrist_roll", "wrist_pitch", "wrist_yaw")), 40.0, 10.0, 300.0, 0.001),
)
# actuated joints in action order: the groups' joints (29)
ACTION_JOINTS = tuple(j for joints, *_ in GROUPS for j in joints)
JOINT_PARAMS = {j: (kp, kv, eff, arm) for joints, kp, kv, eff, arm in GROUPS for j in joints}
DEFAULT_POS = {
    "left_hip_pitch_joint": -0.20, "right_hip_pitch_joint": -0.20,
    "left_knee_joint": 0.42, "right_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.23, "right_ankle_pitch_joint": -0.23,
    "left_elbow_joint": 0.87, "right_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.16, "right_shoulder_roll_joint": -0.16,
    "left_shoulder_pitch_joint": 0.35, "right_shoulder_pitch_joint": 0.35,
}
FEET_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_FRICTION = "0.6 0.005 0.0001"


def _collision_mesh_geoms(root: ET.Element):
    """(parent, geom) for every mesh collision geom, in document order."""
    for parent in root.iter():
        for geom in list(parent):
            if geom.tag == "geom" and geom.get("class") == "collision" and geom.get("mesh"):
                yield parent, geom


def _strip_visuals(root: ET.Element) -> None:
    for parent in root.iter():
        for geom in list(parent):
            if geom.tag == "geom" and (geom.get("class") == "visual" or geom.get("group") == "2"):
                parent.remove(geom)


def _name_mesh_collisions(root: ET.Element) -> list:
    names = []
    for i, (_parent, geom) in enumerate(_collision_mesh_geoms(root)):
        geom.set("name", f"mesh_col_{i}")
        names.append(f"mesh_col_{i}")
    return names


def _oriented_box(model, gid: int):
    """Box (body-frame position, quaternion, half sizes) bounding one mesh geom's vertices, oriented by their principal
    axes (PCA). An axis-aligned box of a tilted link (the knee link's shin) reaches the floor at standing; the oriented
    box follows the link. mesh_vert is in the geom frame (checked against geom_aabb)."""
    import mujoco

    dataid = int(model.geom_dataid[gid])
    va, vn = int(model.mesh_vertadr[dataid]), int(model.mesh_vertnum[dataid])
    v = model.mesh_vert[va:va + vn].astype(np.float64)
    c = v.mean(0)
    _, _, vt = np.linalg.svd(v - c, full_matrices=False)
    rot = vt.T
    if np.linalg.det(rot) < 0:
        rot[:, 2] *= -1
    p = (v - c) @ rot
    lo, hi = p.min(0), p.max(0)
    center = c + rot @ ((lo + hi) / 2)  # box centre in the geom frame
    half = (hi - lo) / 2
    g_rot = np.zeros(9)
    mujoco.mju_quat2Mat(g_rot, model.geom_quat[gid])
    pos = model.geom_pos[gid] + g_rot.reshape(3, 3) @ center
    box_quat = np.zeros(4)
    r_quat = np.zeros(4)
    mujoco.mju_mat2Quat(r_quat, rot.reshape(-1))
    mujoco.mju_mulQuat(box_quat, model.geom_quat[gid], r_quat)
    return pos, box_quat, half


def derive(src_dir: Path) -> str:
    """The derived G1 MJCF text. src_dir holds g1.xml and assets/ (Menagerie layout)."""
    import mujoco

    # 1) the mesh model, compiled with its meshes, gives each collision mesh's vertices in its geom frame
    full = ET.parse(src_dir / "g1.xml").getroot()
    names = _name_mesh_collisions(full)
    named_path = src_dir / "_g1_named_for_aabb.xml"
    named_path.write_text(ET.tostring(full, encoding="unicode"), encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(named_path))
    finally:
        named_path.unlink()
    aabb = {}
    for name in names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        aabb[name] = _oriented_box(model, gid)

    # 2) the derived MJCF: no visual, no mesh; boxes for the mesh collisions
    root = ET.parse(src_dir / "g1.xml").getroot()
    _strip_visuals(root)
    _name_mesh_collisions(root)
    for parent, geom in list(_collision_mesh_geoms(root)):
        name = geom.get("name")
        pos, quat, half = aabb[name]
        parent.remove(geom)
        ET.SubElement(parent, "geom", name=f"{name}_box", type="box",
                      size=" ".join(f"{h:.6f}" for h in half), pos=" ".join(f"{p:.6f}" for p in pos),
                      quat=" ".join(f"{q:.6f}" for q in quat), group="3", friction=FOOT_FRICTION)
    for geom in root.iter("geom"):
        if geom.get("class") == "foot":
            geom.set("friction", FOOT_FRICTION)
    for default in root.iter("default"):
        for joint in default.findall("joint"):
            default.remove(joint)  # the class joint defaults (frictionloss 0.3) would leak into the bodies
        for child in default.findall("default"):
            if child.get("class") == "visual":
                default.remove(child)  # visual class: mesh geoms, unused now
    for asset in root.findall("asset"):
        root.remove(asset)
    root.find("compiler").attrib.pop("meshdir", None)

    for joint in root.iter("joint"):
        name = joint.get("name")
        if name == "floating_base_joint":
            continue
        joint.set("frictionloss", "0")
        joint.set("armature", f"{JOINT_PARAMS[name][3]:g}")

    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    for child in list(act):
        act.remove(child)
    for joint in ACTION_JOINTS:
        kp, kv, eff, _ = JOINT_PARAMS[joint]
        ET.SubElement(act, "position", name=f"A_{joint[:-len('_joint')]}", joint=joint, kp=f"{kp:g}",
                      kv=f"{kv:g}", forcerange=f"{-eff:g} {eff:g}", ctrllimited="false")

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for child in list(contact):
        contact.remove(child)
    parent_of = {}
    body_names = []
    for b in root.iter("body"):
        body_names.append(b.get("name"))
        for c in b.findall("body"):
            parent_of[c.get("name")] = b.get("name")
    for a, b in itertools.combinations(body_names, 2):
        if parent_of.get(a) == b or parent_of.get(b) == a:
            continue
        ET.SubElement(contact, "exclude", body1=a, body2=b)

    option = root.find("option")
    option.set("integrator", "implicitfast")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def write(src_dir: Path) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('<?xml version="1.0"?>\n' + derive(src_dir), encoding="utf-8")
    print("wrote", OUT)


def compile_model():
    import mujoco

    return mujoco.MjModel.from_xml_path(str(OUT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/tmp/g1", help="dir with g1.xml and assets/")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        write(Path(args.src))
    import mujoco

    model = compile_model()
    print("nbody", model.nbody, "njnt", model.njnt, "nu", model.nu, "ngeom", model.ngeom, "nmesh", model.nmesh,
          "ncon_excl", model.nexclude)
    print("first actuators", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(3)])
