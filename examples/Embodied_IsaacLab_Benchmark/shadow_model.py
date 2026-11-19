"""Shadow Hand in-hand repose model for the Isaac-Repose-Cube-Shadow-Direct-v0 benchmark.

Built from MuJoCo Menagerie's shadow_hand/right_hand.xml (vendored under assets/menagerie_shadow_hand, see its
PROVENANCE.json) with the settings of Isaac Lab v2.3.0's SHADOW_HAND_CFG (source/isaaclab_assets/.../robots/
shadow_hand.py) and the repose scene (direct/shadow_hand/shadow_hand_env_cfg.py), derived in memory:

  * visual meshes removed; collision meshes replaced by capsules fitted to the mesh bounds (the fingertip and
    thumb distal meshes span 0.0086 / 0.0106 m in x, 0.029 / 0.032 m in z; the forearm mesh is a 0.15 m cylinder
    behind the palm). hipPhysics' lab build takes primitive collision only.
  * the hand is mounted under body 'hand_root' at ``HAND_ROOT_POS`` with orientation ``HAND_ROOT_QUAT`` (palm up);
    Isaac's USD root frame is not reproduced.
  * gravity off for the hand (disable_gravity=True): gravcomp 1 on every hand body.
  * actuators: the 20 Isaac actuated joints map to Menagerie actuators by role. Isaac drives the wrist, thumb and
    the abduction / MCP joints with PD drives (wrist kp 5 kv 0.5, other joints kp 1 kv 0.1) and force limits
    (effort_limit_sim). Isaac's distal DIP is coupled to the PIP through a fixed tendon, so the Menagerie tendon
    actuators (rh_FFJ0, rh_MFJ0, rh_RFJ0, rh_LFJ0) take the Isaac PIP-actuator slots (FFJ1, MFJ1, RFJ1, LFJ1).
    Tendon actuator targets are tendon lengths (PIP + DIP), with the range the sum of the two joint ranges.
  * object: Isaac's DexCube (dex_cube_instanceable.usd, no scale): 0.06 m cube, density 567 kg/m^3 (mass
    0.1225 kg), friction 1.0 (sim physics_material static = dynamic = 1.0).
  * option: integrator implicitfast, elliptic cone, impratio 10 (the Isaac newton_mjwarp preset, as the Allegro
    port). The timestep comes from the env (dt 1/120, one physics substep per Isaac physics step).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ASSET = Path(__file__).resolve().parent / "assets" / "menagerie_shadow_hand" / "right_hand.xml"

HAND_ROOT_POS = (0.0, 0.0, 0.5)  # SHADOW_HAND_CFG.init_state.pos
# Palm-up mount. Found by palm_frame_report() on the upstream hand: the fingers point along world +y and the palm
# normal along world +z, so the in-hand object sits above the palm.
HAND_ROOT_QUAT = (0.7071068, 0.0, 0.0, 0.7071068)
CUBE_EDGE = 0.06
CUBE_DENSITY = 567.0
CUBE_MASS = CUBE_DENSITY * CUBE_EDGE**3
CUBE_HALF = CUBE_EDGE / 2
CUBE_FRICTION = "1 0.005 0.0001"
# The palm top is at z 0.522 (palm box, local -y side, which the flexion test showed faces up) over the palm centre
# (x 0, y 0.236..0.334). Isaac starts the cube 0.04 m above its in-hand pose; the in-hand pose here is the palm rest
# (0, 0.285, 0.552), so the start is (0, 0.285, 0.592).
CUBE_IN_HAND_POS = (0.0, 0.285, 0.552)
CUBE_DEFAULT_POS = (0.0, 0.285, 0.592)

# Action order (the 20 Isaac actuated joints; ShadowHandEnvCfg.actuated_joint_names minus FF/MF/RF/LF J0 knuckles,
# which are the Menagerie tendons). (action name, Menagerie joint or tendon, kind, kp, kv, effort limit N m)
# Isaac joint -> Menagerie: WRJ1 = wrist flex (rh_WRJ2), WRJ0 = wrist deviation (rh_WRJ1). FFJ3 = abduction
# (rh_FFJ4), FFJ2 = MCP flex (rh_FFJ3), FFJ1 = PIP, which the fixed tendon rh_FFJ0 drives with the DIP. LFJ4 =
# metacarpal (rh_LFJ5), LFJ3 = abduction (rh_LFJ4), LFJ2 = MCP flex (rh_LFJ3), LFJ1 = PIP (rh_LFJ0 tendon). THJ4..THJ0
# = rh_THJ5..rh_THJ1.
ACTUATORS = (
    ("WRJ1", "rh_WRJ2", "joint", 5.0, 0.5, 4.785),
    ("WRJ0", "rh_WRJ1", "joint", 5.0, 0.5, 2.175),
    ("FFJ3", "rh_FFJ4", "joint", 1.0, 0.1, 0.9),
    ("FFJ2", "rh_FFJ3", "joint", 1.0, 0.1, 0.9),
    ("FFJ1", "rh_FFJ0", "tendon", 1.0, 0.1, 0.7245),
    ("MFJ3", "rh_MFJ4", "joint", 1.0, 0.1, 0.9),
    ("MFJ2", "rh_MFJ3", "joint", 1.0, 0.1, 0.9),
    ("MFJ1", "rh_MFJ0", "tendon", 1.0, 0.1, 0.7245),
    ("RFJ3", "rh_RFJ4", "joint", 1.0, 0.1, 0.9),
    ("RFJ2", "rh_RFJ3", "joint", 1.0, 0.1, 0.9),
    ("RFJ1", "rh_RFJ0", "tendon", 1.0, 0.1, 0.7245),
    ("LFJ4", "rh_LFJ5", "joint", 1.0, 0.1, 0.9),
    ("LFJ3", "rh_LFJ4", "joint", 1.0, 0.1, 0.9),
    ("LFJ2", "rh_LFJ3", "joint", 1.0, 0.1, 0.9),
    ("LFJ1", "rh_LFJ0", "tendon", 1.0, 0.1, 0.7245),
    ("THJ4", "rh_THJ5", "joint", 1.0, 0.1, 2.3722),
    ("THJ3", "rh_THJ4", "joint", 1.0, 0.1, 1.45),
    ("THJ2", "rh_THJ3", "joint", 1.0, 0.1, 0.99),
    ("THJ1", "rh_THJ2", "joint", 1.0, 0.1, 0.99),
    ("THJ0", "rh_THJ1", "joint", 1.0, 0.1, 0.81),
)
# Tendon -> the two coupled joints (PIP, DIP) of the upstream fixed tendons.
TENDON_JOINTS = {
    "rh_FFJ0": ("rh_FFJ2", "rh_FFJ1"),
    "rh_MFJ0": ("rh_MFJ2", "rh_MFJ1"),
    "rh_RFJ0": ("rh_RFJ2", "rh_RFJ1"),
    "rh_LFJ0": ("rh_LFJ2", "rh_LFJ1"),
}
# The 24 hand joints in upstream order (the articulation's DOFs; Isaac's num_hand_dofs is 24 too).
JOINT_NAMES = (
    "rh_WRJ2", "rh_WRJ1",
    "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
    "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
    "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
    "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
    "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
)
# Fingertip bodies in Isaac's order (ShadowHandEnvCfg.fingertip_body_names).
FINGERTIP_BODIES = ("rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal")
# Collision capsules that replace the upstream mesh collisions: body -> (fromto, radius) in that body's frame.
MESH_CAPSULES = {
    "rh_ffdistal": ("0 0 0.012", "0 0 0.025", 0.008),
    "rh_mfdistal": ("0 0 0.012", "0 0 0.025", 0.008),
    "rh_rfdistal": ("0 0 0.012", "0 0 0.025", 0.008),
    "rh_lfdistal": ("0 0 0.012", "0 0 0.025", 0.008),
    "rh_thdistal": ("0 0 0.0128", "0 0 0.0255", 0.0095),
    "rh_forearm": ("0 0 0.02", "0 0 0.16", 0.07),
}


def _upstream_tree() -> ET.Element:
    return ET.parse(ASSET).getroot()


def _strip_visuals_and_meshes(root: ET.Element) -> None:
    """Remove mesh assets and every mesh geom (visual and collision); add the capsules of MESH_CAPSULES.

    Every mesh geom of the upstream hand is either visual (class plastic_visual) or the collision mesh of one of the
    bodies in MESH_CAPSULES, so removing them all and adding those capsules leaves no mesh in the model."""
    for asset in root.findall("asset"):
        root.remove(asset)
    root.find("compiler").attrib.pop("meshdir", None)
    for parent in root.iter():
        for geom in list(parent):
            if geom.tag == "geom" and (geom.get("mesh") or (geom.get("class") or "").endswith("_visual")):
                parent.remove(geom)
    bodies = {b.get("name"): b for b in root.iter("body")}
    for name, (fromto_a, fromto_b, radius) in MESH_CAPSULES.items():
        ET.SubElement(bodies[name], "geom", name=f"{name}_capsule", type="capsule", size=str(radius),
                      fromto=f"{fromto_a} {fromto_b}", friction=CUBE_FRICTION, group="3")
    for geom in root.find("worldbody").iter("geom"):
        if geom.get("type") == "mesh" or geom.get("mesh"):
            raise ValueError(f"mesh geom in worldbody left behind (body geom with class {geom.get('class')})")
    for tendon in root.iter("fixed"):
        tendon.set("damping", "0.1")  # Isaac fixed_tendons_props damping 0.1


def _mount_and_object(root: ET.Element) -> None:
    """Mount the hand, and add gravity-free hand bodies and the Isaac DexCube object."""
    worldbody = root.find("worldbody")
    forearm = None
    for body in worldbody.iter("body"):
        if body.get("name") == "rh_forearm":
            forearm = body
    worldbody.remove(forearm)
    mount = ET.SubElement(worldbody, "body", name="hand_root",
                          pos=" ".join(map(str, HAND_ROOT_POS)), quat=" ".join(map(str, HAND_ROOT_QUAT)))
    mount.append(forearm)
    for body in forearm.iter("body"):
        body.set("gravcomp", "1")
    obj = ET.SubElement(worldbody, "body", name="object", pos=" ".join(map(str, CUBE_DEFAULT_POS)))
    ET.SubElement(obj, "freejoint", name="object_root")
    ET.SubElement(obj, "geom", name="object", type="box", size=f"{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}",
                  mass=f"{CUBE_MASS:.6f}", friction=CUBE_FRICTION)


def _friction_defaults(root: ET.Element) -> None:
    """Isaac's sim physics_material (static = dynamic 1.0) applies to every hand collision geom."""
    for geom in root.iter("geom"):
        if geom.get("group") == "3" and geom.get("friction") is None and geom.get("type") != "mesh":
            geom.set("friction", CUBE_FRICTION)


def _joint_ranges(root: ET.Element) -> dict:
    import mujoco

    probe = ET.fromstring(ET.tostring(root))
    for act in list(probe.iter("actuator")):
        for child in list(act):
            act.remove(child)
    model = mujoco.MjModel.from_xml_string(ET.tostring(probe, encoding="unicode"))
    out = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name:
            out[name] = (float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1]))
    return out


def _actuators(root: ET.Element, ranges: dict) -> dict:
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    for child in list(act):
        act.remove(child)
    spans = {}
    for name, target, kind, kp, kv, effort in ACTUATORS:
        if kind == "joint":
            lo, hi = ranges[target]
            ET.SubElement(act, "position", name=f"A_{name}", joint=target, kp=str(kp), kv=str(kv),
                          ctrlrange=f"{lo} {hi}", forcerange=f"{-effort} {effort}")
        else:
            a, b = TENDON_JOINTS[target]
            lo = ranges[a][0] + ranges[b][0]
            hi = ranges[a][1] + ranges[b][1]
            ET.SubElement(act, "position", name=f"A_{name}", tendon=target, kp=str(kp), kv=str(kv),
                          ctrlrange=f"{lo} {hi}", forcerange=f"{-effort} {effort}")
        spans[name] = (lo, hi)
    return spans


def mjcf() -> str:
    """The derived Shadow Hand repose scene as MJCF text."""
    root = _upstream_tree()
    _strip_visuals_and_meshes(root)
    _friction_defaults(root)
    _mount_and_object(root)
    option = root.find("option")
    option.attrib.update(integrator="implicitfast", cone="elliptic", impratio="10")
    ranges = _joint_ranges(root)
    _actuators(root, ranges)
    return ET.tostring(root, encoding="unicode")


def action_ranges() -> np.ndarray:
    """(20, 2) control range of each action in ACTUATORS order (joint limits, or the tendon length range)."""
    root = _upstream_tree()
    _strip_visuals_and_meshes(root)
    _mount_and_object(root)
    ranges = _joint_ranges(root)
    out = []
    for _name, target, kind, _kp, _kv, _effort in ACTUATORS:
        if kind == "joint":
            out.append(ranges[target])
        else:
            a, b = TENDON_JOINTS[target]
            out.append((ranges[a][0] + ranges[b][0], ranges[a][1] + ranges[b][1]))
    return np.asarray(out, dtype=np.float64)


def dof_to_ctrl_matrix() -> np.ndarray:
    """(20, 24) matrix M with ctrl = M @ qpos over the 24 joints: a joint row selects its joint, a tendon row sums
    its PIP and DIP (the tendon length)."""
    index = {n: i for i, n in enumerate(JOINT_NAMES)}
    mat = np.zeros((len(ACTUATORS), len(JOINT_NAMES)))
    for row, (_name, target, kind, _kp, _kv, _effort) in enumerate(ACTUATORS):
        if kind == "joint":
            mat[row, index[target]] = 1.0
        else:
            for joint in TENDON_JOINTS[target]:
                mat[row, index[joint]] = 1.0
    return mat


def compile_model():
    import mujoco

    return mujoco.MjModel.from_xml_string(mjcf())


def palm_frame_report() -> str:
    """Diagnostic: palm and fingertip directions for the mount chosen in HAND_ROOT_QUAT (world frame)."""
    import mujoco

    model = compile_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    palm = data.xpos[body("rh_palm")]
    tips = np.array([data.xpos[body(n)] for n in FINGERTIP_BODIES])
    rot = data.xmat[body("rh_palm")].reshape(3, 3)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    return (
        f"palm pos {palm.round(4).tolist()} palm z-axis {rot[:, 2].round(3).tolist()} "
        f"grasp_site {data.site_xpos[site].round(4).tolist()} "
        f"fingertip centroid {tips.mean(0).round(4).tolist()}"
    )


if __name__ == "__main__":
    print(palm_frame_report())
    print(action_ranges())
