"""Run the G1 AMP velocity policy in standalone MuJoCo from ONNX.

Policy discovery: latest log dir under logs/rsl_rl/g1_amp_locomotion/, newest
ONNX file (policy.onnx or export/*.onnx) by modification time.

Observation (actor "frame" term, 96-D, history 4 → 384-D):
  imu_ang_vel (3, noise ±0.2 in training)
  projected_gravity (3, ±0.05)
  command (3)
  joint_pos - default (29, ±0.01)
  joint_vel - default (29, ±0.5)
  previous action (29)
  height_scan placeholder is NOT included (flat policy has 96 = 3+3+3+29+29+29)

Gamepad: left stick X/Y = lateral/forward gear, right stick X = yaw gear.
Gears quantize stick magnitude in steps of 0.5 m/s (yaw 0.5 rad/s).
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnx
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_XML = PROJECT_ROOT / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "scene_g1.xml"
LOGS_ROOT = PROJECT_ROOT / "logs" / "rsl_rl"

ACTION_SIZE = 29
HISTORY_LENGTH = 4
FRAME_SIZE = 96
OBS_SIZE = FRAME_SIZE * HISTORY_LENGTH

GEAR_STEP = 0.5  # m/s per gear; yaw rad/s per gear
MAX_FORWARD_GEAR = 6   # 3.0 m/s
MAX_BACKWARD_GEAR = 3  # 1.5 m/s
MAX_LATERAL_GEAR = 2   # 1.0 m/s
MAX_YAW_GEAR = 3       # 1.57 rad/s


def find_latest_onnx() -> Path:
  """Newest ONNX (with usable metadata) under logs/rsl_rl/*/*/.

  Skips export/*.onnx files that carry no metadata (they lack joint_names etc.).
  """
  run_dirs = sorted(
    (d for d in LOGS_ROOT.rglob("*") if d.is_dir()),
    key=lambda d: d.stat().st_mtime,
  )
  for run_dir in reversed(run_dirs):
    candidates = list(run_dir.glob("*.onnx")) + list(run_dir.glob("export/*.onnx"))
    for cand in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
      try:
        model = onnx.load(str(cand))
      except Exception:
        continue
      keys = {p.key for p in model.metadata_props}
      if {"joint_names", "default_joint_pos"} <= keys:
        return cand
  raise FileNotFoundError(f"No metadata-bearing .onnx found under {LOGS_ROOT}")


def _csv(metadata: dict[str, str], key: str) -> tuple[str, ...]:
  value = metadata.get(key)
  if value is None:
    raise ValueError(f"ONNX metadata missing {key!r}")
  return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_floats(metadata: dict[str, str], key: str) -> np.ndarray:
  return np.asarray([float(v) for v in _csv(metadata, key)], dtype=np.float32)


def _gear(stick: float, deadzone: float, max_gear: int) -> float:
  """Quantize stick magnitude to gear steps of GEAR_STEP."""
  if abs(stick) < deadzone:
    return 0.0
  n = math.ceil(abs(stick) * max_gear)
  n = max(1, min(n, max_gear))
  return math.copysign(n * GEAR_STEP, stick)


class GamepadCommand:
  """Left stick XY + right stick X → geared velocity command (pygame).

  RB/LB change the forward max speed (gears) live by ±0.5 m/s.
  """

  def __init__(self, deadzone: float = 0.15, max_speed: float = 3.0, yaw_axis: int = 3) -> None:
    os_env_backup = None
    import os

    if "SDL_VIDEODRIVER" not in os.environ:
      os_env_backup = os.environ.get("SDL_VIDEODRIVER")
      os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    if os_env_backup is not None:
      os.environ["SDL_VIDEODRIVER"] = os_env_backup
    pygame.display.init()
    pygame.joystick.init()
    self._pygame = pygame
    self.joystick = None
    self.deadzone = deadzone
    self.max_speed = float(max(0.5, max_speed))  # forward speed cap in m/s
    self.yaw_axis = yaw_axis
    self._rb_lb_cooldown = 0.0
    if pygame.joystick.get_count() > 0:
      self.joystick = pygame.joystick.Joystick(0)
      self.joystick.init()
      print(
        f"[INFO] Gamepad: {self.joystick.get_name()} "
        f"({self.joystick.get_numaxes()} axes, {self.joystick.get_numbuttons()} buttons, "
        f"{self.joystick.get_numhats()} hats)"
      )
      print(
        f"[INFO] Axes now: "
        f"{[round(self.joystick.get_axis(i), 2) for i in range(self.joystick.get_numaxes())]}"
        " (run scripts/pad_probe.py to find your yaw axis)"
      )
    else:
      print("[WARN] No gamepad detected; command stays (0,0,0)")
    print(f"[INFO] Max forward speed: {self.max_speed:.1f} m/s (RB +0.5 / LB -0.5)")

  def update(self) -> None:
    for _ in self._pygame.event.get():
      pass

  def get_button(self, idx: int) -> bool:
    if self.joystick is None:
      return False
    return bool(self.joystick.get_button(idx))

  def tick_speed_buttons(self) -> None:
    """RB raises / LB lowers the forward speed cap by one gear step."""
    if self.joystick is None:
      return
    now = time.perf_counter()
    if now - self._rb_lb_cooldown < 0.25:
      return
    if self.get_button(7):  # RB (see gamepad_controller mapping: 7=RB)
      self.max_speed = min(self.max_speed + GEAR_STEP, MAX_FORWARD_GEAR * GEAR_STEP)
      self._rb_lb_cooldown = now
      print(f"\n[INFO] Max forward speed -> {self.max_speed:.1f} m/s")
    elif self.get_button(6):  # LB
      self.max_speed = max(self.max_speed - GEAR_STEP, GEAR_STEP)
      self._rb_lb_cooldown = now
      print(f"\n[INFO] Max forward speed -> {self.max_speed:.1f} m/s")

  def get_command(self) -> np.ndarray:
    self.update()
    self.tick_speed_buttons()
    if self.joystick is None:
      return np.zeros(3, dtype=np.float32)
    lx = self.joystick.get_axis(0)
    ly = self.joystick.get_axis(1)
    # Right stick X: try configured axis; some 4-axis pads expose it elsewhere.
    rx = 0.0
    if self.joystick.get_numaxes() > self.yaw_axis:
      rx = self.joystick.get_axis(self.yaw_axis)
    # D-pad hat as yaw fallback (left/right).
    if self.joystick.get_numhats() > 0:
      hat_x = self.joystick.get_hat(0)[0]
      if hat_x != 0:
        rx = float(hat_x)

    fwd_gears = max(1, int(round(self.max_speed / GEAR_STEP)))
    back_gears = max(1, int(round(fwd_gears * MAX_BACKWARD_GEAR / MAX_FORWARD_GEAR)))
    cmd_x = _gear(-ly, self.deadzone, fwd_gears if -ly >= 0 else back_gears)
    cmd_y = _gear(-lx, self.deadzone, MAX_LATERAL_GEAR)
    # Stick right → positive yaw (no sign flip; previous -rx was inverted).
    cmd_yaw = _gear(rx, self.deadzone, MAX_YAW_GEAR)
    return np.asarray((cmd_x, cmd_y, cmd_yaw), dtype=np.float32)


class OnnxPolicy:
  def __init__(self, path: Path) -> None:
    self.model = onnx.load(str(path))
    self.metadata = {p.key: p.value for p in self.model.metadata_props}
    self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    out = self.session.run(None, {"obs": obs.reshape(1, -1).astype(np.float32)})[0]
    return np.asarray(out, dtype=np.float32).reshape(-1)


class G1Sim2Sim:
  def __init__(
    self,
    policy_path: Path,
    xml_path: Path,
    sim_dt: float,
    decimation: int,
    initial_height: float,
  ) -> None:
    self.policy = OnnxPolicy(policy_path)
    meta = self.policy.metadata
    self.joint_names = _csv(meta, "joint_names")
    assert len(self.joint_names) == ACTION_SIZE, len(self.joint_names)
    self.default_q = _csv_floats(meta, "default_joint_pos")
    self.stiffness = _csv_floats(meta, "joint_stiffness")
    self.damping = _csv_floats(meta, "joint_damping")
    self.action_scale = _csv_floats(meta, "action_scale")

    self.model = mujoco.MjModel.from_xml_path(str(xml_path))
    self.data = mujoco.MjData(self.model)
    self.model.opt.timestep = sim_dt
    self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    self._sim_dt = sim_dt

    qpos_ids, qvel_ids = [], []
    for name in self.joint_names:
      jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
      if jid < 0:
        raise ValueError(f"XML missing joint {name!r}")
      qpos_ids.append(int(self.model.jnt_qposadr[jid]))
      qvel_ids.append(int(self.model.jnt_dofadr[jid]))
    self.qpos_ids = np.asarray(qpos_ids, dtype=np.int32)
    self.qvel_ids = np.asarray(qvel_ids, dtype=np.int32)

    # Debug-arrow drawing state (user_scn arrows, same as mjlab play viewer).
    # Anchors hover well above the robot head so arrows are clearly visible.
    self.viz_scale = 0.5
    self.viz_height_cmd = 1.0  # cmd arrows anchor this far above the pelvis
    self.viz_height_act = 0.8  # actual-velocity arrows anchor

    self.decimation = decimation
    self.command = np.zeros(3, dtype=np.float32)
    self.previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
    self.history = np.zeros((HISTORY_LENGTH, FRAME_SIZE), dtype=np.float32)
    self.default_height = float(initial_height)
    self._reset(initial_height)

  @property
  def control_dt(self) -> float:
    return float(self.model.opt.timestep * self.decimation)

  def _reset(self, height: float) -> None:
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[0:3] = (0.0, 0.0, height)
    self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    self.data.qpos[self.qpos_ids] = self.default_q
    mujoco.mj_forward(self.model, self.data)
    self.previous_action.fill(0.0)
    self.history[:] = self._actor_frame()

  def set_command(self, command: np.ndarray) -> None:
    self.command[:] = command

  def _projected_gravity(self) -> np.ndarray:
    quat = self.data.qpos[3:7]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    gravity_w = np.asarray((0.0, 0.0, -1.0))
    return rot.reshape(3, 3).T @ gravity_w

  def _actor_frame(self) -> np.ndarray:
    ang_vel = self.data.sensor("imu_gyro").data.copy()
    joint_pos = self.data.qpos[self.qpos_ids]
    joint_vel = self.data.qvel[self.qvel_ids]
    frame = np.concatenate(
      (
        ang_vel,
        self._projected_gravity(),
        self.command,
        joint_pos - self.default_q,
        joint_vel,
        self.previous_action,
      )
    ).astype(np.float32)
    return frame

  def update_policy(self) -> np.ndarray:
    mujoco.mj_forward(self.model, self.data)
    self.history[:-1] = self.history[1:]
    self.history[-1] = self._actor_frame()
    action = self.policy(self.history.reshape(-1))
    self.previous_action = action.copy()
    self.target_q = self.default_q + self.action_scale * action
    return action

  def physics_step(self) -> None:
    joint_pos = self.data.qpos[self.qpos_ids]
    joint_vel = self.data.qvel[self.qvel_ids]
    torque = self.stiffness * (self.target_q - joint_pos) - self.damping * joint_vel
    # Clamp to a sane torque range to avoid explosive first steps.
    torque = np.clip(torque, -88.0, 88.0)
    self.data.qfrc_applied[:] = 0.0
    self.data.qfrc_applied[self.qvel_ids] = torque
    mujoco.mj_step(self.model, self.data)

  # ------------------------------------------------------------------
  # Debug arrows — ported from mjlab UniformVelocityCommand._debug_vis_impl
  # (drawn into viewer.user_scn as mjGEOM_ARROW decorations each frame).
  # ------------------------------------------------------------------

  def _add_arrow(
    self,
    scn: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[float, float, float, float],
  ) -> None:
    if scn.ngeom >= scn.maxgeom:
      return
    scn.ngeom += 1
    geom = scn.geoms[scn.ngeom - 1]
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    mujoco.mjv_initGeom(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      size=np.zeros(3),
      pos=np.zeros(3),
      mat=np.zeros(9),
      rgba=np.asarray(color, dtype=np.float32),
    )
    mujoco.mjv_connector(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      width=0.015,
      from_=np.asarray(start, dtype=np.float64),
      to=np.asarray(end, dtype=np.float64),
    )

  def draw_debug_arrows(self, scn: mujoco.MjvScene) -> None:
    """Draw cmd/actual linear+angular velocity arrows, same style as play.

    cmd linear  blue   (0.2, 0.2, 0.6, 0.6)
    cmd angular green  (0.2, 0.6, 0.2, 0.6)
    act linear  cyan   (0.0, 0.6, 1.0, 0.7)
    act angular lgreen (0.0, 1.0, 0.4, 0.7)
    """
    scn.ngeom = 0
    base_pos_w = self.data.qpos[0:3].copy()
    pelvis_quat = self.data.qpos[3:7].copy()
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, pelvis_quat)
    base_mat_w = rot.reshape(3, 3)
    # Body-frame linear velocity: R^T v_world.
    lin_vel_w = self.data.qvel[0:3].copy()
    lin_vel_b = base_mat_w.T @ lin_vel_w
    # Body-frame angular velocity: gyro sensor is already body-frame.
    ang_vel_b = self.data.sensor("imu_gyro").data.copy()

    scale = self.viz_scale
    cmd = self.command

    def anchor_at(height: float) -> np.ndarray:
      """Anchor directly above the pelvis at the given world-space height."""
      return base_pos_w + np.asarray((0.0, 0.0, height))

    # Two stacked anchor levels so cmd and actual arrows don't overlap.
    cmd_anchor = anchor_at(self.viz_height_cmd)
    act_anchor = anchor_at(self.viz_height_act)

    # Command linear velocity (body frame xy → world).
    cmd_lin_end = cmd_anchor + base_mat_w @ (
      np.asarray((cmd[0], cmd[1], 0.0)) * scale
    )
    self._add_arrow(scn, cmd_anchor, cmd_lin_end, (0.2, 0.2, 0.6, 0.6))

    # Command angular velocity (yaw only, drawn as vertical arrow).
    cmd_ang_end = cmd_anchor + np.asarray((0.0, 0.0, cmd[2] * scale))
    self._add_arrow(scn, cmd_anchor, cmd_ang_end, (0.2, 0.6, 0.2, 0.6))

    # Actual linear velocity.
    act_lin_end = act_anchor + base_mat_w @ (
      np.asarray((lin_vel_b[0], lin_vel_b[1], 0.0)) * scale
    )
    self._add_arrow(scn, act_anchor, act_lin_end, (0.0, 0.6, 1.0, 0.7))

    # Actual angular velocity (yaw).
    act_ang_end = act_anchor + np.asarray((0.0, 0.0, ang_vel_b[2] * scale))
    self._add_arrow(scn, act_anchor, act_ang_end, (0.0, 1.0, 0.4, 0.7))

  def standing_fall_check(self) -> bool:
    return self.data.qpos[2] < 0.3


def run(args: argparse.Namespace) -> None:
  policy_path = args.policy or find_latest_onnx()
  print(f"[INFO] Policy: {policy_path}")
  runner = G1Sim2Sim(
    policy_path=policy_path,
    xml_path=args.xml.resolve(),
    sim_dt=args.sim_dt,
    decimation=args.decimation,
    initial_height=args.initial_height,
  )
  gamepad = GamepadCommand(deadzone=args.deadzone, max_speed=args.max_speed, yaw_axis=args.yaw_axis) if args.gamepad else None
  torso_id = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

  print(
    f"[INFO] obs={OBS_SIZE} act={ACTION_SIZE} "
    f"control={1.0 / runner.control_dt:.0f}Hz physics={1.0 / args.sim_dt:.0f}Hz"
  )
  print(f"[INFO] Gear step {GEAR_STEP}; RB/LB change max speed; B resets, A+B exits.")

  viewer_context: Any = (
    contextlib.nullcontext(None)
    if args.headless
    else _launch_viewer(runner)
  )

  last_log = -1e9
  last_cmd = runner.command.copy()
  prev_b = False
  prev_ab = False
  with viewer_context as viewer:
    _setup_cam(viewer, runner, torso_id)
    for step in itertools.count():
      step_start = time.perf_counter()
      if gamepad is not None:
        runner.set_command(gamepad.get_command())
        # Edge-triggered buttons with visible feedback.
        b_now = gamepad.get_button(1)
        ab_now = gamepad.get_button(0) and gamepad.get_button(1)
        if ab_now and not prev_ab:
          print("\n[INFO] A+B pressed: exit")
          prev_ab = ab_now
          break
        if b_now and not prev_b:
          print("\n[INFO] B pressed: reset")
          runner._reset(runner.default_height)
        prev_b = b_now
        prev_ab = ab_now
      if step % runner.decimation == 0:
        runner.update_policy()
      runner.physics_step()
      if viewer is not None:
        if not viewer.is_running():
          break
        # Draw play-style cmd/actual velocity arrows into the viewer scene.
        runner.draw_debug_arrows(viewer.user_scn)
        viewer.sync()
      # Print immediately when the command changes; else on log interval.
      cmd_changed = not np.array_equal(runner.command, last_cmd)
      if cmd_changed or runner.data.time - last_log >= args.log_interval:
        print(
          f"\r[SIM] t={runner.data.time:7.2f}s "
          f"cmd=({runner.command[0]:+.1f},{runner.command[1]:+.1f},"
          f"{runner.command[2]:+.1f}) max={gamepad.max_speed if gamepad else 0.0:.1f} "
          f"h={runner.data.qpos[2]:.3f}   ",
          end="",
          flush=True,
        )
        last_log = float(runner.data.time)
        last_cmd = runner.command.copy()
      if args.realtime:
        remaining = step_start + runner.model.opt.timestep - time.perf_counter()
        if remaining > 0.0:
          time.sleep(remaining)


def _launch_viewer(runner: G1Sim2Sim) -> Any:
  from mujoco import viewer as mujoco_viewer

  return mujoco_viewer.launch_passive(runner.model, runner.data)


def _setup_cam(viewer: Any, runner: G1Sim2Sim, torso_id: int) -> None:
  if viewer is None:
    return
  viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  viewer.cam.trackbodyid = torso_id
  viewer.cam.distance = 3.5
  viewer.cam.azimuth = 120.0
  viewer.cam.elevation = -10.0


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="G1 AMP ONNX sim2sim with gamepad")
  parser.add_argument("--policy", type=Path, default=None, help="ONNX path; default: latest")
  parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
  parser.add_argument("--gamepad", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--deadzone", type=float, default=0.15)
  parser.add_argument("--max-speed", type=float, default=3.0, help="Initial forward max speed (m/s); RB/LB change it live")
  parser.add_argument("--yaw-axis", type=int, default=3, help="Gamepad axis used for yaw (USB pads: 3=right stick X; fallback: D-pad left/right)")
  parser.add_argument("--command", type=float, nargs=3, default=(0.0, 0.0, 0.0))
  parser.add_argument("--sim-dt", type=float, default=0.005)
  parser.add_argument("--decimation", type=int, default=4)
  parser.add_argument("--initial-height", type=float, default=0.75)
  parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--log-interval", type=float, default=1.0)
  return parser.parse_args()


if __name__ == "__main__":
  run(parse_args())
