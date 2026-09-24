"""Run the Go2 AMP velocity policy in standalone MuJoCo from ONNX.

Policy discovery: latest log dir under logs/rsl_rl/go2_amp_locomotion/, newest
ONNX file (policy.onnx or export/*.onnx) by modification time.

Observation (actor "frame" term, 45-D, history 4 → 180-D):
  imu_ang_vel (3)
  projected_gravity (3)
  command (3)
  joint_pos - default (12)
  joint_vel (12)
  previous action (12)

Gamepad: left stick X/Y = lateral/forward gear, right stick X = yaw gear.
Gears quantize stick magnitude in steps of 0.5 m/s (yaw 0.5 rad/s).

Usage:
  python scripts/sim2sim_go2_gamepad.py
  python scripts/sim2sim_go2_gamepad.py --policy logs/rsl_rl/go2_amp_locomotion/<run>/policy.onnx
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
DEFAULT_XML = PROJECT_ROOT / "src" / "assets" / "robots" / "unitree_go2" / "xmls" / "scene_go2.xml"
LOGS_ROOT = PROJECT_ROOT / "logs" / "rsl_rl" / "go2_amp_locomotion"

ACTION_SIZE = 12
HISTORY_LENGTH = 4
FRAME_SIZE = 3 + 3 + 3 + ACTION_SIZE * 3  # 45
OBS_SIZE = FRAME_SIZE * HISTORY_LENGTH  # 180

GEAR_STEP = 0.5
MAX_FORWARD_GEAR = 4   # 2.0 m/s
MAX_BACKWARD_GEAR = 3  # 1.5 m/s
MAX_LATERAL_GEAR = 2   # 1.0 m/s
MAX_YAW_GEAR = 3       # 1.5 rad/s
TORQUE_LIMIT = 45.0


def find_latest_onnx() -> Path:
  """Newest metadata-bearing ONNX under go2_amp_locomotion logs."""
  if not LOGS_ROOT.is_dir():
    raise FileNotFoundError(f"No Go2 AMP logs under {LOGS_ROOT}")
  run_dirs = sorted(
    (d for d in LOGS_ROOT.iterdir() if d.is_dir()),
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
  if abs(stick) < deadzone:
    return 0.0
  n = math.ceil(abs(stick) * max_gear)
  n = max(1, min(n, max_gear))
  return math.copysign(n * GEAR_STEP, stick)


class GamepadCommand:
  def __init__(self, deadzone: float = 0.15, max_speed: float = 2.0, yaw_axis: int = 3) -> None:
    import os

    os_env_backup = None
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
    self.max_speed = float(max(0.5, max_speed))
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
    if self.joystick is None:
      return
    now = time.perf_counter()
    if now - self._rb_lb_cooldown < 0.25:
      return
    if self.get_button(7):
      self.max_speed = min(self.max_speed + GEAR_STEP, MAX_FORWARD_GEAR * GEAR_STEP)
      self._rb_lb_cooldown = now
      print(f"\n[INFO] Max forward speed -> {self.max_speed:.1f} m/s")
    elif self.get_button(6):
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
    rx = 0.0
    if self.joystick.get_numaxes() > self.yaw_axis:
      rx = self.joystick.get_axis(self.yaw_axis)
    if self.joystick.get_numhats() > 0:
      hat_x = self.joystick.get_hat(0)[0]
      if hat_x != 0:
        rx = float(hat_x)
    fwd_gears = max(1, int(round(self.max_speed / GEAR_STEP)))
    back_gears = max(1, int(round(fwd_gears * MAX_BACKWARD_GEAR / MAX_FORWARD_GEAR)))
    cmd_x = _gear(-ly, self.deadzone, fwd_gears if -ly >= 0 else back_gears)
    cmd_y = _gear(-lx, self.deadzone, MAX_LATERAL_GEAR)
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


class Go2Sim2Sim:
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
    assert len(self.joint_names) == ACTION_SIZE, (
      f"expected {ACTION_SIZE} joints, got {len(self.joint_names)}: {self.joint_names}"
    )
    self.default_q = _csv_floats(meta, "default_joint_pos")
    self.stiffness = _csv_floats(meta, "joint_stiffness")
    self.damping = _csv_floats(meta, "joint_damping")
    self.action_scale = _csv_floats(meta, "action_scale")

    self.model = mujoco.MjModel.from_xml_path(str(xml_path))
    self.data = mujoco.MjData(self.model)
    self.model.opt.timestep = sim_dt
    self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    qpos_ids, qvel_ids = [], []
    for name in self.joint_names:
      jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
      if jid < 0:
        raise ValueError(f"XML missing joint {name!r}")
      qpos_ids.append(int(self.model.jnt_qposadr[jid]))
      qvel_ids.append(int(self.model.jnt_dofadr[jid]))
    self.qpos_ids = np.asarray(qpos_ids, dtype=np.int32)
    self.qvel_ids = np.asarray(qvel_ids, dtype=np.int32)

    self.viz_scale = 0.5
    self.viz_height_cmd = 0.55
    self.viz_height_act = 0.40

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

  def _imu_ang_vel(self) -> np.ndarray:
    # go2.xml: <gyro name="imu_ang_vel" ...>
    return self.data.sensor("imu_ang_vel").data.copy()

  def _actor_frame(self) -> np.ndarray:
    joint_pos = self.data.qpos[self.qpos_ids]
    joint_vel = self.data.qvel[self.qvel_ids]
    frame = np.concatenate(
      (
        self._imu_ang_vel(),
        self._projected_gravity(),
        self.command,
        joint_pos - self.default_q,
        joint_vel,
        self.previous_action,
      )
    ).astype(np.float32)
    assert frame.shape[0] == FRAME_SIZE, frame.shape
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
    torque = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)
    self.data.qfrc_applied[:] = 0.0
    self.data.qfrc_applied[self.qvel_ids] = torque
    mujoco.mj_step(self.model, self.data)

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
    scn.ngeom = 0
    base_pos_w = self.data.qpos[0:3].copy()
    quat = self.data.qpos[3:7].copy()
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    base_mat_w = rot.reshape(3, 3)
    lin_vel_b = base_mat_w.T @ self.data.qvel[0:3].copy()
    ang_vel_b = self._imu_ang_vel()
    scale = self.viz_scale
    cmd = self.command

    def anchor_at(height: float) -> np.ndarray:
      return base_pos_w + np.asarray((0.0, 0.0, height))

    cmd_anchor = anchor_at(self.viz_height_cmd)
    act_anchor = anchor_at(self.viz_height_act)
    self._add_arrow(
      scn, cmd_anchor, cmd_anchor + base_mat_w @ np.asarray((cmd[0], cmd[1], 0.0)) * scale,
      (0.2, 0.2, 0.6, 0.6),
    )
    self._add_arrow(
      scn, cmd_anchor, cmd_anchor + np.asarray((0.0, 0.0, cmd[2] * scale)),
      (0.2, 0.6, 0.2, 0.6),
    )
    self._add_arrow(
      scn, act_anchor, act_anchor + base_mat_w @ np.asarray((lin_vel_b[0], lin_vel_b[1], 0.0)) * scale,
      (0.0, 0.6, 1.0, 0.7),
    )
    self._add_arrow(
      scn, act_anchor, act_anchor + np.asarray((0.0, 0.0, ang_vel_b[2] * scale)),
      (0.0, 1.0, 0.4, 0.7),
    )


def run(args: argparse.Namespace) -> None:
  policy_path = args.policy or find_latest_onnx()
  print(f"[INFO] Policy: {policy_path}")
  runner = Go2Sim2Sim(
    policy_path=policy_path,
    xml_path=args.xml.resolve(),
    sim_dt=args.sim_dt,
    decimation=args.decimation,
    initial_height=args.initial_height,
  )
  gamepad = (
    GamepadCommand(deadzone=args.deadzone, max_speed=args.max_speed, yaw_axis=args.yaw_axis)
    if args.gamepad
    else None
  )
  if gamepad is None:
    runner.set_command(np.asarray(args.command, dtype=np.float32))

  trunk_id = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
  print(
    f"[INFO] obs={OBS_SIZE} act={ACTION_SIZE} "
    f"control={1.0 / runner.control_dt:.0f}Hz physics={1.0 / args.sim_dt:.0f}Hz"
  )
  print("[INFO] Gear step 0.5; RB/LB change max speed; B resets, A+B exits.")

  viewer_context: Any = (
    contextlib.nullcontext(None) if args.headless else _launch_viewer(runner)
  )

  last_log = -1e9
  last_cmd = runner.command.copy()
  prev_b = False
  prev_ab = False
  with viewer_context as viewer:
    _setup_cam(viewer, trunk_id)
    for step in itertools.count():
      step_start = time.perf_counter()
      if gamepad is not None:
        runner.set_command(gamepad.get_command())
        b_now = gamepad.get_button(1)
        ab_now = gamepad.get_button(0) and gamepad.get_button(1)
        if ab_now and not prev_ab:
          print("\n[INFO] A+B pressed: exit")
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
        runner.draw_debug_arrows(viewer.user_scn)
        viewer.sync()
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
  print()


def _launch_viewer(runner: Go2Sim2Sim) -> Any:
  from mujoco import viewer as mujoco_viewer

  return mujoco_viewer.launch_passive(runner.model, runner.data)


def _setup_cam(viewer: Any, trunk_id: int) -> None:
  if viewer is None:
    return
  viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  viewer.cam.trackbodyid = trunk_id
  viewer.cam.distance = 2.0
  viewer.cam.azimuth = 120.0
  viewer.cam.elevation = -15.0


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Go2 AMP ONNX sim2sim with gamepad")
  parser.add_argument("--policy", type=Path, default=None, help="ONNX path; default: latest go2_amp_locomotion")
  parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
  parser.add_argument("--gamepad", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--deadzone", type=float, default=0.15)
  parser.add_argument("--max-speed", type=float, default=2.0)
  parser.add_argument("--yaw-axis", type=int, default=3)
  parser.add_argument("--command", type=float, nargs=3, default=(0.0, 0.0, 0.0))
  parser.add_argument("--sim-dt", type=float, default=0.005)
  parser.add_argument("--decimation", type=int, default=4)
  parser.add_argument("--initial-height", type=float, default=0.35)
  parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--log-interval", type=float, default=1.0)
  return parser.parse_args()


if __name__ == "__main__":
  run(parse_args())
