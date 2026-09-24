"""Interactive motion-CSV viewer with slow-motion playback.

Visualizes raw motion CSVs in a plain MuJoCo passive viewer — no mjlab/GPU needed.

G1 CSV (36 cols @ typically 30 Hz)::
  cols 0-2   root position (world, m)
  cols 3-6   root quaternion **xyzw**
  cols 7-35  29 G1 joint angles (rad)

Go2 CSV (19 cols @ typically 25 Hz, from ``mocap_motions_go2``)::
  cols 0-2   root position (world, m)
  cols 3-6   root quaternion **xyzw** (Isaac; converted to wxyz for MuJoCo)
  cols 7-18  12 Go2 joints FL/FR/RL/RR × (hip, thigh, calf)

Controls (keyboard, with the viewer window focused):
  Space        pause / resume
  ← / →        step one frame back / forward (while paused or playing)
  ↑ / ↓        playback speed ×2 / ÷2  (0.02x – 16x, default 1.0)
  R            restart from frame 0
  L            toggle loop
  N / P        next / previous CSV in the folder
  Q / Esc      quit

Usage:
  # G1 (default folder motion_data_csv/amp)
  python scripts/play_motion_csv.py
  python scripts/play_motion_csv.py --file <g1_motion.csv>

  # Go2 mocap CSVs
  python scripts/play_motion_csv.py --robot go2
  python scripts/play_motion_csv.py --robot go2 --file src/assets/motions/go2/mocap_csv/forward_hip_sym.csv
  python scripts/play_motion_csv.py --robot go2 --speed 0.25
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer as mujoco_viewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

G1_JOINT_NAMES = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]

GO2_JOINT_NAMES = [
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

SPEEDS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

# Keycodes: ASCII for letters/space/esc; arrows use X11 keysym values as
# delivered by mujoco's key_callback on Linux.
K_SPACE = 32
K_ESC = 65307
K_LEFT, K_RIGHT, K_UP, K_DOWN = 65361, 65363, 65362, 65364


@dataclass(frozen=True)
class RobotProfile:
  name: str
  xml: Path
  joint_names: tuple[str, ...]
  track_body: str
  default_dir: Path
  default_fps: float
  min_cols: int
  cam_distance: float


PROFILES: dict[str, RobotProfile] = {
  "g1": RobotProfile(
    name="g1",
    xml=PROJECT_ROOT / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "scene_g1.xml",
    joint_names=tuple(G1_JOINT_NAMES),
    track_body="pelvis",
    default_dir=PROJECT_ROOT / "motion_data_csv" / "amp",
    default_fps=30.0,
    min_cols=3 + 4 + len(G1_JOINT_NAMES),
    cam_distance=3.0,
  ),
  "go2": RobotProfile(
    name="go2",
    xml=PROJECT_ROOT / "src" / "assets" / "robots" / "unitree_go2" / "xmls" / "go2.xml",
    joint_names=tuple(GO2_JOINT_NAMES),
    track_body="trunk",
    default_dir=PROJECT_ROOT / "src" / "assets" / "motions" / "go2" / "mocap_csv",
    default_fps=25.0,  # mocap FrameDuration=0.04
    min_cols=3 + 4 + len(GO2_JOINT_NAMES),
    cam_distance=2.0,
  ),
}


def _infer_robot(csv_path: Path) -> str:
  raw = np.loadtxt(csv_path, delimiter=",", max_rows=1)
  ncols = int(raw.shape[0]) if raw.ndim == 1 else int(raw.shape[1])
  if ncols >= PROFILES["g1"].min_cols:
    return "g1"
  if ncols >= PROFILES["go2"].min_cols:
    return "go2"
  raise ValueError(f"{csv_path.name}: unexpected column count {ncols}")


class MotionPlayer:
  def __init__(
    self,
    csv_path: Path,
    profile: RobotProfile,
    fps: float,
    speed: float,
    loop: bool = True,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
  ):
    self.name = csv_path.stem
    self.profile = profile
    raw = np.loadtxt(csv_path, delimiter=",")
    n_joints = len(profile.joint_names)
    need = 3 + 4 + n_joints
    if raw.ndim != 2 or raw.shape[1] < need:
      raise ValueError(f"{csv_path.name}: expected >={need} columns, got {raw.shape}")
    self.frames = raw.shape[0]
    self.fps = fps
    self.duration = self.frames / self.fps
    self.root_pos = raw[:, 0:3].copy()
    # CSV stores Isaac/AMP xyzw; MuJoCo freejoint wants wxyz.
    self.root_quat = raw[:, 3:7][:, [3, 0, 1, 2]].copy()
    self.joint_pos = raw[:, 7:7 + n_joints].copy()

    if model is None:
      self.model = mujoco.MjModel.from_xml_path(str(profile.xml))
      self.data = mujoco.MjData(self.model)
    else:
      self.model = model
      self.data = data if data is not None else mujoco.MjData(model)

    qpos_ids, qvel_ids = [], []
    for name in profile.joint_names:
      jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
      if jid < 0:
        raise ValueError(f"XML missing joint {name!r}")
      qpos_ids.append(int(self.model.jnt_qposadr[jid]))
      qvel_ids.append(int(self.model.jnt_dofadr[jid]))
    self.qpos_ids = np.asarray(qpos_ids)
    self.qvel_ids = np.asarray(qvel_ids)

    self.frame = 0
    self.paused = False
    self.speed = min(SPEEDS, key=lambda s: abs(s - speed))
    self.loop = loop

  def apply_frame(self, frame: int) -> None:
    frame = int(np.clip(frame, 0, self.frames - 1))
    self.frame = frame
    self.data.qpos[0:3] = self.root_pos[frame]
    self.data.qpos[3:7] = self.root_quat[frame]
    self.data.qpos[self.qpos_ids] = self.joint_pos[frame]
    self.data.qvel[:] = 0.0
    mujoco.mj_forward(self.model, self.data)

  def step_frame(self, delta: int) -> None:
    nxt = self.frame + delta
    if nxt >= self.frames:
      nxt = 0 if self.loop else self.frames - 1
    elif nxt < 0:
      nxt = self.frames - 1 if self.loop else 0
    self.apply_frame(nxt)

  def status(self) -> str:
    state = "PAUSED" if self.paused else "RUN"
    return (
      f"[{self.frame + 1:4d}/{self.frames}] t={self.frame / self.fps:6.2f}s "
      f"| {state} x{self.speed:g} {'loop' if self.loop else 'once'}"
    )


def _pick_file(default_dir: Path) -> Path | None:
  """Numeric file picker when no --file is given."""
  files = sorted(default_dir.glob("*.csv"))
  if not files:
    print(f"No CSVs found in {default_dir}")
    return None
  if len(files) == 1:
    return files[0]
  print("Available motions:")
  for i, f in enumerate(files):
    print(f"  [{i:2d}] {f.name}")
  while True:
    try:
      raw = input(f"Select file index [0-{len(files) - 1}, Enter=0]: ").strip()
      idx = int(raw) if raw else 0
      if 0 <= idx < len(files):
        return files[idx]
    except ValueError:
      pass
    print("Invalid input, try again.")


def main() -> None:
  parser = argparse.ArgumentParser(description="Motion CSV viewer with slow-mo (G1 / Go2)")
  wrapper = parser.add_mutually_exclusive_group()
  wrapper.add_argument("--file", type=Path, default=None, help="CSV to play (default: browse)")
  wrapper.add_argument("--dir", type=Path, default=None, help="Folder to browse")
  parser.add_argument(
    "--robot",
    choices=("auto", "g1", "go2"),
    default="auto",
    help="Robot profile (auto-detect from CSV column count)",
  )
  parser.add_argument("--xml", type=Path, default=None, help="Override MuJoCo XML")
  parser.add_argument("--fps", type=float, default=None, help="CSV frame rate (robot default if omitted)")
  parser.add_argument("--speed", type=float, default=1.0, help="Initial speed factor (try 0.25 for slow-mo)")
  parser.add_argument("--start", type=int, default=0, help="Start frame index")
  parser.add_argument("--no-loop", action="store_true", help="Play once instead of looping")
  args = parser.parse_args()

  # Resolve robot / search directory before loading the first CSV.
  if args.robot != "auto":
    robot = args.robot
  elif args.file is not None:
    robot = _infer_robot(args.file)
  elif args.dir is not None:
    sample = next(iter(sorted(args.dir.glob("*.csv"))), None)
    robot = _infer_robot(sample) if sample is not None else "g1"
  else:
    # No file/dir: prefer go2 mocap folder if user likely wants that; else g1.
    # Keep historical default = g1 amp folder when --robot auto with no args.
    robot = "g1"

  profile = PROFILES[robot]
  if args.xml is not None:
    # dataclasses.replace-like override via a shallow copy of fields.
    profile = RobotProfile(
      name=profile.name,
      xml=args.xml,
      joint_names=profile.joint_names,
      track_body=profile.track_body,
      default_dir=profile.default_dir,
      default_fps=profile.default_fps,
      min_cols=profile.min_cols,
      cam_distance=profile.cam_distance,
    )

  search_dir = (
    args.file.parent if args.file is not None
    else args.dir if args.dir is not None
    else profile.default_dir
  )
  files = sorted(search_dir.glob("*.csv"))
  if not files:
    raise SystemExit(f"No CSV files in {search_dir}")
  if args.file is not None:
    if args.file not in files:
      files.append(args.file)
    cur = files.index(args.file)
  else:
    picked = _pick_file(search_dir)
    if picked is None:
      return
    cur = files.index(picked)

  # Re-infer if auto and we now know the file.
  if args.robot == "auto":
    robot = _infer_robot(files[cur])
    profile = PROFILES[robot]
    if args.xml is not None:
      profile = RobotProfile(
        name=profile.name,
        xml=args.xml,
        joint_names=profile.joint_names,
        track_body=profile.track_body,
        default_dir=profile.default_dir,
        default_fps=profile.default_fps,
        min_cols=profile.min_cols,
        cam_distance=profile.cam_distance,
      )

  fps = float(args.fps) if args.fps is not None else profile.default_fps
  player = MotionPlayer(files[cur], profile, fps, args.speed, loop=not args.no_loop)
  player.apply_frame(args.start)

  state = {
    "player": player,
    "profile": profile,
    "files": files,
    "cur": cur,
    "speed_i": SPEEDS.index(player.speed) if player.speed in SPEEDS else SPEEDS.index(
      min(SPEEDS, key=lambda s: abs(s - player.speed))
    ),
    "quit": False,
    "dirty": True,
    "fps": fps,
  }

  def _reload(delta: int) -> None:
    p = state["player"]
    state["cur"] = (state["cur"] + delta) % len(state["files"])
    path = state["files"][state["cur"]]
    state["player"] = MotionPlayer(
      path, state["profile"], state["fps"], p.speed, p.loop, model=p.model, data=p.data
    )
    state["player"].apply_frame(0)
    state["dirty"] = True

  def _bump_speed(delta: int) -> None:
    state["speed_i"] = min(max(state["speed_i"] + delta, 0), len(SPEEDS) - 1)
    state["player"].speed = SPEEDS[state["speed_i"]]
    state["dirty"] = True

  def on_key(key: int) -> None:
    p = state["player"]
    if key == K_SPACE:
      p.paused = not p.paused
    elif key in (ord("a"), ord("A"), K_LEFT):
      p.step_frame(-1)
    elif key in (ord("d"), ord("D"), K_RIGHT):
      p.step_frame(+1)
    elif key in (ord("w"), ord("W"), K_UP, ord("="), ord("+")):
      _bump_speed(+1)
    elif key in (ord("s"), ord("S"), K_DOWN, ord("-"), ord("_")):
      _bump_speed(-1)
    elif key in (ord("r"), ord("R")):
      p.apply_frame(0)
    elif key in (ord("l"), ord("L")):
      p.loop = not p.loop
    elif key in (ord("n"), ord("N"), ord("]")):
      _reload(+1)
    elif key in (ord("p"), ord("P"), ord("[")):
      _reload(-1)
    elif key in (ord("q"), ord("Q"), K_ESC):
      state["quit"] = True
    state["dirty"] = True

  with mujoco_viewer.launch_passive(player.model, player.data, key_callback=on_key) as v:
    body_id = mujoco.mj_name2id(player.model, mujoco.mjtObj.mjOBJ_BODY, profile.track_body)
    if body_id >= 0:
      v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      v.cam.trackbodyid = body_id
      v.cam.distance = profile.cam_distance
      v.cam.azimuth = 130.0
      v.cam.elevation = -8.0

    next_frame_t = time.perf_counter()
    print(
      f"\n[{profile.name}] {player.name}: {player.frames} frames @ {player.fps:.0f}fps "
      f"({player.duration:.1f}s)  speed={player.speed:g}x"
    )
    print(
      "Controls (focus the MuJoCo window):\n"
      "  Space pause | A/D or ←/→ frame step\n"
      "  S/- slower , W/= faster   (0.02x … 16x)\n"
      "  R restart | L loop | N/P next/prev file | Q quit\n"
      f"  Tip: start slow with  --speed 0.25\n"
    )
    shown_name = player.name

    while v.is_running() and not state["quit"]:
      p = state["player"]
      if p.name != shown_name:
        shown_name = p.name
        print(
          f"\n[{profile.name}] {p.name}: {p.frames} frames @ {p.fps:.0f}fps "
          f"({p.duration:.1f}s)  speed={p.speed:g}x"
        )
        next_frame_t = time.perf_counter()

      if not p.paused:
        now = time.perf_counter()
        frame_dt = 1.0 / max(p.fps, 1e-6)
        steps = 0
        while now >= next_frame_t and steps < 4:
          p.step_frame(+1)
          next_frame_t += frame_dt / max(p.speed, 1e-6)
          steps += 1
        if steps == 4:
          next_frame_t = now + frame_dt / max(p.speed, 1e-6)

      try:
        v.set_texts(
          (
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            "Robot\nFile\nFrame\nSpeed\nFPS\nKeys",
            f"{profile.name}\n"
            f"{state['files'][state['cur']].name}\n"
            f"{p.frame + 1}/{p.frames}  {'PAUSED' if p.paused else 'RUN'}\n"
            f"{p.speed:g}x   (S/- slower, W/= faster)\n"
            f"{p.fps:.0f}\n"
            f"Space N/P R L Q",
          )
        )
      except Exception:
        pass

      v.sync()
      print(
        f"\r{p.status()} | {state['files'][state['cur']].name[:44]:<44}",
        end="",
        flush=True,
      )
      time.sleep(0.002)

  print()


if __name__ == "__main__":
  main()
