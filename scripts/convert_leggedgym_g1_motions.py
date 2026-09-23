#!/usr/bin/env python3
"""Convert LeggedGym-Ex G1 reference motions (raw_run PKL) to AMP_mjlab formats.

Source PKL (raw_run) fields:
  fps          float, typically ~60
  root_pos     (N, 3)  world XYZ of pelvis [m]
  root_rot     (N, 4)  root quaternion **xyzw**
  dof_pos      (N, 29) joint angles [rad], order = G1Flat29DofCommonCfg.dof_names
                       (identical to AMP_mjlab csv_to_npz joint_names)

Outputs:
  1) 30 Hz CSV: 36 cols = root_pos(3) + root_quat_xyzw(4) + dof_pos(29)
  2) 50 Hz NPZ via scripts/csv_to_npz.py (FK through MuJoCo)

Usage:
  python scripts/convert_leggedgym_g1_motions.py
  python scripts/convert_leggedgym_g1_motions.py --skip-npz   # CSV only
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = (
  Path("/home/zju/lab/LeggedGym-Ex-main/resources/reference_motion/unitree_g1/raw_run")
)
DEFAULT_CSV_DIR = PROJECT_ROOT / "motion_data_csv" / "amp" / "leggedgym_run"
DEFAULT_NPZ_DIR = PROJECT_ROOT / "src" / "assets" / "motions" / "g1" / "amp" / "WalkandRun"

# Must match LeggedGym G1Flat29DofCommonCfg.dof_names AND AMP csv_to_npz.
JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
assert len(JOINT_NAMES) == 29


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
  """Batch slerp; q are (..., 4) xyzw; t is (N,) in [0,1]."""
  q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
  q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
  dot = np.sum(q0 * q1, axis=-1, keepdims=True)
  q1 = np.where(dot < 0, -q1, q1)
  dot = np.abs(dot).clip(-1.0, 1.0)
  theta = np.arccos(dot)
  sin_theta = np.sin(theta)
  # Near-parallel: fall back to lerp.
  near = (sin_theta < 1e-6).reshape(-1)
  t = t.reshape(-1, 1)
  out = np.empty_like(q0)
  if near.any():
    out[near] = (1.0 - t[near]) * q0[near] + t[near] * q1[near]
    out[near] /= np.linalg.norm(out[near], axis=-1, keepdims=True)
  far = ~near
  if far.any():
    w0 = np.sin((1.0 - t[far]) * theta[far]) / sin_theta[far]
    w1 = np.sin(t[far] * theta[far]) / sin_theta[far]
    out[far] = w0 * q0[far] + w1 * q1[far]
  return out


def resample_motion(
  root_pos: np.ndarray,
  root_rot_xyzw: np.ndarray,
  dof_pos: np.ndarray,
  src_fps: float,
  dst_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  n = root_pos.shape[0]
  duration = (n - 1) / src_fps
  times = np.arange(0.0, duration + 1e-9, 1.0 / dst_fps)
  src_t = np.arange(n) / src_fps
  idx = np.searchsorted(src_t, times, side="right") - 1
  idx = np.clip(idx, 0, n - 2)
  t0, t1 = src_t[idx], src_t[idx + 1]
  alpha = ((times - t0) / np.maximum(t1 - t0, 1e-12)).astype(np.float64)

  pos = (1.0 - alpha)[:, None] * root_pos[idx] + alpha[:, None] * root_pos[idx + 1]
  dof = (1.0 - alpha)[:, None] * dof_pos[idx] + alpha[:, None] * dof_pos[idx + 1]
  quat = _slerp(root_rot_xyzw[idx], root_rot_xyzw[idx + 1], alpha)
  return pos.astype(np.float64), quat.astype(np.float64), dof.astype(np.float64)


def pkl_to_csv(pkl_path: Path, csv_path: Path, csv_fps: float = 30.0) -> dict:
  with open(pkl_path, "rb") as f:
    data = pickle.load(f)

  src_fps = float(data["fps"])
  root_pos = np.asarray(data["root_pos"], dtype=np.float64)
  root_rot = np.asarray(data["root_rot"], dtype=np.float64)  # xyzw
  dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)

  if dof_pos.ndim != 2 or dof_pos.shape[1] != 29:
    raise ValueError(f"{pkl_path.name}: expected dof_pos (N,29), got {dof_pos.shape}")
  if root_rot.shape[1] != 4:
    raise ValueError(f"{pkl_path.name}: expected root_rot (N,4) xyzw, got {root_rot.shape}")

  # Normalize quaternions; keep xyzw.
  root_rot = root_rot / np.linalg.norm(root_rot, axis=-1, keepdims=True)

  pos, quat, dof = resample_motion(root_pos, root_rot, dof_pos, src_fps, csv_fps)
  rows = np.concatenate([pos, quat, dof], axis=1)  # (T, 36)
  assert rows.shape[1] == 36

  csv_path.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(csv_path, rows, delimiter=",", fmt="%.6f")

  meta = {
    "source": str(pkl_path),
    "src_fps": src_fps,
    "csv_fps": csv_fps,
    "src_frames": int(root_pos.shape[0]),
    "csv_frames": int(rows.shape[0]),
    "duration_s": (rows.shape[0] - 1) / csv_fps,
    "joint_order": list(JOINT_NAMES),
    "layout": "root_pos(3)+root_quat_xyzw(4)+dof_pos(29)",
  }
  print(
    f"[CSV] {pkl_path.name}: {meta['src_frames']}@{src_fps:.2f}Hz "
    f"-> {meta['csv_frames']}@{csv_fps:.0f}Hz ({meta['duration_s']:.2f}s) -> {csv_path.name}"
  )
  return meta


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--src-dir", type=Path, default=DEFAULT_SRC)
  parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
  parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
  parser.add_argument("--csv-fps", type=float, default=30.0)
  parser.add_argument("--npz-fps", type=float, default=50.0)
  parser.add_argument("--skip-npz", action="store_true")
  parser.add_argument("--device", type=str, default="cuda:0")
  args = parser.parse_args()

  pkls = sorted(args.src_dir.glob("*.pkl"))
  if not pkls:
    raise SystemExit(f"No .pkl in {args.src_dir}")

  print(f"Joint order ({len(JOINT_NAMES)}):")
  for i, n in enumerate(JOINT_NAMES):
    print(f"  [{i:2d}] {n}")
  print(
    "CSV layout: cols0-2 root_pos xyz | cols3-6 root_quat xyzw | "
    "cols7-35 joint angles (above order)\n"
  )

  csv_files = []
  for pkl in pkls:
    # Strip stageii suffix for cleaner names; keep motion id.
    stem = pkl.stem  # e.g. C3_-_run_stageii
    csv_path = args.csv_dir / f"{stem}.csv"
    pkl_to_csv(pkl, csv_path, csv_fps=args.csv_fps)
    csv_files.append(csv_path)

  if args.skip_npz:
    print(f"\nSkipped NPZ. CSVs in {args.csv_dir}")
    return

  args.npz_dir.mkdir(parents=True, exist_ok=True)
  cmd = [
    sys.executable,
    str(PROJECT_ROOT / "scripts" / "csv_to_npz.py"),
    "--input-dir",
    str(args.csv_dir),
    "--output-dir",
    str(args.npz_dir),
    "--input-fps",
    str(args.csv_fps),
    "--output-fps",
    str(args.npz_fps),
    "--device",
    args.device,
    "--render",
    "False",
  ]
  print("\n[NPZ]", " ".join(cmd))
  subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))
  print(f"\nDone. CSV -> {args.csv_dir}\n      NPZ -> {args.npz_dir}")


if __name__ == "__main__":
  main()
