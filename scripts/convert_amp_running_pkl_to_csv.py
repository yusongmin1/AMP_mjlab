#!/usr/bin/env python3
"""Convert AMP_Running_baseline / legged_lab G1 AMP PKL motions to AMP_mjlab CSV.

Source PKL (joblib) uses **Isaac Lab joint order** (breadth-first USD order) and
``root_rot`` in **wxyz**. AMP_mjlab CSV expects **MuJoCo joint order** and
``root_quat`` in **xyzw**.

Lab order (from ``scripts/tools/retarget/config/g1_29dof.yaml`` / symmetry docs):
  L/R interleaved by DOF type (pitch pairs, then roll, ...).

MuJoCo / AMP_mjlab order:
  left leg (6) | right leg (6) | waist (3) | left arm (7) | right arm (7)

Output CSV @ 30 Hz:
  36 cols = root_pos(3) + root_quat_xyzw(4) + dof_pos(29 MuJoCo order)

Usage:
  python scripts/convert_amp_running_pkl_to_csv.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = (
  Path("/home/zju/lab/AMP_Running_baseline-main/source/legged_lab/legged_lab/data")
  / "MotionData"
  / "g1_29dof"
  / "amp"
)
DEFAULT_CSV_DIR = PROJECT_ROOT / "motion_data_csv" / "amp_running_baseline"

# Isaac Lab articulation order (PKL dof_pos).
LAB_DOF_NAMES = (
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
)

# MuJoCo / AMP_mjlab CSV joint order.
MUJOCO_DOF_NAMES = (
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

assert len(LAB_DOF_NAMES) == len(MUJOCO_DOF_NAMES) == 29
# Index into lab dof_pos to produce mujoco order.
_LAB_TO_MUJOCO = tuple(LAB_DOF_NAMES.index(n) for n in MUJOCO_DOF_NAMES)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
  q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
  q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
  dot = np.sum(q0 * q1, axis=-1, keepdims=True)
  q1 = np.where(dot < 0, -q1, q1)
  dot = np.abs(dot).clip(-1.0, 1.0)
  theta = np.arccos(dot)
  sin_theta = np.sin(theta)
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
  if abs(src_fps - dst_fps) < 1e-6:
    return root_pos, root_rot_xyzw, dof_pos
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


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
  q = np.asarray(q, dtype=np.float64)
  return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def lab_dof_to_mujoco(dof_lab: np.ndarray) -> np.ndarray:
  """Reorder (N, 29) from Isaac Lab order to MuJoCo order."""
  return dof_lab[..., list(_LAB_TO_MUJOCO)]


def pkl_to_csv(pkl_path: Path, csv_path: Path, csv_fps: float = 30.0) -> dict:
  data = joblib.load(pkl_path)
  if not isinstance(data, dict):
    raise ValueError(f"{pkl_path}: expected dict, got {type(data)}")

  src_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
  root_pos = np.asarray(data["root_pos"], dtype=np.float64)
  root_rot_wxyz = np.asarray(data["root_rot"], dtype=np.float64)
  dof_lab = np.asarray(data["dof_pos"], dtype=np.float64)

  if dof_lab.ndim != 2 or dof_lab.shape[1] != 29:
    raise ValueError(f"{pkl_path.name}: expected dof_pos (N,29), got {dof_lab.shape}")
  if root_rot_wxyz.shape[1] != 4:
    raise ValueError(f"{pkl_path.name}: expected root_rot (N,4) wxyz, got {root_rot_wxyz.shape}")

  dof_mujoco = lab_dof_to_mujoco(dof_lab)
  root_rot_xyzw = wxyz_to_xyzw(root_rot_wxyz)
  root_rot_xyzw = root_rot_xyzw / np.linalg.norm(root_rot_xyzw, axis=-1, keepdims=True)

  pos, quat, dof = resample_motion(root_pos, root_rot_xyzw, dof_mujoco, src_fps, csv_fps)
  rows = np.concatenate([pos, quat, dof], axis=1)
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
    "joint_remap": "lab -> mujoco",
  }
  print(
    f"[CSV] {pkl_path.name}: {meta['src_frames']}@{src_fps:.2f}Hz "
    f"-> {meta['csv_frames']}@{csv_fps:.0f}Hz ({meta['duration_s']:.2f}s) "
    f"[lab→mujoco] -> {csv_path}"
  )
  return meta


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--src-dir", type=Path, default=DEFAULT_SRC)
  parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
  parser.add_argument("--csv-fps", type=float, default=30.0)
  args = parser.parse_args()

  pkls = sorted(args.src_dir.rglob("*.pkl"))
  if not pkls:
    raise SystemExit(f"No .pkl under {args.src_dir}")

  real = []
  for p in pkls:
    if p.read_bytes()[:40].startswith(b"version https://git-lfs"):
      print(f"[SKIP LFS pointer] {p}")
      continue
    real.append(p)
  if not real:
    raise SystemExit(f"No real .pkl under {args.src_dir}")

  print(f"Converting {len(real)} motions -> {args.csv_dir} @ {args.csv_fps} Hz")
  print("Remap: Isaac Lab dof order -> MuJoCo / AMP_mjlab order")
  print("CSV layout: root_pos xyz | root_quat xyzw | 29 joints (MuJoCo)\n")

  for pkl in real:
    rel = pkl.relative_to(args.src_dir)
    csv_path = args.csv_dir / rel.with_suffix(".csv")
    pkl_to_csv(pkl, csv_path, csv_fps=args.csv_fps)

  print(f"\nDone. CSVs in {args.csv_dir}")


if __name__ == "__main__":
  main()
