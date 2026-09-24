"""Convert Go2 AMP mocap ``.txt`` (JSON) clips to AMP_mjlab CSV.

Source format (``My_unitree_go2_gym/datasets/mocap_motions_go2/*.txt``):
  JSON with keys ``FrameDuration``, ``Frames``, ...
  Each frame has 49 floats (Isaac Gym / AMPLoader layout)::

    [0:3]   root_pos  (x, y, z)  [m]
    [3:7]   root_rot  (x, y, z, w)  ***xyzw***  (Isaac; NOT MuJoCo wxyz)
    [7:19]  joint_pos (12)  FL/FR/RL/RR × (hip, thigh, calf)  [rad]
    [19:31] foot_pos_local (12)
    [31:34] root_lin_vel (3)
    [34:37] root_ang_vel (3)
    [37:49] joint_vel (12)  [and trailing foot_vel in full 61-layout; here 49]

Confirmed by ``AMPLoader`` + ``replay_amp_mujoco.quat_xyzw_to_wxyz`` (w = q[-1]).

Output CSV (no header, comma-separated, 19 cols) — matches ``convert_gc_go2.py``::

    root_pos(3) + root_quat_xyzw(4) + dof_pos(12)

Joint order (same as mjlab Go2 / convert_gc_go2)::

    FL_hip, FL_thigh, FL_calf,
    FR_hip, FR_thigh, FR_calf,
    RL_hip, RL_thigh, RL_calf,
    RR_hip, RR_thigh, RR_calf

Usage::

  python scripts/convert_go2_mocap_txt_to_csv.py
  python scripts/convert_go2_mocap_txt_to_csv.py \\
      --src /path/to/mocap_motions_go2 \\
      --out src/assets/motions/go2/mocap_csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# AMPLoader indices (see My_unitree_go2_gym/rsl_rl/.../motion_loader.py).
_ROOT_POS = slice(0, 3)
_ROOT_ROT = slice(3, 7)  # xyzw
_JOINT_POS = slice(7, 19)
_EXPECTED_COLS = 49
_CSV_COLS = 19  # pos3 + quat_xyzw4 + dof12


def _standardize_xyzw(q: np.ndarray) -> np.ndarray:
  """Unit-normalize and flip so w >= 0 (same as motion_util.standardize_quaternion)."""
  q = np.asarray(q, dtype=np.float64)
  n = np.linalg.norm(q, axis=-1, keepdims=True)
  q = q / np.clip(n, 1e-12, None)
  flip = q[..., 3:4] < 0
  return np.where(flip, -q, q)


def convert_file(src: Path, dst: Path) -> tuple[int, float]:
  data = json.loads(src.read_text())
  frames = np.asarray(data["Frames"], dtype=np.float64)
  if frames.ndim != 2 or frames.shape[1] < _CSV_COLS:
    raise ValueError(f"{src.name}: expected (>=){_CSV_COLS} cols, got {frames.shape}")
  if frames.shape[1] != _EXPECTED_COLS:
    print(f"[WARN] {src.name}: cols={frames.shape[1]} (expected {_EXPECTED_COLS}); using first {_CSV_COLS}")

  pos = frames[:, _ROOT_POS]
  quat_xyzw = _standardize_xyzw(frames[:, _ROOT_ROT])
  dof = frames[:, _JOINT_POS]
  out = np.concatenate([pos, quat_xyzw, dof], axis=1)
  assert out.shape[1] == _CSV_COLS

  dst.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(dst, out, delimiter=",", fmt="%.8f")
  dt = float(data.get("FrameDuration", 0.04))
  return out.shape[0], dt


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
    "--src",
    type=Path,
    default=Path("/home/zju/My_unitree_go2_gym/datasets/mocap_motions_go2"),
    help="Directory of mocap .txt JSON clips",
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "src" / "assets" / "motions" / "go2" / "mocap_csv",
    help="Output directory for CSV files",
  )
  args = parser.parse_args()

  files = sorted(args.src.glob("*.txt"))
  if not files:
    raise SystemExit(f"No .txt files under {args.src}")

  print(f"[INFO] src={args.src}")
  print(f"[INFO] out={args.out}")
  print("[INFO] quat convention: xyzw (Isaac); CSV keeps xyzw (convert_gc_go2 / AMP_mjlab)")
  print("[INFO] joints: FL/FR/RL/RR × hip,thigh,calf")

  for src in files:
    dst = args.out / f"{src.stem}.csv"
    n, dt = convert_file(src, dst)
    print(f"  {src.name:28s} -> {dst.name:28s}  T={n:4d}  dt={dt:.3f}s  fps={1.0/dt:.1f}")

  print(f"[DONE] wrote {len(files)} CSV files to {args.out}")


if __name__ == "__main__":
  main()
