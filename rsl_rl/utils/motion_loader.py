from __future__ import annotations

import os

import numpy as np
import torch
from tqdm import tqdm


class AMPLoader:
  """Expert AMP dataset providing ``[joint_pos, joint_vel]`` transitions.

  Matches the TienKung-style joint-only AMP observation (no body-relative
  kinematics). Each transition is sampled as consecutive frames ``(s, s_next)``.
  """

  def __init__(self, motion_file: str, device: str = "cuda:0", **_unused):
    """Load AMP motion data.

    Args:
      motion_file: Path to a single ``.npz`` or a directory of ``.npz`` files.
      device: Torch device.
      **_unused: Ignored (compat with older callers that passed body/anchor names).
    """
    del _unused
    assert os.path.exists(motion_file), f"Invalid path: {motion_file}"

    if os.path.isfile(motion_file):
      motion_files = [motion_file]
      motion_names = [os.path.splitext(os.path.basename(motion_file))[0]]
    elif os.path.isdir(motion_file):
      motion_names = []
      motion_files = []
      for root, _dirs, files in os.walk(motion_file):
        for filename in sorted(files):
          if filename.endswith(".npz"):
            motion_names.append(os.path.splitext(filename)[0])
            motion_files.append(os.path.join(root, filename))
      if motion_files:
        motion_files, motion_names = zip(*sorted(zip(motion_files, motion_names)))
        motion_files, motion_names = list(motion_files), list(motion_names)
      assert len(motion_files) > 0, f"No npz files found in directory: {motion_file}"
    else:
      raise ValueError(f"Path is neither a file nor a directory: {motion_file}")

    self.motion_names = motion_names
    self.device = device
    self._dof_pos_list: list[torch.Tensor] = []
    self._dof_vel_list: list[torch.Tensor] = []

    for motion_idx, (motion_name, motion_path) in enumerate(zip(motion_names, motion_files)):
      print(f"Processing motion {motion_idx + 1}/{len(motion_files)}: {motion_name}")
      data = np.load(motion_path)
      if motion_idx == 0:
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])

      dof_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
      dof_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
      assert dof_pos.ndim == 2 and dof_vel.ndim == 2, (
        f"{motion_name}: expected joint_pos/joint_vel of shape (T, DoF), "
        f"got {dof_pos.shape}, {dof_vel.shape}"
      )
      assert dof_pos.shape == dof_vel.shape, (
        f"{motion_name}: joint_pos {dof_pos.shape} != joint_vel {dof_vel.shape}"
      )
      self._dof_pos_list.append(dof_pos)
      self._dof_vel_list.append(dof_vel)
      print(
        f"  frames={dof_pos.shape[0]} dof={dof_pos.shape[1]} "
        f"duration={dof_pos.shape[0] / self.fps:.2f}s"
      )

    self._num_dof = int(self._dof_pos_list[0].shape[1])
    self.time_step_total = int(self._dof_pos_list[0].shape[0])
    self.motion_total_time = self.time_step_total / self.fps

  @property
  def observation_dim(self) -> int:
    """Size of one AMP frame: joint_pos (DoF) + joint_vel (DoF)."""
    return 2 * self._num_dof

  def _frame(self, dof_pos: torch.Tensor, dof_vel: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
    return torch.cat((dof_pos[idxs], dof_vel[idxs]), dim=-1)

  def feed_forward_generator(self, num_mini_batch, mini_batch_size):
    """Yield ``(s, s_next)`` mini-batches of AMP transitions."""
    num_motions = len(self._dof_pos_list)
    for batch_idx in range(num_mini_batch):
      motion_idx = batch_idx % num_motions
      dof_pos = self._dof_pos_list[motion_idx]
      dof_vel = self._dof_vel_list[motion_idx]
      n_frames = dof_pos.shape[0]
      # Leave room for s_next = idxs + 1.
      idxs = torch.randint(0, max(n_frames - 1, 1), (mini_batch_size,), device=dof_pos.device)
      idxs = torch.clamp(idxs, max=max(n_frames - 2, 0))
      next_idxs = idxs + 1
      yield self._frame(dof_pos, dof_vel, idxs), self._frame(dof_pos, dof_vel, next_idxs)
