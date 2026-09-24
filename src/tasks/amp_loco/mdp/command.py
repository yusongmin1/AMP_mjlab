"""Velocity command variants for AMP locomotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class SpinningUniformVelocityCommand(UniformVelocityCommand):
  """Uniform twist command with a fraction of envs forced to spin in place.

  After the base resample (and standing override), ``rel_spinning_envs`` of
  environments get ``lin_vel_xy = 0`` while keeping a non-zero ``ang_vel_z``.
  Spinning and standing are mutually exclusive (standing wins).
  """

  cfg: "SpinningUniformVelocityCommandCfg"

  def __init__(self, cfg: "SpinningUniformVelocityCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.is_spinning_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    r = torch.empty(len(env_ids), device=self.device)
    spinning = r.uniform_(0.0, 1.0) <= self.cfg.rel_spinning_envs
    # Standing already zeros the command; do not mark those as spinning.
    spinning = spinning & (~self.is_standing_env[env_ids])
    self.is_spinning_env[env_ids] = spinning
    spin_ids = env_ids[spinning]
    if len(spin_ids) == 0:
      return
    # No heading servo while spinning — keep sampled yaw rate.
    self.is_heading_env[spin_ids] = False
    self.vel_command_b[spin_ids, :2] = 0.0
    # Ensure |ωz| is not near zero (resample if needed).
    ang = self.vel_command_b[spin_ids, 2]
    too_small = ang.abs() < 0.2
    if too_small.any():
      n = int(too_small.sum().item())
      sign = torch.where(
        torch.rand(n, device=self.device) < 0.5,
        -torch.ones(n, device=self.device),
        torch.ones(n, device=self.device),
      )
      lo, hi = self.cfg.ranges.ang_vel_z
      mag = torch.empty(n, device=self.device).uniform_(max(0.2, abs(lo) * 0.3), max(abs(lo), abs(hi)))
      ang = ang.clone()
      ang[too_small] = sign * mag
      self.vel_command_b[spin_ids, 2] = ang

  def _update_command(self) -> None:
    super()._update_command()
    spinning_ids = self.is_spinning_env.nonzero(as_tuple=False).flatten()
    if len(spinning_ids) == 0:
      return
    # Re-assert in-place spin after heading / standing updates.
    standing = self.is_standing_env[spinning_ids]
    active = spinning_ids[~standing]
    self.vel_command_b[active, :2] = 0.0


@dataclass(kw_only=True)
class SpinningUniformVelocityCommandCfg(UniformVelocityCommandCfg):
  """Like ``UniformVelocityCommandCfg`` plus in-place spinning fraction."""

  rel_spinning_envs: float = 0.1
  """Fraction of environments commanded to spin with zero linear velocity."""

  def build(self, env: "ManagerBasedRlEnv") -> SpinningUniformVelocityCommand:
    return SpinningUniformVelocityCommand(self, env)
