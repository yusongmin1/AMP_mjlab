"""Go2-specific AMP locomotion rewards (ported from legged_lab_yu Go2AmpVae).

Kept separate from ``mdp/rewards.py`` so G1 / shared AMP rewards stay unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_lin_vel_xy_yaw_frame_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exp kernel on xy linear velocity error in the yaw (heading) frame."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  vel_yaw = quat_apply_inverse(
    yaw_quat(asset.data.root_link_quat_w),
    asset.data.root_link_lin_vel_w,
  )
  lin_vel_error = torch.sum(
    torch.square(command[:, :2] - vel_yaw[:, :2]),
    dim=1,
  )
  return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exp kernel on world-frame yaw-rate tracking error."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  ang_vel_error = torch.square(
    command[:, 2] - asset.data.root_link_ang_vel_w[:, 2]
  )
  return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize base-frame vertical linear velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def base_height_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared error between root height (world z − env origin z) and target."""
  asset: Entity = env.scene[asset_cfg.name]
  origins = env.scene.env_origins
  height = asset.data.root_link_pos_w[:, 2] - origins[:, 2]
  return torch.square(height - target_height)


def base_height_above_terrain(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  region_size: tuple[float, float] = (0.4, 0.4),
  sensor_name: str = "terrain_scan",
) -> torch.Tensor:
  """Abs error of height above local terrain vs target; falls back to L2 on flat."""
  if sensor_name not in env.scene.sensors:
    return base_height_l2(env, target_height, asset_cfg)

  asset: Entity = env.scene[asset_cfg.name]
  sensor: RayCastSensor = env.scene[sensor_name]
  hits = sensor.data.hit_pos_w  # [B, N, 3]
  distances = sensor.data.distances  # [B, N]

  local_offsets = getattr(sensor, "_local_offsets", None)
  if local_offsets is not None:
    half_x, half_y = region_size[0] * 0.5, region_size[1] * 0.5
    mask = (local_offsets[:, 0].abs() <= half_x) & (
      local_offsets[:, 1].abs() <= half_y
    )
  else:
    mask = torch.ones(hits.shape[1], dtype=torch.bool, device=hits.device)

  if not bool(mask.any()):
    mask = torch.ones(hits.shape[1], dtype=torch.bool, device=hits.device)

  hit_z = hits[:, mask, 2]
  valid = distances[:, mask] >= 0
  # Fallback to root-relative plane height when a ray misses.
  origins_z = env.scene.env_origins[:, 2].unsqueeze(1)
  root_z = asset.data.root_link_pos_w[:, 2].unsqueeze(1)
  fallback = root_z - origins_z
  hit_z = torch.where(valid, hit_z, root_z - fallback)
  terrain_z = hit_z.mean(dim=1)
  height = asset.data.root_link_pos_w[:, 2] - terrain_z
  return torch.abs(height - target_height)


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize base-frame roll/pitch angular velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def energy(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize |torque * joint_vel| L2 norm (mechanical power proxy)."""
  asset: Entity = env.scene[asset_cfg.name]
  power = torch.abs(asset.data.actuator_force * asset.data.joint_vel)
  return torch.norm(power, dim=-1)


def action_smoothness_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Second-order action difference: a_t − 2 a_{t-1} + a_{t-2}."""
  am = env.action_manager
  diff = am.action - 2.0 * am.prev_action + am.prev_prev_action
  return torch.sum(torch.square(diff), dim=1)


def undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 1.0,
) -> torch.Tensor:
  """Count non-foot bodies (or geoms) in contact with terrain above threshold."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force is not None:
    force_mag = torch.norm(data.force, dim=-1)  # [B, N]
    return (force_mag > threshold).sum(dim=-1).float()
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > threshold).any(dim=-1)  # [B, N]
    return hit.sum(dim=-1).float()
  assert data.found is not None
  return data.found.squeeze(-1).float()


def flat_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat base orientation via projected gravity xy."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """L1 joint deviation from default pose."""
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  angle = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - default_joint_pos[:, asset_cfg.joint_ids]
  )
  return torch.sum(torch.abs(angle), dim=1)


def stand_still_without_cmd(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint deviation from default when command ≈ 0 (and base upright)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  diff = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - default_joint_pos[:, asset_cfg.joint_ids]
  )
  reward = torch.sum(torch.abs(diff), dim=1)
  cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  reward = reward * (cmd_norm < command_threshold).float()
  upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
  return reward * upright
