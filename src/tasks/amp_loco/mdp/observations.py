from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.observations import (
  builtin_sensor,
  generated_commands,
  height_scan,
  joint_pos_rel,
  joint_vel_rel,
  last_action,
  projected_gravity,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _uniform_noise(x: torch.Tensor, n_min: float, n_max: float) -> torch.Tensor:
  return x + torch.rand_like(x) * (n_max - n_min) + n_min


def _actor_frame(
  env: ManagerBasedRlEnv,
  command_name: str,
  include_height_scan: bool,
  height_scan_sensor_name: str,
  height_scan_scale: float,
  corrupt: bool,
) -> torch.Tensor:
  """Pack one actor observation frame.

  Keeping the whole frame in one term preserves frame-major history order under
  stock mjlab (no ``history_ordering`` patch needed).
  """
  ang_vel = builtin_sensor(env, "robot/imu_ang_vel")
  gravity = projected_gravity(env)
  command = generated_commands(env, command_name)
  joint_pos = joint_pos_rel(env)
  joint_vel = joint_vel_rel(env)
  actions = last_action(env)

  if corrupt:
    ang_vel = _uniform_noise(ang_vel, -0.2, 0.2)
    gravity = _uniform_noise(gravity, -0.05, 0.05)
    joint_pos = _uniform_noise(joint_pos, -0.01, 0.01)
    joint_vel = _uniform_noise(joint_vel, -0.5, 0.5)

  parts = [ang_vel, gravity, command, joint_pos, joint_vel, actions]

  if include_height_scan:
    heights = height_scan(env, height_scan_sensor_name)
    if corrupt:
      heights = _uniform_noise(heights, -0.1, 0.1)
    parts.append(heights * height_scan_scale)

  return torch.cat(parts, dim=-1)


def actor_frame(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  include_height_scan: bool = True,
  height_scan_sensor_name: str = "terrain_scan",
  height_scan_scale: float = 0.2,
) -> torch.Tensor:
  corrupt = env.cfg.observations["actor"].enable_corruption
  return _actor_frame(
    env,
    command_name=command_name,
    include_height_scan=include_height_scan,
    height_scan_sensor_name=height_scan_sensor_name,
    height_scan_scale=height_scan_scale,
    corrupt=corrupt,
  )


def critic_frame(
  env: ManagerBasedRlEnv,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  command_name: str = "twist",
  include_height_scan: bool = True,
  height_scan_sensor_name: str = "terrain_scan",
  height_scan_scale: float = 0.2,
) -> torch.Tensor:
  actor = _actor_frame(
    env,
    command_name=command_name,
    include_height_scan=include_height_scan,
    height_scan_sensor_name=height_scan_sensor_name,
    height_scan_scale=height_scan_scale,
    corrupt=False,
  )
  lin_vel = builtin_sensor(env, "robot/imu_lin_vel")
  return torch.cat(
    (
      actor,
      lin_vel,
      robot_body_pos_b(env, anchor_cfg=anchor_cfg, body_cfg=body_cfg),
      robot_body_ori_b(env, anchor_cfg=anchor_cfg, body_cfg=body_cfg),
    ),
    dim=-1,
  )


def amp_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Discriminator observation: ``[joint_pos, joint_vel]`` (absolute), matching AMPLoader."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  joint_pos = asset.data.joint_pos[:, joint_ids]
  joint_vel = asset.data.joint_vel[:, joint_ids]
  return torch.cat((joint_pos, joint_vel), dim=-1)

def robot_body_pos_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    anchor_pos_w = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]   # (num_envs, 3)
    anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]  # (num_envs, 4)
    
    body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]     # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 4)

    num_bodies = body_pos_w.shape[1]
    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)

def robot_body_ori_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    anchor_pos_w = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]   # (num_envs, 3)
    anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]  # (num_envs, 4)
    
    body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]     # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 4)

    num_bodies = body_pos_w.shape[1]
    _, ori_b = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)

def robot_body_lin_vel_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    body_lin_vel_w = asset.data.body_link_lin_vel_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]       # (num_envs, num_bodies, 4)

    num_bodies = body_lin_vel_w.shape[1]

    body_lin_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_lin_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    return body_lin_vel_b.reshape(env.num_envs, -1)

def robot_body_ang_vel_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]       # (num_envs, num_bodies, 4)

    num_bodies = body_ang_vel_w.shape[1]

    body_ang_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_ang_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    return body_ang_vel_b.reshape(env.num_envs, -1)