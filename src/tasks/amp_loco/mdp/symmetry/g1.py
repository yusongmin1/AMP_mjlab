"""Left-right symmetry for Unitree G1 29-DoF AMP observations / actions.

Matches the RSL-RL symmetry API used by ``AMPPPO``:

.. code-block:: python

    obs_aug, actions_aug = compute_symmetric_states(
        env=env, obs=obs, actions=actions, obs_type="policy"
    )

Observation layout (per history frame), matching ``mdp.observations._actor_frame`` /
``critic_frame``:

* actor: ``ang_vel(3) | gravity(3) | cmd(3) | joint_pos(29) | joint_vel(29) | last_action(29) [| height_scan]``
* critic: actor frame + ``lin_vel(3) | body_pos_b(3*N) | body_ori_b(6*N)``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["compute_symmetric_states"]

# G1 MuJoCo / Entity joint order (29-DoF).
_NUM_JOINTS = 29

# Left <-> right joint index pairs.
_LEFT_JOINTS = (0, 1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21)
_RIGHT_JOINTS = (6, 7, 8, 9, 10, 11, 22, 23, 24, 25, 26, 27, 28)

# Roll / yaw joints flip sign under sagittal reflection (pitch stays).
_SIGN_FLIP_JOINTS = (
  1, 2, 5,  # left hip roll/yaw, ankle roll
  7, 8, 11,  # right hip roll/yaw, ankle roll
  12, 13,  # waist yaw/roll
  16, 17, 19, 21,  # left shoulder roll/yaw, wrist roll/yaw
  23, 24, 26, 28,  # right shoulder roll/yaw, wrist roll/yaw
)

# Critic key-body order from ``g1_amp_rough_env_cfg`` (pelvis + L/R pairs).
_BODY_PAIR_LEFT = (1, 2, 3, 7, 8, 9)
_BODY_PAIR_RIGHT = (4, 5, 6, 10, 11, 12)
_NUM_BODIES = 13


@torch.no_grad()
def compute_symmetric_states(
  env: "ManagerBasedRlEnv",
  obs: torch.Tensor | None = None,
  actions: torch.Tensor | None = None,
  obs_type: str = "policy",
):
  """Augment obs/actions with the left-right mirrored copy (batch x2)."""
  if obs is not None:
    batch_size = obs.shape[0]
    obs_aug = obs.repeat(2, 1)
    obs_aug[:batch_size] = obs
    if obs_type in ("policy", "actor"):
      obs_aug[batch_size:] = _transform_actor_obs_left_right(env, obs)
    elif obs_type == "critic":
      obs_aug[batch_size:] = _transform_critic_obs_left_right(env, obs)
    else:
      raise ValueError(f"Unsupported obs_type '{obs_type}'. Use 'policy'/'actor' or 'critic'.")
  else:
    obs_aug = None

  if actions is not None:
    batch_size = actions.shape[0]
    actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype)
    actions_aug[:batch_size] = actions
    actions_aug[batch_size:] = _switch_joints_left_right(actions)
  else:
    actions_aug = None

  return obs_aug, actions_aug


def _unwrap_env(env):
  return env.unwrapped if hasattr(env, "unwrapped") else env


def _obs_group_cfg(env, obs_type: str):
  unwrapped = _unwrap_env(env)
  group_name = "actor" if obs_type in ("policy", "actor") else "critic"
  return unwrapped.cfg.observations[group_name]


def _frame_params(env, obs_type: str) -> tuple[int, bool, int, int]:
  """Return ``(history_length, include_height_scan, height_scan_dim, num_bodies)``."""
  group = _obs_group_cfg(env, obs_type)
  history_length = int(group.history_length or 1)
  frame_params = group.terms["frame"].params
  include_hs = bool(frame_params.get("include_height_scan", False))
  hs_dim = 0
  if include_hs:
    hs_dim = _height_scan_dim(env, frame_params.get("height_scan_sensor_name", "terrain_scan"))
  num_bodies = _NUM_BODIES
  if obs_type == "critic":
    body_cfg = frame_params.get("body_cfg")
    if body_cfg is not None and getattr(body_cfg, "body_names", None):
      num_bodies = len(body_cfg.body_names)
      if num_bodies != _NUM_BODIES:
        raise ValueError(
          f"Symmetry expects {_NUM_BODIES} critic bodies (pelvis + L/R pairs), got {num_bodies}."
        )
  return history_length, include_hs, hs_dim, num_bodies


def _height_scan_dim(env, sensor_name: str) -> int:
  unwrapped = _unwrap_env(env)
  try:
    sensor = unwrapped.scene[sensor_name]
    return int(sensor.data.hit_pos_w.shape[-2])
  except Exception:
    # Fallback for GridPatternCfg(size=(1.6, 1.0), resolution=0.1) -> 17x11.
    return 17 * 11


def _height_scan_grid(env, sensor_name: str) -> tuple[int, int]:
  """Return ``(ny, nx)`` for left-right flip of a grid height scan."""
  unwrapped = _unwrap_env(env)
  try:
    pattern = unwrapped.scene[sensor_name].cfg.pattern
    size_x, size_y = pattern.size
    res = pattern.resolution
    # Same construction as ``GridPatternCfg.compute`` (indexing="xy").
    nx = int(torch.arange(-size_x / 2, size_x / 2 + res * 0.5, res).numel())
    ny = int(torch.arange(-size_y / 2, size_y / 2 + res * 0.5, res).numel())
    return ny, nx
  except Exception:
    return 11, 17


def _actor_frame_dim(include_hs: bool, hs_dim: int) -> int:
  base = 3 + 3 + 3 + _NUM_JOINTS * 3  # 96
  return base + (hs_dim if include_hs else 0)


def _transform_actor_obs_left_right(env, obs: torch.Tensor) -> torch.Tensor:
  history_length, include_hs, hs_dim, _ = _frame_params(env, "actor")
  frame_dim = _actor_frame_dim(include_hs, hs_dim)
  expected = frame_dim * history_length
  if obs.shape[-1] != expected:
    raise ValueError(
      f"Actor obs dim {obs.shape[-1]} != history({history_length}) * frame({frame_dim}) = {expected}. "
      "Update symmetry layout if observation packing changed."
    )

  obs = obs.clone()
  device = obs.device
  ang_scale = torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=obs.dtype)
  grav_scale = torch.tensor([1.0, -1.0, 1.0], device=device, dtype=obs.dtype)
  cmd_scale = torch.tensor([1.0, -1.0, -1.0], device=device, dtype=obs.dtype)
  hs_ny, hs_nx = _height_scan_grid(env, "terrain_scan") if include_hs else (0, 0)

  for h in range(history_length):
    base = h * frame_dim
    obs[:, base : base + 3] *= ang_scale
    obs[:, base + 3 : base + 6] *= grav_scale
    obs[:, base + 6 : base + 9] *= cmd_scale
    jp = base + 9
    jv = jp + _NUM_JOINTS
    la = jv + _NUM_JOINTS
    obs[:, jp:jv] = _switch_joints_left_right(obs[:, jp:jv])
    obs[:, jv:la] = _switch_joints_left_right(obs[:, jv:la])
    obs[:, la : la + _NUM_JOINTS] = _switch_joints_left_right(obs[:, la : la + _NUM_JOINTS])
    if include_hs:
      hs0 = la + _NUM_JOINTS
      hs = obs[:, hs0 : hs0 + hs_dim]
      obs[:, hs0 : hs0 + hs_dim] = hs.reshape(-1, hs_ny, hs_nx).flip(-2).reshape(-1, hs_dim)

  return obs


def _transform_critic_obs_left_right(env, obs: torch.Tensor) -> torch.Tensor:
  history_length, include_hs, hs_dim, num_bodies = _frame_params(env, "critic")
  actor_dim = _actor_frame_dim(include_hs, hs_dim)
  # Critic appends privileged terms once per history frame (packed in the same frame term).
  # ``critic_frame`` concatenates actor + lin_vel + body_pos + body_ori into one frame.
  frame_dim = actor_dim + 3 + num_bodies * 3 + num_bodies * 6
  expected = frame_dim * history_length
  if obs.shape[-1] != expected:
    raise ValueError(
      f"Critic obs dim {obs.shape[-1]} != history({history_length}) * frame({frame_dim}) = {expected}. "
      "Update symmetry layout if observation packing changed."
    )

  # Reuse actor transform on the actor slice of each frame, then mirror privileged extras.
  obs = obs.clone()
  device = obs.device
  ang_scale = torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=obs.dtype)
  grav_scale = torch.tensor([1.0, -1.0, 1.0], device=device, dtype=obs.dtype)
  cmd_scale = torch.tensor([1.0, -1.0, -1.0], device=device, dtype=obs.dtype)
  lin_scale = torch.tensor([1.0, -1.0, 1.0], device=device, dtype=obs.dtype)
  hs_ny, hs_nx = _height_scan_grid(env, "terrain_scan") if include_hs else (0, 0)

  for h in range(history_length):
    base = h * frame_dim
    # --- actor part ---
    obs[:, base : base + 3] *= ang_scale
    obs[:, base + 3 : base + 6] *= grav_scale
    obs[:, base + 6 : base + 9] *= cmd_scale
    jp = base + 9
    jv = jp + _NUM_JOINTS
    la = jv + _NUM_JOINTS
    obs[:, jp:jv] = _switch_joints_left_right(obs[:, jp:jv])
    obs[:, jv:la] = _switch_joints_left_right(obs[:, jv:la])
    obs[:, la : la + _NUM_JOINTS] = _switch_joints_left_right(obs[:, la : la + _NUM_JOINTS])
    cursor = la + _NUM_JOINTS
    if include_hs:
      hs = obs[:, cursor : cursor + hs_dim]
      obs[:, cursor : cursor + hs_dim] = hs.reshape(-1, hs_ny, hs_nx).flip(-2).reshape(-1, hs_dim)
      cursor += hs_dim

    # --- privileged ---
    obs[:, cursor : cursor + 3] *= lin_scale
    cursor += 3
    pos = obs[:, cursor : cursor + num_bodies * 3].reshape(-1, num_bodies, 3)
    obs[:, cursor : cursor + num_bodies * 3] = _switch_body_pos_left_right(pos).reshape(-1, num_bodies * 3)
    cursor += num_bodies * 3
    ori = obs[:, cursor : cursor + num_bodies * 6].reshape(-1, num_bodies, 6)
    obs[:, cursor : cursor + num_bodies * 6] = _switch_body_ori_left_right(ori).reshape(-1, num_bodies * 6)

  return obs


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
  out = joint_data.clone()
  out[..., _LEFT_JOINTS] = joint_data[..., _RIGHT_JOINTS]
  out[..., _RIGHT_JOINTS] = joint_data[..., _LEFT_JOINTS]
  out[..., _SIGN_FLIP_JOINTS] *= -1.0
  return out


def _switch_body_pos_left_right(pos: torch.Tensor) -> torch.Tensor:
  """``pos`` shape ``(B, num_bodies, 3)`` in anchor frame."""
  out = pos.clone()
  # Unpaired pelvis: only flip y.
  out[:, 0, 1] *= -1.0
  out[:, _BODY_PAIR_LEFT] = pos[:, _BODY_PAIR_RIGHT]
  out[:, _BODY_PAIR_RIGHT] = pos[:, _BODY_PAIR_LEFT]
  out[:, _BODY_PAIR_LEFT, 1] *= -1.0
  out[:, _BODY_PAIR_RIGHT, 1] *= -1.0
  return out


def _switch_body_ori_left_right(ori: torch.Tensor) -> torch.Tensor:
  """``ori`` shape ``(B, num_bodies, 6)`` = first two rotation-matrix columns.

  Sagittal reflection ``S=diag(1,-1,1)`` maps column ``c -> S c``, i.e. negate the
  y component of each stored column: indices 1 and 4 within the 6-vector.
  """
  out = ori.clone()
  # Pelvis: flip y components of both columns.
  out[:, 0, 1] *= -1.0
  out[:, 0, 4] *= -1.0
  out[:, _BODY_PAIR_LEFT] = ori[:, _BODY_PAIR_RIGHT]
  out[:, _BODY_PAIR_RIGHT] = ori[:, _BODY_PAIR_LEFT]
  out[:, _BODY_PAIR_LEFT, 1] *= -1.0
  out[:, _BODY_PAIR_LEFT, 4] *= -1.0
  out[:, _BODY_PAIR_RIGHT, 1] *= -1.0
  out[:, _BODY_PAIR_RIGHT, 4] *= -1.0
  return out
