"""Unitree Go2 AMP locomotion environment configurations.

Builds on ``make_amp_env_cfg`` (AMP obs/rewards/motion reset) and applies Go2
tracking-style domain randomization: DelayedActuator 0–3, foot friction,
trunk COM, encoder bias, push ranges from tracking.
"""

from __future__ import annotations

import os

from mjlab.actuator import DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.assets.robots import GO2_ACTION_SCALE, get_go2_robot_cfg
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg
from src.tasks.amp_loco.mdp.command import SpinningUniformVelocityCommandCfg

# Tracking push / COM ranges (mjlab tracking_env_cfg.VELOCITY_RANGE + base_com).
_TRACKING_PUSH_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def go2_amp_rough_env_cfg(
  play: bool = False,
  enable_actuator_delay: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain AMP configuration."""
  cfg = make_amp_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  robot_cfg = get_go2_robot_cfg()
  if enable_actuator_delay and not play:
    articulation = robot_cfg.articulation
    assert articulation is not None
    robot_cfg.articulation = EntityArticulationInfoCfg(
      actuators=tuple(
        DelayedActuatorCfg(
          base_cfg=act,
          delay_target="position",
          delay_min_lag=0,
          delay_max_lag=3,
        )
        for act in articulation.actuators
      ),
      soft_joint_pos_limit_factor=articulation.soft_joint_pos_limit_factor,
    )
  cfg.scene.entities = {"robot": robot_cfg}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "trunk"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{n}_foot_collision" for n in foot_names)
  body_names = (
    "trunk",
    "FL_hip",
    "FL_thigh",
    "FL_foot",
    "FR_hip",
    "FR_thigh",
    "FR_foot",
    "RL_hip",
    "RL_thigh",
    "RL_foot",
    "RR_hip",
    "RR_thigh",
    "RR_foot",
  )
  anchor_name = "trunk"
  root_name = "trunk"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO2_ACTION_SCALE

  cfg.viewer.body_name = "trunk"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # Replace twist with spinning-aware command (10% in-place yaw).
  base_twist = cfg.commands["twist"]
  assert isinstance(base_twist, UniformVelocityCommandCfg)
  cfg.commands["twist"] = SpinningUniformVelocityCommandCfg(
    entity_name=base_twist.entity_name,
    resampling_time_range=base_twist.resampling_time_range,
    rel_standing_envs=base_twist.rel_standing_envs,
    rel_heading_envs=base_twist.rel_heading_envs,
    heading_command=base_twist.heading_command,
    heading_control_stiffness=base_twist.heading_control_stiffness,
    debug_vis=base_twist.debug_vis,
    ranges=base_twist.ranges,
    viz=base_twist.viz,
    rel_spinning_envs=0.1,
  )
  twist_cmd = cfg.commands["twist"]
  twist_cmd.viz.z_offset = 0.45

  # --- Tracking-style DR (replace G1-heavy mass / Unitree PD) ---
  cfg.events.pop("add_base_mass", None)
  cfg.events.pop("add_mass", None)
  # Go2 uses BuiltinPositionActuator; UnitreeActuator PD rand does not apply.
  cfg.events.pop("randomize_actuator_gains", None)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.025, 0.025),
    1: (-0.05, 0.05),
    2: (-0.05, 0.05),
  }
  cfg.events["encoder_bias"].params["bias_range"] = (-0.01, 0.01)
  cfg.events["push_robot"].params["velocity_range"] = dict(_TRACKING_PUSH_RANGE)
  cfg.events["recovery_assist_force"].params["asset_cfg"].body_names = ("trunk",)
  # Lower assist for lighter quadruped.
  cfg.events["recovery_assist_force"].params["initial_force"] = 80.0
  if "recovery_assist_force" in cfg.curriculum:
    cfg.curriculum["recovery_assist_force"].params["initial_force"] = 80.0
    cfg.curriculum["recovery_assist_force"].params["force_decay"] = 10.0

  # Motion dirs (loco + stand placeholder for recovery pool).
  _motion_base = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "go2", "amp"
  )
  _motion_dir = os.path.abspath(os.path.join(_motion_base, "WalkandRun"))
  _recovery_dir = os.path.abspath(os.path.join(_motion_base, "Recovery"))

  # No dedicated fall/getup clips yet — keep delay off (tracking-like).
  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 0
  cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = _recovery_dir
  cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir
  cfg.events["reset_from_motion"].params["recovery_dir"] = _recovery_dir

  cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (root_name,)
  # Keep only root projected-gravity flatness (drop body_orientation + hip deviation).
  cfg.rewards.pop("body_orientation_l2", None)
  cfg.rewards.pop("joint_deviation_hip", None)
  cfg.rewards["flat_orientation_l2"].weight = -0.2

  # Standing height ~0.28–0.35 m; terminate if collapsed.
  cfg.terminations["bad_base_height"].params["minimum_height"] = 0.12

  cfg.observations["critic"].terms["frame"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["frame"].params["body_cfg"].body_names = body_names

  # Milder velocity curriculum for quadruped mocap speeds.
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
      {"step": 5000 * 24, "lin_vel_x": (-1.0, 1.5), "lin_vel_y": (-0.5, 0.5)},
      {"step": 10000 * 24, "lin_vel_x": (-1.5, 2.0), "lin_vel_y": (-0.8, 0.8), "ang_vel_z": (-1.5, 1.5)},
    ]

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    # Enable delayed termination for play demos if desired later.
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def go2_amp_flat_env_cfg(
  play: bool = False,
  enable_actuator_delay: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain AMP configuration."""
  cfg = go2_amp_rough_env_cfg(play=play, enable_actuator_delay=enable_actuator_delay)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  cfg.observations["actor"].terms["frame"].params["include_height_scan"] = False
  cfg.observations["critic"].terms["frame"].params["include_height_scan"] = False

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, SpinningUniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.lin_vel_y = (-0.8, 0.8)
    twist_cmd.ranges.ang_vel_z = (-1.5, 1.5)

  return cfg
