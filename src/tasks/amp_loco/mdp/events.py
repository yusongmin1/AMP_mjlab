from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

from src.tasks.amp_loco.ampmotion_loader import MotionLoader
from src.tasks.amp_loco.mdp.terminations import DelayedTerminationManager

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class MotionResetManager:
    """Manages motion frame data and delayed-reset logic for AMP environments."""

    _instance: MotionResetManager | None = None

    def __init__(self) -> None:
        self.walk_run_frames: dict[str, dict[str, torch.Tensor]] = {}
        self.recovery_frames: dict[str, dict[str, torch.Tensor]] = {}

    @classmethod
    def get(cls) -> MotionResetManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(
        self,
        env: ManagerBasedRlEnv,
        motion_dir: str,
        recovery_dir: str | None = None,
    ) -> None:
        if motion_dir in self.walk_run_frames:
            return

        loader = MotionLoader(
            motion_dir=motion_dir,
            tgt_body_indexes=[],
            tgt_anchor_indexes=0,
            feet_indexes=0,
            device=str(env.device),
            recovery_dir=recovery_dir,
        )

        self.walk_run_frames[motion_dir] = self._concat_frames(loader.motion_data)
        motion_count = self.walk_run_frames[motion_dir]["root_pos"].shape[0]
        print(f"[MotionResetManager] Loaded {len(loader.motion_data)} clips, {motion_count} frames from {motion_dir}")

        # Key recovery data by recovery_dir (not motion_dir) so the two pools are explicit.
        if recovery_dir is not None and loader.motion_data_recovery:
            self.recovery_frames[recovery_dir] = self._concat_frames(loader.motion_data_recovery)
            recovery_count = self.recovery_frames[recovery_dir]["root_pos"].shape[0]
            print(f"[MotionResetManager] Loaded {len(loader.motion_data_recovery)} recovery clips, {recovery_count} frames from {recovery_dir}")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        motion_dir: str,
        recovery_dir: str | None = None,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

        if len(env_ids) == 0:
            return

        # Split into delay envs and normal envs.
        delay_mask = self._get_delay_env_mask(env)
        if delay_mask is not None:
            is_delay = delay_mask[env_ids]
            delay_ids = env_ids[is_delay]
            normal_ids = env_ids[~is_delay]
        else:
            delay_ids = env_ids[:0]  # empty
            normal_ids = env_ids

        # Reset normal envs with walk/run data.
        if len(normal_ids) > 0:
            self._write_reset_state(env, normal_ids, self.walk_run_frames[motion_dir], asset_cfg)

        # Reset delay envs with recovery data (fallback to walk/run if unavailable).
        if len(delay_ids) > 0:
            recovery = (
                self.recovery_frames.get(recovery_dir)
                if recovery_dir is not None
                else None
            )
            frames = recovery if recovery is not None else self.walk_run_frames[motion_dir]
            self._write_reset_state(env, delay_ids, frames, asset_cfg)

    def _get_delay_env_mask(self, env: ManagerBasedRlEnv) -> torch.Tensor | None:
        """Get delay env mask from DelayedTerminationManager if installed."""
        tm = env.termination_manager
        if isinstance(tm, DelayedTerminationManager):
            return tm._delay_env_mask
        return None

    def _write_reset_state(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        frames: dict[str, torch.Tensor],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        total_frames = frames["root_pos"].shape[0]
        num_reset = len(env_ids)
        idx = torch.randint(0, total_frames, (num_reset,), device=env.device)

        asset: Entity = env.scene[asset_cfg.name]

        # --- Root pose ---
        root_pos = frames["root_pos"][idx]
        root_quat = frames["root_quat"][idx]
        positions = env.scene.env_origins[env_ids].clone()

        # --- Key Fix for terrain ---
        terrain_z = positions[:, 2].clone()
        positions[:, 2] = terrain_z + root_pos[:, 2]

        root_pose = torch.cat([positions, root_quat], dim=-1)
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

        # --- Root velocity ---
        root_vel = torch.cat([frames["root_lin_vel"][idx], frames["root_ang_vel"][idx]], dim=-1)
        asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

        # --- Joint state ---
        joint_pos = frames["joint_pos"][idx]
        joint_vel = frames["joint_vel"][idx]

        soft_joint_pos_limits = asset.data.soft_joint_pos_limits
        assert soft_joint_pos_limits is not None
        joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
        joint_pos_clamped = joint_pos[:, asset_cfg.joint_ids].clamp_(
            joint_pos_limits[..., 0], joint_pos_limits[..., 1]
        )

        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device)

        asset.write_joint_state_to_sim(
            joint_pos_clamped,
            joint_vel[:, asset_cfg.joint_ids],
            env_ids=env_ids,
            joint_ids=joint_ids,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_frames(motions: list[dict]) -> dict[str, torch.Tensor]:
        root_pos_list = []
        root_quat_list = []
        root_lin_vel_list = []
        root_ang_vel_list = []
        joint_pos_list = []
        joint_vel_list = []
        for motion in motions:
            root_pos_list.append(motion["body_pos_w"][:, 0, :])
            root_quat_list.append(motion["body_quat_w"][:, 0, :])
            root_lin_vel_list.append(motion["body_lin_vel_w"][:, 0, :])
            root_ang_vel_list.append(motion["body_ang_vel_w"][:, 0, :])
            joint_pos_list.append(motion["dof_pos"])
            joint_vel_list.append(motion["dof_vel"])
        return {
            "root_pos": torch.cat(root_pos_list, dim=0),
            "root_quat": torch.cat(root_quat_list, dim=0),
            "root_lin_vel": torch.cat(root_lin_vel_list, dim=0),
            "root_ang_vel": torch.cat(root_ang_vel_list, dim=0),
            "joint_pos": torch.cat(joint_pos_list, dim=0),
            "joint_vel": torch.cat(joint_vel_list, dim=0),
        }


# ------------------------------------------------------------------
# Event callback wrappers (thin delegates to singleton)
# ------------------------------------------------------------------

def init_motion_loader(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    recovery_dir: str | None = None,
    delay_reset_env_ratio: float = 0.0,
    max_delay_steps: int = 0,
) -> None:
    """Startup event: load motion data and optionally install delayed termination."""
    MotionResetManager.get().init(
        env=env,
        motion_dir=motion_dir,
        recovery_dir=recovery_dir,
    )

    # Install DelayedTerminationManager if requested.
    num_delay = int(env.num_envs * delay_reset_env_ratio)
    if num_delay > 0 and max_delay_steps > 0:
        delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        delay_indices = torch.randperm(env.num_envs, device=env.device)[:num_delay]
        delay_mask[delay_indices] = True
        env.termination_manager = DelayedTerminationManager(
            base=env.termination_manager,
            delay_env_mask=delay_mask,
            max_delay_steps=max_delay_steps,
        )
        print(
            "[init_motion_loader] DelayedTerminationManager installed: "
            f"{num_delay}/{env.num_envs} envs, max_delay_steps={max_delay_steps}"
        )


def reset_from_motion_data(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    recovery_dir: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset event: reset envs from random motion frames, with delay support."""
    MotionResetManager.get().reset(
        env=env,
        env_ids=env_ids,
        motion_dir=motion_dir,
        recovery_dir=recovery_dir,
        asset_cfg=asset_cfg,
    )


class apply_recovery_assist_force:
    """Upward assist force while delayed-termination envs are recovering.

    When a delay env's counter first hits ``trigger_delay_steps``, sample
    ``apply_prob`` and (if accepted and ``current_force > 0``) apply a constant
    world-+Z wrench on the selected body for ``duration_steps`` env steps
    (or until the delay ends / env resets, whichever comes first).

    ``current_force`` is decayed by the ``recovery_assist_force`` curriculum.
    Use with ``mode="step"``.
    """

    @dataclass
    class VizCfg:
        """Arrow visualization for the active upward assist."""

        rgba: tuple[float, float, float, float] = (0.15, 0.85, 0.35, 0.95)
        scale: float = 0.004  # meters per Newton
        width: float = 0.02
        min_force: float = 1.0

    def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
        self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        self._body_ids = cfg.params["asset_cfg"].body_ids
        self._num_envs = env.num_envs
        self._device = env.device
        self._viz_cfg: apply_recovery_assist_force.VizCfg = cfg.params.get(
            "viz_cfg", apply_recovery_assist_force.VizCfg()
        )
        self._num_bodies = (
            len(self._body_ids)
            if isinstance(self._body_ids, list)
            else self._asset.num_bodies
        )
        # Curriculum writes this; play keeps the initial value.
        self.current_force: float = float(cfg.params.get("initial_force", 250.0))
        self._duration_steps: int = int(cfg.params.get("duration_steps", 25))
        self._active = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self._remaining = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        trigger_delay_steps: int = 50,
        apply_prob: float = 0.8,
        initial_force: float = 250.0,
        duration_steps: int = 25,
        viz_cfg: VizCfg | None = None,
    ) -> None:
        del env_ids, asset_cfg, initial_force, duration_steps, viz_cfg  # Used at init / via self.

        tm = env.termination_manager
        if not isinstance(tm, DelayedTerminationManager):
            return

        delay_mask = tm._delay_env_mask
        counters = tm._delay_counters

        # Clear when delay ends (recovered or max-delay reset).
        clear = self._active & (counters == 0)
        if clear.any():
            self._clear_force(clear.nonzero(as_tuple=False).squeeze(-1))

        # Tick duration for still-active assists.
        if self._active.any():
            self._remaining[self._active] -= 1
            expired = self._active & (self._remaining <= 0)
            if expired.any():
                self._clear_force(expired.nonzero(as_tuple=False).squeeze(-1))

        if self.current_force <= 0.0:
            if self._active.any():
                self.clear_all()
            return

        # Decide once when the counter first reaches the trigger step.
        candidates = delay_mask & (counters == trigger_delay_steps) & (~self._active)
        if candidates.any():
            cand_ids = candidates.nonzero(as_tuple=False).squeeze(-1)
            accept = torch.rand(len(cand_ids), device=self._device) < apply_prob
            trigger_ids = cand_ids[accept]
            if len(trigger_ids) > 0:
                self._active[trigger_ids] = True
                self._remaining[trigger_ids] = self._duration_steps

        if not self._active.any():
            return

        active_ids = self._active.nonzero(as_tuple=False).squeeze(-1)
        self._write_upward_force(active_ids, self.current_force)

    def _write_upward_force(self, env_ids: torch.Tensor, force_n: float) -> None:
        n = len(env_ids)
        forces = torch.zeros((n, self._num_bodies, 3), device=self._device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = force_n  # world +Z
        self._asset.write_external_wrench_to_sim(
            forces, torques, env_ids=env_ids, body_ids=self._body_ids
        )

    def _clear_force(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        zeros = torch.zeros((len(env_ids), self._num_bodies, 3), device=self._device)
        self._asset.write_external_wrench_to_sim(
            zeros, zeros, env_ids=env_ids, body_ids=self._body_ids
        )
        self._active[env_ids] = False
        self._remaining[env_ids] = 0

    def clear_all(self) -> None:
        if self._active.any():
            self._clear_force(self._active.nonzero(as_tuple=False).squeeze(-1))

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        if isinstance(env_ids, slice):
            active = self._active.nonzero(as_tuple=False).squeeze(-1)
        else:
            active = env_ids[self._active[env_ids]]
        if len(active) > 0:
            self._clear_force(active)
        self._active[env_ids] = False
        self._remaining[env_ids] = 0

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        if not self._active.any():
            return
        viz = self._viz_cfg
        min_sq = viz.min_force * viz.min_force
        wrench = self._asset.data.body_external_wrench  # (E, B, 6)
        com_pos = self._asset.data.body_com_pos_w
        body_ids = self._body_ids
        if isinstance(body_ids, slice):
            body_ids = list(range(wrench.shape[1]))
        elif not isinstance(body_ids, list):
            body_ids = list(body_ids)

        for env_idx in visualizer.get_env_indices(self._num_envs):
            if not self._active[env_idx]:
                continue
            for body_i in body_ids:
                force = wrench[env_idx, body_i, :3]
                if (force * force).sum().item() < min_sq:
                    continue
                start_np = com_pos[env_idx, body_i].cpu().numpy()
                force_np = force.cpu().numpy()
                visualizer.add_arrow(
                    start=start_np,
                    end=start_np + force_np * viz.scale,
                    color=viz.rgba,
                    width=viz.width,
                )
