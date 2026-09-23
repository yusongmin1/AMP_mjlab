"""Curriculum terms for AMP locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def recovery_assist_force(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str = "recovery_assist_force",
  initial_force: float = 250.0,
  force_decay: float = 20.0,
  decay_interval_iters: int = 500,
  steps_per_iter: int = 24,
) -> torch.Tensor:
  """Decay the recovery upward-assist force over training iterations.

  Force schedule:
    F(iter) = max(0, initial_force - force_decay * floor(iter / decay_interval_iters))

  where ``iter = common_step_counter // steps_per_iter`` (one PPO iteration).
  When F reaches 0 the step event stops applying assist forces.
  """
  del env_ids  # Unused.
  iteration = env.common_step_counter // steps_per_iter
  stages = iteration // decay_interval_iters
  force = max(0.0, initial_force - force_decay * stages)

  try:
    term_cfg = env.event_manager.get_term_cfg(event_name)
  except ValueError:
    return torch.tensor([force])

  func = term_cfg.func
  if hasattr(func, "current_force"):
    prev = float(func.current_force)
    func.current_force = force
    if force <= 0.0 and prev > 0.0 and hasattr(func, "clear_all"):
      func.clear_all()
    if abs(prev - force) > 1e-6:
      print(
        f"[recovery_assist_force] iter={iteration} force={force:.1f}N "
        f"(decay {force_decay}N every {decay_interval_iters} iters)"
      )

  return torch.tensor([force])
