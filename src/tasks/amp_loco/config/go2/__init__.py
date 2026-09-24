from mjlab.tasks.registry import register_mjlab_task
from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs import (
  go2_amp_flat_env_cfg,
  go2_amp_rough_env_cfg,
)
from .rl_cfg import go2_amp_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-AMP-Rough",
  env_cfg=go2_amp_rough_env_cfg(),
  play_env_cfg=go2_amp_rough_env_cfg(play=True),
  rl_cfg=go2_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-AMP-Flat",
  env_cfg=go2_amp_flat_env_cfg(),
  play_env_cfg=go2_amp_flat_env_cfg(play=True),
  rl_cfg=go2_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)
