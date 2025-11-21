"""Registered RoboVerse task for BeyondMimic motion tracking."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from roboverse_pack.tasks.beyondmimic.tracking.tracking_task_rv import TrackingTaskCfg, TrackingTaskRV


@dataclass
class DefaultTrackingScenarioCfg(ScenarioCfg):
    """Default scenario configuration for tracking task."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # Set default simulator if not specified
        if self.simulator is None:
            self.simulator = "isaaclab"
        # Set default decimation
        if self.decimation == 25:  # Default from ScenarioCfg
            self.decimation = 4  # TODO check appropriate value


@register_task("beyondmimic.tracking", "beyondmimic_tracking")
class TrackingTaskRVRegistered(TrackingTaskRV):
    """Registered tracking task for RoboVerse."""

    # Default scenario configuration
    scenario: ScenarioCfg = DefaultTrackingScenarioCfg(
        robots=["g1_dof29"],  # TODO check whether this works with BeyondMimic
        num_envs=4096,
        env_spacing=2.5,
        decimation=4,
        simulator="isaaclab",
        # TODO specify `sim_params` according to `WalkG1Dof29Task` or not
    )

    # Default task configuration
    task_cfg: TrackingTaskCfg = TrackingTaskCfg(
        episode_length_s=10.0,
        max_episode_steps=2000,
        body_names=list(body_names=scenario.robots[0].actuators.keys()),  # TODO add "pelvis" into `body_names` or not
    )

    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        task_cfg: TrackingTaskCfg | None = None,
        device: str | None = None,
    ):
        """Initialize the registered task.

        Args:
            scenario: Scenario configuration (uses class default if None)
            task_cfg: Task configuration (uses class default if None)
            device: Device to use
        """
        # TODO check if deepcopy is necessary
        if scenario is None:
            scenario = copy.deepcopy(self.scenario)
        if task_cfg is None:
            task_cfg = copy.deepcopy(self.task_cfg)

        super().__init__(scenario=scenario, task_cfg=task_cfg, device=device)
