from __future__ import annotations

import os
import copy
import wandb
import pathlib
import torch
import rootutils
from datetime import datetime

rootutils.setup_root(__file__, pythonpath=True)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import get_task_class

from roboverse_learn.rl.beyondmimic.wrappers import MasterRunner
from roboverse_learn.rl.beyondmimic.helper.utils import get_args, make_objects, make_robots, set_seed, get_checkpoint_path


def train(args):
    task_cls = get_task_class(args.task)
    scenario_template = getattr(task_cls, "scenario", ScenarioCfg())
    scenario = copy.deepcopy(scenario_template)

    overrides = {
        "num_envs": args.num_envs,
        "simulator": args.sim,
        "headless": args.headless,
    }
    if args.robots:
        overrides["robots"] = make_robots(args.robots)
        overrides["cameras"] = [
            camera
            for robot in overrides["robots"]
            if hasattr(robot, "cameras")
            for camera in getattr(robot, "cameras", [])
        ]
    if args.objects:
        overrides["objects"] = make_objects(args.objects)

    scenario.update(**overrides)

    device = "cpu" if args.sim == "mujoco" else ("cuda" if torch.cuda.is_available() else "cpu")

    # load the motion file from the wandb registry
    registry_name = args.registry_name
    # check if the registry name includes alias, if not, append ":latest"
    if ":" not in registry_name:
        registry_name += ":latest"

    api = wandb.Api()
    artifact = api.artifact(registry_name)

    # load from WandB and set motion file path
    args.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    master_runner = MasterRunner(
        env_cls=task_cls,
        scenario=scenario,
        lib_name="rsl_rl",
        device=device,
        args=args,
    )

    # resume training from previous checkpoint
    if args.resume:
        resume_path = get_checkpoint_path(master_runner.log_root_path, args.load_run, args.checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        master_runner.load(resume_path)

    master_runner.learn(max_iterations=args.max_iterations)


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    train(args)
