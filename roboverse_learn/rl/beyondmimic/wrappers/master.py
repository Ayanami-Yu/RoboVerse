from __future__ import annotations

import copy
import os
import pickle as pkl
import torch
from datetime import datetime
from argparse import Namespace
from typing import Type

from metasim.scenario.scenario import ScenarioCfg
from roboverse_learn.rl.beyondmimic.configs.cfg_base import BaseEnvCfg
from roboverse_learn.rl.beyondmimic.helper.utils import get_class, get_checkpoint_path
from roboverse_pack.tasks.beyondmimic.base.types import EnvTypes

from .runners import RslRlWrapper


class MasterRunner:
    def __init__(
        self,
        task_cls: Type[EnvTypes],
        scenario: ScenarioCfg,
        lib_name: str = "rsl_rl",
        device: str | torch.device = "cuda",
        args: Namespace = None,
    ):
        self.task_cls = task_cls
        self.task_name = getattr(task_cls, "task_name", task_cls.__name__)
        self.runners = {}
        self.envs = {}
        self.scenario = scenario

        task_cfg_cls: Type[BaseEnvCfg] = getattr(task_cls, "env_cfg_cls", BaseEnvCfg)
        train_cfg_cls = getattr(task_cls, "train_cfg_cls", None)
        runner_cls = get_class(lib_name, suffix="Wrapper", library="roboverse_learn.rl.beyondmimic.wrappers")

        robot_cfgs = scenario.robots if isinstance(scenario.robots, list) else [scenario.robots]
        for robot in robot_cfgs:
            scenario_copy = copy.deepcopy(scenario)
            scenario_copy.robots = [robot]
            scenario_copy.__post_init__()

            task_cfg = task_cfg_cls()
            task: EnvTypes = task_cls(
                scenario=scenario_copy,
                device=device,
                env_cfg=task_cfg,
            )
            train_cfg = train_cfg_cls()

            # specify directory for logging experiments
            log_root_path = os.path.join("logs", "rsl_rl", args.exp_name)
            self.log_root_path = os.path.abspath(log_root_path)
            print(f"[INFO] Logging experiment in directory: {self.log_root_path}")

            # specify directory for logging runs: {time-stamp}_{run_name}
            log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if args.run_name:
                log_dir += f"_{args.run_name}"
            log_dir = os.path.join(log_root_path, log_dir)

            runner: RslRlWrapper = runner_cls(env=task, train_cfg=train_cfg, log_dir=None if args.eval else log_dir)
            self.runners[task.robot.name] = runner
            self.envs[task.robot.name] = task

            # log task and train configs
            params_path = f"{log_dir}/params"
            os.makedirs(params_path, exist_ok=True)
            pkl.dump(task_cfg, open(f"{params_path}/task_cfg.pkl", "wb"))
            pkl.dump(train_cfg, open(f"{params_path}/train_cfg.pkl", "wb"))

    # FIXME having multiple runners but only using the first one
    def learn(self, max_iterations=30000):
        if not self.runners:
            raise RuntimeError("No runners instantiated for training.")
        first_runner = next(iter(self.runners.values()))
        first_runner.learn(max_iterations=max_iterations)

    def load(self, resume_path: str):
        """Load trained checkpoints for evaluation."""
        self.policies = {}
        for _robot_name, _runner in self.runners.items():
            _runner.load(resume_path)
            # self.policies[_robot_name] = _runner.get_policy()  # TODO difference between `get_inference_policy()`?
            self.policies[_robot_name] = _runner.get_inference_policy()
        return self.policies
