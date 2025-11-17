import os
import argparse
import datetime
import importlib
from loguru import logger as log

import random
import torch
import numpy as np

from metasim.utils.setup_util import get_robot
from metasim.utils.string_util import is_camel_case, is_snake_case, to_camel_case


def make_robots(robots_str: str) -> list[any]:
    robot_names = robots_str.split()
    robots = []
    for _name in robot_names:
        robots.append(get_robot(_name))
    return robots


def make_objects(objects_str: str) -> list[any]:
    object_names = objects_str.split()
    objects = []
    for _name in object_names:
        objects.append(
            get_class(
                _name,
                suffix="Cfg",
                library="roboverse_learn.rl.unitree_rl.configs.cfg_objects",
            )()
        )
    return objects


def get_class(name: str, suffix: str, library="roboverse_learn.rl.unitree_rl"):
    """Get the class wrappers.
    Example:
        get_class("ReachOrigin", "Cfg") -> ReachOriginCfg
        get_class("reach_origin", "Cfg") -> ReachOriginCfg
    """
    if is_camel_case(name):
        task_name_camel = name
    elif is_snake_case(name):
        task_name_camel = to_camel_case(name)

    wrapper_module = importlib.import_module(library)
    wrapper_cls = getattr(wrapper_module, f"{task_name_camel}{suffix}")
    return wrapper_cls


def get_log_dir(task_name: str, now=None) -> str:
    """Get the log directory."""
    if now is None:
        now = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
    log_dir = f"./outputs/unitree_rl/{task_name}/{now}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log.info("Log directory: {}", log_dir)
    return log_dir


def get_load_path(load_root: str, checkpoint: int | str = None) -> str:
    """Get the path to load the model from."""
    if isinstance(checkpoint, int):
        if checkpoint == -1:
            models = [
                file
                for file in os.listdir(load_root)
                if "model" in file and file.endswith(".pt")
            ]
            models.sort(key=lambda m: f"{m!s:0>15}")
            model = models[-1]
            load_path = f"{load_root}/{model}"
        else:
            load_path = f"{load_root}/model_{checkpoint}.pt"
    else:
        load_path = f"{load_root}/{checkpoint}.pt"
    log.info(f"Loading checkpoint {checkpoint} from {load_root}")
    return load_path


def get_args():
    """Get the command line arguments."""
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "walk_g1_dof29",
            "help": "Task name for training/testing.",
        },
        {"name": "--robots", "type": str, "default": "", "help": "The used robots."},
        {
            "name": "--objects",
            "type": str,
            "default": None,
            "help": "The used objects.",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": 128,
            "help": "number of parallel environments.",
        },
        {
            "name": "--iter",
            "type": int,
            "default": 15000,
            "help": "Max number of training iterations.",
        },
        {
            "name": "--sim",
            "type": str,
            "default": "isaacgym",
            "help": "simulator type, currently only isaacgym is supported",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": True,
            "help": "Force display off at all times",
        },
        {
            "name": "--resume",
            "type": str,
            "default": None,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "default": -1,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "default": -1,
            "help": "The random seed for the run. If -1, will be randomly generated.",
        },
        {
            "name": "--eval",
            "action": "store_true",
            "default": False,
            "help": "Whether to run in eval mode",
        },
        {
            "name": "--jit_load",
            "action": "store_true",
            "default": False,
            "help": "Whether to load the JIT model",
        },
    ]
    args = parse_arguments(custom_parameters=custom_parameters)
    return args


def parse_arguments(description="humanoid rl task arguments", custom_parameters=None):
    """Parse command line arguments."""
    if custom_parameters is None:
        custom_parameters = []
    parser = argparse.ArgumentParser(description=description)
    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(
                        argument["name"],
                        type=argument["type"],
                        default=argument["default"],
                        help=help_str,
                    )
                else:
                    parser.add_argument(
                        argument["name"], type=argument["type"], help=help_str
                    )
            elif "action" in argument:
                parser.add_argument(
                    argument["name"], action=argument["action"], help=help_str
                )

        else:
            log.error(
                "ERROR: command line argument name, type/action must be defined, argument not added to parser"
            )
            log.error("supported keys: name, type, default, action, help")

    return parser.parse_args()


def set_seed(seed=-1):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    log.info(f"Setting seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
