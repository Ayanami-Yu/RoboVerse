from __future__ import annotations

import re
import os
import copy
import argparse
import importlib
from loguru import logger as log
from functools import lru_cache

import random
import torch
import numpy as np

from metasim.utils.setup_util import get_robot
from metasim.utils.string_util import is_camel_case, is_snake_case, to_camel_case
from metasim.scenario.scenario import ScenarioCfg
from roboverse_pack.tasks.beyondmimic.base.types import EnvTypes


def get_args():
    """Get the command line arguments."""
    parser = argparse.ArgumentParser(description="Arguments for BeyondMimic motion tracking task")
    parser.add_argument("--task", type=str, default=None, help="Name of the task")
    parser.add_argument("--robots", type=str, default=None, help="Names of the robots to use")
    parser.add_argument("--objects", type=str, default=None, help="Names of the objects to use")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--sim", type=str, default="isaacsim", help="Simulator type")
    parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")

    # for logging
    parser.add_argument(
        "--exp_name", type=str, default=None, help="Name of the experiment folder where logs will be stored"
    )
    parser.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory")
    parser.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    parser.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using WandB or neptune"
    )

    # for loading
    parser.add_argument("--resume", type=bool, default=False, help="Whether to resume from a checkpoint. Should only be used for training")
    parser.add_argument("--load_run", type=str, default=None, help="Name of the local folder to resume from if not using WandB")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from if not using WandB. Format: model_xxx.pt")

    # evaluation args
    parser.add_argument("--eval", action="store_true", default=False, help="Run in evaluation mode")
    parser.add_argument(
        "--motion_file", type=str, default=None, help="Path to the local motion file"
    )
    parser.add_argument(
        "--wandb_path", type=str, default=None, help="The WandB run path to load from. Format: org/project/run_id(/model_xxx.pt)"
    )

    # training args
    parser.add_argument("--max_iterations", type=int, default=None, help="Max number of training iterations")
    parser.add_argument("--registry_name", type=str, default=None, help="Name of the WandB registry")  # TODO should be required

    return parser.parse_args()


def set_seed(seed: int | None = None):
    """Seed will be randomly initialized if it's None."""
    if not seed:
        seed = np.random.randint(0, 10000)
    log.info(f"Setting seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_class(name: str, suffix: str, library="roboverse_learn.rl.beyondmimic"):
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


def get_indexes_from_substring(
    candidates_list: list[str] | tuple[str] | str,
    data_base: list[str],
    fullmatch: bool = True,
) -> torch.Tensor:
    """Get indexes of items matching the candidates patterns.

    Args:
        candidates_list: Single pattern or list of patterns (supports regex if use_regex=True)
        data_base: List of names to search in
        use_regex: If True, treat candidates as regex patterns. If False, use substring matching.

    Returns:
        Sorted tensor of matching indexes

    Examples:
        >>> get_indexes_from_substring(".*ankle.*", ["left_ankle", "right_ankle", "knee"])
        tensor([0, 1])
        >>> get_indexes_from_substring([".*ankle.*", ".*knee.*"], ["left_ankle", "knee"])
        tensor([0, 1])
    """
    found_indexes = []
    if isinstance(candidates_list, str):
        candidates_list = (candidates_list,)
    assert isinstance(
        candidates_list, (list, tuple)
    ), "candidates_list must be a list, tuple or string."

    for candidate in candidates_list:
        # compile regex pattern for efficiency
        try:
            pattern = re.compile(candidate)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{candidate}': {e}")

        for i, name in enumerate(data_base):
            if fullmatch and pattern.fullmatch(name):
                found_indexes.append(i)
            elif not fullmatch and pattern.search(name):
                found_indexes.append(i)

    # remove duplicates and sort
    found_indexes = sorted(set(found_indexes))
    return torch.tensor(found_indexes, dtype=torch.int32, requires_grad=False)


@lru_cache(maxsize=128)
def hash_names(names: str | tuple[str]) -> str:
    if isinstance(names, str):
        names = (names,)
    assert isinstance(names, tuple) and all(
        isinstance(_, str) for _ in names
    ), "body_names must be a string or a list of strings."
    hash_key = "_".join(sorted(names))
    return hash_key


def get_indexes(
    env: EnvTypes, sub_names: tuple[str] | str, all_names: list[str] | tuple[str]
):
    hash_key = hash_names(sub_names)
    if hash_key not in env.extras_buffer:
        env.extras_buffer[hash_key] = get_indexes_from_substring(
            sub_names, all_names, fullmatch=True
        ).to(env.device)
    return env.extras_buffer[hash_key]


def pattern_match(sub_names: dict[str, any], all_names: list[str]) -> dict[str, any]:
    """Pattern match the sub_names to all_names using regex."""
    matched_names = {_key: 0.0 for _key in all_names}
    for sub_key, sub_val in sub_names.items():
        pattern = re.compile(sub_key)
        for name in all_names:
            if pattern.fullmatch(name):
                matched_names[name] = sub_val
    return matched_names


def get_axis_params(value, axis_idx, x_value=0.0, n_dims=3):
    """Construct arguments to `Vec` according to axis index."""
    zs = torch.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = torch.where(zs == 1.0, value, zs)
    params[0] = x_value
    return params.tolist()


# copied from `isaaclab_tasks.utils.parse_cfg.py`

def get_checkpoint_path(
    log_path: str, run_dir: str = ".*", checkpoint: str = ".*", other_dirs: list[str] = None, sort_alpha: bool = True
) -> str:
    """Get path to the model checkpoint in input directory.

    The checkpoint file is resolved as: ``<log_path>/<run_dir>/<*other_dirs>/<checkpoint>``, where the
    :attr:`other_dirs` are intermediate folder names to concatenate. These cannot be regex expressions.

    If :attr:`run_dir` and :attr:`checkpoint` are regex expressions then the most recent (highest alphabetical order)
    run and checkpoint are selected. To disable this behavior, set the flag :attr:`sort_alpha` to False.

    Args:
        log_path: The log directory path to find models in.
        run_dir: The regex expression for the name of the directory containing the run. Defaults to the most
            recent directory created inside :attr:`log_path`.
        other_dirs: The intermediate directories between the run directory and the checkpoint file. Defaults to
            None, which implies that checkpoint file is directly under the run directory.
        checkpoint: The regex expression for the model checkpoint file. Defaults to the most recent
            torch-model saved in the :attr:`run_dir` directory.
        sort_alpha: Whether to sort the runs by alphabetical order. Defaults to True.
            If False, the folders in :attr:`run_dir` are sorted by the last modified time.

    Returns:
        The path to the model checkpoint.

    Raises:
        ValueError: When no runs are found in the input directory.
        ValueError: When no checkpoints are found in the input directory.

    """
    # check if runs present in directory
    try:
        # find all runs in the directory that math the regex expression
        runs = [
            os.path.join(log_path, run) for run in os.scandir(log_path) if run.is_dir() and re.match(run_dir, run.name)
        ]
        # sort matched runs by alphabetical order (latest run should be last)
        if sort_alpha:
            runs.sort()
        else:
            runs = sorted(runs, key=os.path.getmtime)
        # create last run file path
        if other_dirs is not None:
            run_path = os.path.join(runs[-1], *other_dirs)
        else:
            run_path = runs[-1]
    except IndexError:
        raise ValueError(f"No runs present in the directory: '{log_path}' match: '{run_dir}'.")

    # list all model checkpoints in the directory
    model_checkpoints = [f for f in os.listdir(run_path) if re.match(checkpoint, f)]
    # check if any checkpoints are present
    if len(model_checkpoints) == 0:
        raise ValueError(f"No checkpoints in the directory: '{run_path}' match '{checkpoint}'.")
    # sort alphabetically while ensuring that *_10 comes after *_9
    model_checkpoints.sort(key=lambda m: f"{m:0>15}")
    # get latest matched checkpoint file
    checkpoint_file = model_checkpoints[-1]

    return os.path.join(run_path, checkpoint_file)
