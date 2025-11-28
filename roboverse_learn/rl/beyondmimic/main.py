from __future__ import annotations

import os
import copy
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

from roboverse_pack.tasks.beyondmimic.base.types import EnvTypes
from roboverse_learn.rl.beyondmimic.wrappers import EnvWrapperTypes, MasterRunner
from roboverse_learn.rl.beyondmimic.helper import get_args, make_objects, make_robots, set_seed, get_checkpoint_path
from roboverse_learn.rl.beyondmimic.helper.exporter import export_motion_policy_as_onnx, attach_onnx_metadata


def prepare(args):
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

    master_runner = MasterRunner(
        task_cls=task_cls,
        scenario=scenario,
        lib_name="rsl_rl",
        device=device,
    )
    return master_runner


def play(args):
    master_runner = prepare(args)
    env = list(master_runner.envs.values())[0]

    # get resume path and set motion file
    if args.wandb_path:
        import wandb

        run_path = args.wandb_path

        api = wandb.Api()
        # if specific model file provided, extract run path
        if "model" in args.wandb_path:
            run_path = "/".join(args.wandb_path.split("/")[:-1])
            # e.g., "org/project/run_id/model_1000.pt" yields "org/project/run_id"

        wandb_run = api.run(run_path)
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        if "model" in args.wandb_path:
            file = args.wandb_path.split("/")[-1]  # use specified model file
        else:
            # files are formatted as model_xxx.pt, find the largest filename (max iter)
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)
        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = os.path.join("logs", "rsl_rl", "temp", file)

        if args.motion_file:
            print(f"[INFO]: Using motion file from CLI: {args.motion_file}")
            env.cfg.commands.motion.motion_file = args.motion_file
        else:
            art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
            if art is None:
                print("[WARN] No model artifact found in the run.")
            else:
                env.cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
    else:
        resume_path = get_checkpoint_path(master_runner.log_root_path, args.load_run, args.checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    name = list(master_runner.runners.keys())[0]
    policy = master_runner.load(resume_path)[name]
    runner = master_runner.runners[name]

    # export policy to ONNX
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_motion_policy_as_onnx(
        env,
        runner.alg.policy,
        normalizer=runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    # attach_onnx_metadata(  # TODO adapt this
    #     env,
    #     args.wandb_path if args.wandb_path else "none",
    #     export_model_dir,
    # )
    print(f"[INFO]: Exported policy as ONNX to : {export_model_dir}")

    env: EnvTypes = runner.env
    env_wrapper: EnvWrapperTypes = runner.env_wrapper

    # reset environment and simulate
    env.reset()
    obs, _, _, _ = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    # obs = env_wrapper.get_observations()  # TODO remove this

    # TODO check if `torch.inference_mode()` is on
    for _ in range(1000000):
        actions = policy(obs)
        obs, _, _, _ = env_wrapper.step(actions)


def train(args):
    master_runner = prepare(args)
    env = list(master_runner.envs.values())[0]

    # load the motion file from the wandb registry
    registry_name = args.registry_name
    # check if the registry name includes alias, if not, append ":latest"
    if ":" not in registry_name:
        registry_name += ":latest"

    import pathlib
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)

    # load from WandB and set motion file path
    env.cfg.commands.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    # resume training from previous checkpoint
    if args.resume:
        resume_path = get_checkpoint_path(master_runner.log_root_path, args.load_run, args.checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        master_runner.load(resume_path)

    master_runner.learn(max_iterations=args.max_iterations)


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    if args.eval:
        play(args)
    else:
        train(args)
