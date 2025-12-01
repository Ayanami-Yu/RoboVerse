from __future__ import annotations

import os
import copy
import pathlib
import torch
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import get_task_class

from roboverse_learn.rl.beyondmimic.wrappers import MasterRunner
from roboverse_learn.rl.beyondmimic.helper.utils import get_args, make_objects, make_robots, set_seed, get_checkpoint_path
from roboverse_learn.rl.beyondmimic.helper.exporter import export_motion_policy_as_onnx, attach_onnx_metadata


def play(args):
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
        else:
            art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
            if art is None:
                print("[WARN] No model artifact found in the run.")
            else:
                args.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
    else:
        log_root_path = os.path.join("logs", "rsl_rl", args.exp_name)
        resume_path = get_checkpoint_path(log_root_path, args.load_run, args.checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    master_runner = MasterRunner(
        env_cls=task_cls,
        scenario=scenario,
        lib_name="rsl_rl",
        device=device,
        args=args,
    )

    # obtain the trained policy for inference
    name = list(master_runner.runners.keys())[0]
    policy = master_runner.load(resume_path)[name]
    runner = master_runner.runners[name]
    env = list(master_runner.envs.values())[0]

    # export policy to ONNX
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_motion_policy_as_onnx(
        env,
        runner.runner.alg.policy,
        normalizer=runner.runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    # attach_onnx_metadata(  # TODO adapt this
    #     env,
    #     args.wandb_path if args.wandb_path else "none",
    #     export_model_dir,
    # )
    print(f"[INFO]: Exported policy as ONNX to : {export_model_dir}")

    # reset environment and simulate
    env.reset()
    obs, _, _, _ = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    # obs = runner.env_wrapper.get_observations()  # TODO remove this

    # TODO check if `torch.inference_mode()` is on
    for _ in range(1000000):
        actions = policy(obs)
        obs, _, _, _ = runner.env_wrapper.step(actions)


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    play(args)
