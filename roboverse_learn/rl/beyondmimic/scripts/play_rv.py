"""Script to play/evaluate a checkpoint of an RL agent using RoboVerse framework."""

from __future__ import annotations

import argparse
import copy
import os
import pathlib
import sys

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import get_task_class
from rsl_rl.runners import OnPolicyRunner

# Try to import CLI args helpers, but provide fallback
try:
    from roboverse_learn.rl.beyondmimic.helper.cli_args import add_rsl_rl_args, parse_rsl_rl_cfg
except ImportError:
    # Fallback if not available
    def add_rsl_rl_args(parser):
        """Add RSL-RL arguments."""
        pass

    def parse_rsl_rl_cfg(task_name, args):
        """Parse RSL-RL config."""
        # Return a minimal config dict
        return type("Config", (), {
            "to_dict": lambda: {},
            "experiment_name": getattr(args, "experiment_name", "beyondmimic_tracking"),
        })()
from roboverse_learn.rl.beyondmimic.runners.rsl_rl_rv import RslRlVecEnvWrapperRV
from roboverse_pack.tasks.beyondmimic.tasks.tracking.tracking_task_rv import TrackingTaskCfg


def get_checkpoint_path(log_root_path: str, load_run: str, load_checkpoint: str) -> str:
    """Get path to checkpoint file.

    Args:
        log_root_path: Root path for logs
        load_run: Run name to load
        load_checkpoint: Checkpoint name to load

    Returns:
        Path to checkpoint file
    """
    if load_checkpoint == "last":
        # Find the latest checkpoint
        run_path = os.path.join(log_root_path, load_run)
        checkpoints = [f for f in os.listdir(run_path) if f.startswith("model_") and f.endswith(".pt")]
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {run_path}")
        checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
        return os.path.join(run_path, checkpoint)
    else:
        return os.path.join(log_root_path, load_run, f"model_{load_checkpoint}.pt")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL using RoboVerse.")
    parser.add_argument("--task", type=str, default="beyondmimic.tracking", help="Name of the task.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--simulator", type=str, default="isaaclab", help="Simulator to use.")
    parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode.")
    parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory.")
    parser.add_argument("--checkpoint", type=str, default="last", help="Checkpoint to load (default: last).")
    parser.add_argument("--wandb_path", type=str, default=None, help="WandB run path to load model from.")

    # Add RSL-RL arguments
    add_rsl_rl_args(parser)

    args = parser.parse_args()

    # Get task class
    task_cls = get_task_class(args.task)

    # Get scenario template from task class
    scenario_template = getattr(task_cls, "scenario", ScenarioCfg())
    scenario = copy.deepcopy(scenario_template)

    # Override scenario settings
    scenario.num_envs = args.num_envs
    scenario.simulator = args.simulator
    scenario.headless = args.headless

    # Get task configuration
    task_cfg = TrackingTaskCfg()
    if args.motion_file is not None:
        task_cfg.motion_file = args.motion_file

    # Determine device
    device = "cpu" if args.simulator == "mujoco" else ("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment first (needed for config)
    env = task_cls(scenario=scenario, task_cfg=task_cfg, device=device)

    # Parse RSL-RL configuration
    try:
        agent_cfg = parse_rsl_rl_cfg(args.task, args)
    except Exception:
        # Fallback: create minimal config
        class MinimalConfig:
            def __init__(self, env):
                self.experiment_name = getattr(args, "experiment_name", "beyondmimic_tracking")
                self.num_actor_obs = env.num_obs
                self.num_critic_obs = env.num_priv_obs
                self.num_actions = env.num_actions

            def to_dict(self):
                return {
                    "experiment_name": self.experiment_name,
                    "num_actor_obs": self.num_actor_obs,
                    "num_critic_obs": self.num_critic_obs,
                    "num_actions": self.num_actions,
                }

        agent_cfg = MinimalConfig(env)

    # Wrap for RSL-RL
    env_wrapper = RslRlVecEnvWrapperRV(env, train_cfg=agent_cfg.to_dict())

    # Load checkpoint
    if args.wandb_path:
        import wandb

        run_path = args.wandb_path
        api = wandb.Api()
        if "model" in args.wandb_path:
            run_path = "/".join(args.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        if "model" in args.wandb_path:
            file = args.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)
        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        # Get motion file from artifact if available
        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is not None:
            task_cfg.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
            env.motion_command = None  # Force reinitialization
            env._initialize_motion_command()
    else:
        if args.resume is None:
            raise ValueError("Either --resume or --wandb_path must be provided.")
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        resume_path = get_checkpoint_path(log_root_path, args.resume, args.checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Create runner and load checkpoint
    train_cfg = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else agent_cfg
    runner = OnPolicyRunner(
        env_wrapper, train_cfg, log_dir=None, device=device
    )
    runner.load(resume_path)

    # Get inference policy
    policy = runner.get_inference_policy(device=device)

    # Reset environment
    obs, priv_obs = env_wrapper.reset()
    obs_dict = env_wrapper.get_observations()

    # Evaluation loop
    timestep = 0
    max_steps = args.video_length if args.video else 10000

    print("[INFO]: Starting evaluation...")
    try:
        while timestep < max_steps:
            # Get actions from policy
            with torch.inference_mode():
                actions = policy(obs_dict["policy"])

            # Step environment
            obs, priv_obs, rewards, dones, infos = env_wrapper.step(actions)
            obs_dict = env_wrapper.get_observations()

            timestep += 1

            if timestep % 100 == 0:
                print(f"[INFO]: Step {timestep}/{max_steps}")

    except KeyboardInterrupt:
        print("[INFO]: Evaluation interrupted by user.")

    print("[INFO]: Evaluation completed.")
    env.close()


if __name__ == "__main__":
    main()
