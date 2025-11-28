import os
import wandb

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from roboverse_pack.tasks.beyondmimic.tracking.tracking_g1 import TrackingG1Task
from roboverse_learn.rl.beyondmimic.helper.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MotionOnPolicyRunner(OnPolicyRunner):  # only used for training
    def __init__(
        self, env: TrackingG1Task, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            # TODO adapte `attach_onnx_metadata()` and uncomment the following line
            # attach_onnx_metadata(self.env, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
