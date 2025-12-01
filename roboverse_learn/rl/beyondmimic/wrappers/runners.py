from roboverse_pack.tasks.beyondmimic.base import AgentTask
from roboverse_pack.tasks.beyondmimic.base.types import EnvTypes

from .environment import RslRlVecEnvWrapper
from .on_policy_runner import MotionOnPolicyRunner  # TODO check if this is also ok for evaluation


class BaseWrapper:
    def __init__(self, env: EnvTypes, train_cfg: dict, log_dir: str):
        self.env = env
        self.device = env.device
        if not isinstance(train_cfg, dict):
            train_cfg = train_cfg.to_dict()
        self.train_cfg = train_cfg
        self.log_dir = log_dir

    def load(self, path):
        raise NotImplementedError

    def learn(self, max_iterations):
        raise NotImplementedError

    def get_policy(self):
        raise NotImplementedError


class RslRlWrapper(BaseWrapper):
    def __init__(self, env: AgentTask, train_cfg: dict, log_dir: str, registry_name: str = None):
        super().__init__(env, train_cfg, log_dir)

        self.env_wrapper = RslRlVecEnvWrapper(self.env)
        self.runner = MotionOnPolicyRunner(
            env=self.env_wrapper,
            train_cfg=self.train_cfg,
            device=self.device,
            log_dir=log_dir,
            registry_name=registry_name,
        )

    def learn(self, max_iterations=30000):
        self.runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

    def load(self, path):
        self.runner.load(path)

    def get_policy(self):
        return self.runner.get_inference_policy()
