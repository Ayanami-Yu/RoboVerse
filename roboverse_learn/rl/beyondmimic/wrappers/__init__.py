from typing import Union

from .master import MasterRunner
from .environment import RslRlVecEnvWrapper
from .runners import RslRlWrapper

EnvWrapperTypes = Union[RslRlVecEnvWrapper]
