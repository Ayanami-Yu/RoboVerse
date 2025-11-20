"""BeyondMimic motion tracking task package."""

# Import the registered RoboVerse task to ensure it's discovered
from roboverse_pack.tasks.beyondmimic.tasks.tracking.tracking_task_rv_registered import (
    TrackingTaskRVRegistered,
)

__all__ = ["TrackingTaskRVRegistered"]
