# BeyondMimic to RoboVerse Migration Summary

This document summarizes the migration of BeyondMimic's evaluation code to the RoboVerse framework.

## Overview

The migration adapts BeyondMimic's motion tracking evaluation code to use RoboVerse's simulator-agnostic APIs, enabling evaluation across different simulators (Isaac Lab, Isaac Gym, MuJoCo, PyBullet, etc.) while keeping the training code unchanged.

## Key Changes

### 1. New RoboVerse-Compatible Task Class

**File**: `roboverse_pack/tasks/beyondmimic/tasks/tracking/tracking_task_rv.py`

- Created `TrackingTaskRV` class that inherits from `RLTaskEnv` (RoboVerse's base RL task class)
- Uses `ScenarioCfg` for simulator-agnostic configuration
- Implements observation, reward, and termination computation using RoboVerse's state interface

### 2. Adapted MDP Components

**Directory**: `roboverse_pack/tasks/beyondmimic/tasks/tracking/mdp_rv/`

Created new RoboVerse-compatible versions of:
- **commands.py**: `MotionCommandRV` - Works with RoboVerse's state handler instead of Isaac Lab managers
- **observations.py**: Observation computation functions that use `TensorState` interface
- **rewards.py**: Reward computation functions adapted for RoboVerse
- **terminations.py**: Termination computation functions adapted for RoboVerse

### 3. Task Registration

**File**: `roboverse_pack/tasks/beyondmimic/tasks/tracking/tracking_task_rv_registered.py`

- Registered task with name `"beyondmimic.tracking"` for discovery by RoboVerse's task registry
- Provides default scenario and task configurations

### 4. New Evaluation Script

**File**: `roboverse_learn/rl/beyondmimic/scripts/play_rv.py`

- New evaluation script that uses RoboVerse APIs
- Supports multiple simulators via `--simulator` argument
- Compatible with RSL-RL for policy loading and inference
- Supports loading from local checkpoints or WandB

### 5. RSL-RL Wrapper

**File**: `roboverse_learn/rl/beyondmimic/runners/rsl_rl_rv.py`

- `RslRlVecEnvWrapperRV` - Wraps RoboVerse environments for RSL-RL compatibility
- Provides the interface expected by RSL-RL's `OnPolicyRunner`

## File Structure

```
roboverse_pack/tasks/beyondmimic/
├── tasks/
│   └── tracking/
│       ├── tracking_task_rv.py              # Main RoboVerse task class
│       ├── tracking_task_rv_registered.py   # Registered task
│       ├── mdp_rv/                           # RoboVerse-compatible MDP components
│       │   ├── __init__.py
│       │   ├── commands.py
│       │   ├── observations.py
│       │   ├── rewards.py
│       │   └── terminations.py
│       └── ... (original Isaac Lab files remain unchanged)

roboverse_learn/rl/beyondmimic/
├── scripts/
│   ├── play.py          # Original Isaac Lab evaluation (unchanged)
│   └── play_rv.py       # New RoboVerse evaluation script
└── runners/
    └── rsl_rl_rv.py     # RSL-RL wrapper for RoboVerse
```

## Usage

### Evaluation with RoboVerse

```bash
python roboverse_learn/rl/beyondmimic/scripts/play_rv.py \
    --task beyondmimic.tracking \
    --simulator isaaclab \
    --num_envs 1 \
    --motion_file /path/to/motion.npz \
    --resume /path/to/checkpoint/dir \
    --checkpoint last
```

### Supported Simulators

- `isaaclab` (default)
- `isaacgym`
- `mujoco`
- `pybullet`
- `sapien2`
- `sapien3`
- `genesis`

### Loading from WandB

```bash
python roboverse_learn/rl/beyondmimic/scripts/play_rv.py \
    --task beyondmimic.tracking \
    --simulator isaaclab \
    --wandb_path <wandb_entity>/<wandb_project>/<run_id>
```

## Important Notes

1. **Training Code Unchanged**: The original training code (`train.py`) and all Isaac Lab-specific components remain unchanged. Training still uses Isaac Lab.

2. **Evaluation Only**: The RoboVerse adaptation is specifically for evaluation. The training pipeline continues to use Isaac Lab.

3. **Shared Classes**: If classes are shared between training and evaluation and need functional changes, new evaluation-specific classes were created (e.g., `MotionCommandRV` vs `MotionCommand`).

4. **State Interface**: The RoboVerse version uses `TensorState` from `metasim.types` instead of Isaac Lab's manager-based interface.

5. **Configuration**: Task configuration is now done via `TrackingTaskCfg` dataclass and `ScenarioCfg` for simulator-agnostic setup.

## Differences from Original

1. **No Isaac Lab Managers**: The RoboVerse version doesn't use Isaac Lab's command/observation/reward/termination managers. Instead, it directly computes from `TensorState`.

2. **State Handler**: Uses RoboVerse's `BaseSimHandler` interface instead of direct Isaac Lab scene access.

3. **Simulator Agnostic**: Can run on any simulator supported by RoboVerse, not just Isaac Lab.

4. **Simplified Interface**: The evaluation script is simpler and doesn't require Isaac Sim to be launched separately.

## Testing

To test the migration:

1. Ensure you have a trained model checkpoint
2. Have a motion file (.npz format)
3. Run the evaluation script with appropriate arguments
4. Verify that the policy runs correctly and produces expected behavior

## Future Improvements

- Add support for video recording in RoboVerse evaluation
- Enhance adaptive sampling to work with termination information from RoboVerse
- Add more comprehensive error handling and validation
- Support for multi-robot scenarios if needed





