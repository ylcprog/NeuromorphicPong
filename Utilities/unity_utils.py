"""
Utility functions for neural controller systems.

This module contains shared functionality for both regular Nengo and SPA-based
neural controllers that interface with Unity ML-Agents environments and ONNX models.

Dependencies:
    - numpy: Numerical computations
    - mlagents-envs: Unity ML-Agents environment interface
    - onnxruntime: ONNX model inference

Author: [Your name]
Date: [Current date]
"""

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from Utilities.ANN_to_SNN_conversion import OBS_LENGTH



# =============================================================================
# UNITY ENVIRONMENT UTILITIES
# =============================================================================

def initialize_unity_environment(game_file_path, screen_width=811, screen_height=456,
                                 capture_frame_rate=0, worker_id=0, no_graphics=False, time_scale=1):
    """
    Initialize Unity ML-Agents environment with standard configuration.

    Args:
        game_file_path (str): Path to Unity executable
        screen_width (int): Screen width for Unity window
        screen_height (int): Screen height for Unity window
        capture_frame_rate (int): Frame rate for capturing
        worker_id (int): Worker ID for multiple environments

    Returns:
        tuple: (env, behavior_name)
    """
    print("Starting Unity environment...")
    env = UnityEnvironment(
        file_name=game_file_path,
        no_graphics=no_graphics,
        worker_id=worker_id,
        #log_folder="/RG/rg-tsur/yisraelc/Pong/pong_logs",
        additional_args=[
            "-screen-width", str(screen_width),
            "-screen-height", str(screen_height),
            "-timeScale", str(time_scale),
            "-capture-frame-rate", str(capture_frame_rate)
        ]
    )

    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    print(f"Environment initialized with behavior: {behavior_name}")

    return env, behavior_name


def unity_environment_step(env, behavior_name, action):
    """
    Execute one step in the Unity environment and return observation and reward.

    Args:
        env: Unity environment instance
        behavior_name (str): Name of the behavior to control
        action: Action to execute (will be wrapped in ActionTuple)

    Returns:
        tuple: (observation, reward) where observation is numpy array of shape (45,)
    """
    # Only set actions if there are active agents
    if len(env.get_steps(behavior_name)[0]) > 0:
        env.set_actions(behavior_name, ActionTuple(np.array([[action]])))

    # Step the environment
    env.step()

    # Get observation and reward from decision steps
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    if len(decision_steps) == 0:
        return np.zeros(OBS_LENGTH), 0.0  # Safe fallback if no agents active

    # Extract observation and reward
    observation = decision_steps[0].obs[0]  # Full observation vector (shape: [OBS_LENGTH])
    reward = decision_steps[0].reward  # Scalar reward

    return observation, reward

