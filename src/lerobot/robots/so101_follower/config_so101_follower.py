#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("so101_follower")
@dataclass
class SO101FollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str = ""

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False


@RobotConfig.register_subclass("so101_follower_end_effector")
@dataclass
class SO101FollowerEndEffectorConfig(SO101FollowerConfig):
    """Configuration for the SO101FollowerEndEffector robot."""

    # Path to URDF file for kinematics
    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo:
    # https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    urdf_path: str | None = None

    # End-effector frame name in the URDF
    target_frame_name: str = "gripper_frame_link"

    # cameras: dict[str, CameraConfig]  | None = None

    # Default bounds for the end-effector position (in meters)
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-1.0, -1.0, -1.0],  # min x, y, z
            "max": [1.0, 1.0, 1.0],  # max x, y, z
        }
    )

    max_gripper_pos: float = 50

    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.02,
            "y": 0.02,
            "z": 0.02,
        }
    )
    # calibration directory and mock flag (consistent with other configs)
    calibration_dir: str = ".cache/calibration/so101"


# --- add this to src/lerobot/robots/config.py (Style A) ---
# @RobotConfig.register_subclass("so101_follower_end_effector")
# @dataclass
# class So101FollowerEndEffectorConfig_Old(RobotConfig):
#     """
#     Minimal config for so101_follower_end_effector to allow draccus decoding.
#     Matches keys used in your JSON: port, urdf_path, target_frame_name, cameras,
#     end_effector_bounds and end_effector_step_sizes.
#     """
#     # connection / urdf
#     port: str | None = None
#     urdf_path: str | None = None
#     target_frame_name: str | None = None

#     # Cameras: be permissive and accept dict; draccus will decode nested CameraConfig if present
#     cameras: dict[str, CameraConfig]  | None = None

#     # end-effector workspace bounds: {"min": [...], "max": [...]}
#     end_effector_bounds: dict | None = None

#     # step sizes: {"x": 0.025, "y": 0.025, "z": 0.025}
#     end_effector_step_sizes: dict | None = None

#     # calibration directory and mock flag (consistent with other configs)
#     calibration_dir: str = ".cache/calibration/so101"
#     mock: bool = False
# --- end snippet ---
