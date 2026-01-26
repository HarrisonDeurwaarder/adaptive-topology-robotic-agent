import torch
from typing import Any, Sequence

from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg

from configs.python.scene_cfg import SceneCfg
from configs.python.env_cfg import EnvironmentCfg

from utils.config import config


class Environment(DirectRLEnv):
    """
    RL Environment
    """
    def __init__(self,) -> None:
        super().__init__(EnvironmentCfg(), None)
        # Get robot from scene
        self.robot = self.scene["robot"]
        self.cube = self.scene["cube"]
        #self.contact_forces = self.scene["contact_sensor"]
        self.ee_idx = self.robot.find_bodies("panda_link7")[0][0]
        
        
    def _pre_physics_step(
        self,
        actions: torch.Tensor,
    ) -> None:
        """
        Prepares the actions before the decimated physics loop

        Args:
            actions (torch.Tensor): joint efforts
        """
        self.actions = actions
        
        
    def _apply_action(self,) -> None:
        """
        Apply actions in physics loop, [decimation] times
        """
        # Write plain joint efforts
        self.robot.set_joint_effort_target(self.actions,)
        # Update robot buffers
        self.robot.update(self.physics_dt,)
        self.robot.write_data_to_sim()
        
    
    def _get_observations(self,) -> torch.Tensor:
        """
        Gets the observations for all environments

        Returns:
            torch.Tensor: Joint positions and velocities; end-effector contact measurement
        """
        # Get joint localization
        joint_pos: torch.Tensor = self.robot.data.joint_pos
        joint_vel: torch.Tensor = self.robot.data.joint_vel
        # Get end-effector contact forces
        #contact_forces: torch.Tensor = self.contact_forces.data.net_forces_w
        return torch.cat((
                joint_pos,
                joint_vel,
                #contact_forces,
            ),
            dim=1,
        )
    
    
    def _get_rewards(self,) -> torch.Tensor:
        """
        Gets the rewards for all environments

        Returns:
            torch.Tensor: Rewards
        """
        # Compute locations in world frame, then distance
        ee_pos: torch.Tensor = self.robot.data.body_pos_w[:, self.ee_idx]
        cube_pos = self.cube.data.root_link_pose_w[:, 0:3]
        inv_dist: torch.Tensor = 1.0 / torch.sqrt(
            torch.sum(torch.square(ee_pos - cube_pos), dim=1)
        )
        # Extract height of cube
        dist_from_plane: torch.Tensor = cube_pos[:, 2]
        # Apply sparse bonus if height exceeds threshold
        is_above_threshold: torch.Tensor = dist_from_plane >= config["env"]["rewards"]["sparse_height_bonus_threshold"]
        
        # Get episode length
        episode_length: torch.Tensor = self.episode_length_buf
        # Compute mag of ee velocity
        ee_vel_mag: torch.Tensor = torch.linalg.vector_norm(self.robot.data.joint_vel, dim=1)
        return config["env"]["rewards"]["inverse_dist_coef"] * inv_dist + config["env"]["rewards"]["cube_height_coef"] * dist_from_plane + config["env"]["rewards"]["sparse_height_bonus_coef"] * is_above_threshold + config["env"]["rewards"]["passive_penalty_coef"] * episode_length + config["env"]["rewards"]["ee_vel_penalty_coef"] * ee_vel_mag
        
        
    def _get_dones(self,) -> tuple:
        """
        Gets the boolean episode completion flags for all environments
        
        Returns:
            tuple[bool, bool]: A tuple containing the (terminated, truncated) completion flags
        """
        # Truncation term
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Return both
        return torch.full_like(torch.empty(self.num_envs), False, device=self.device), time_out # Termination will be excluded during debugging
    
    
    def _reset_idx(
        self,
        env_ids: Sequence[int] | None,
    ) -> None:
        """
        Reset specified indicies

        Args:
            env_ids (Sequence[int] | None): Indicies to reset
        """
        super()._reset_idx(env_ids)
        # Resolve env ids
        if env_ids is None:
            env_ids = self.robot._ALL_INDICIES
        # Reset robot state
        default_joint_pos = self.robot.data.default_joint_pos.clone()
        default_joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        self.robot.write_data_to_sim()
        self.robot.reset()
        self.robot.update(self.physics_dt,)