import torch
from typing import Any, Sequence

from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg

from configs.python.scene_cfg import SceneCfg
from configs.python.env_cfg import EnvironmentCfg


class Environment(DirectRLEnv):
    """
    RL Environment
    """
    def __init__(self,) -> None:
        super().__init__(EnvironmentCfg(), None)
        # Get robot from scene
        self.robot = self.scene["robot"]
        #self.contact_forces = self.scene["contact_sensor"]
        
    
    def _step_impl(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Advances the episode to the next step

        Args:
            actions (torch.Tensor): joint efforts

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]: A tuple containing:
                - obs: policy observations at step
                - rewards: netted rewards at step
                - terminated: tensor of boolean flags indicating if the episode has been terminated
                - truncated: tensor of boolean flags indicating if the episode has been truncated
                - info: isaaclab information at the step
        """
        obs, rewards, terminated, truncated, info = super()._step_impl(actions,)
        # Update robot buffers
        self.robot.update(self.physics_dt)
        return (
            obs, rewards, terminated, truncated, info,
        )
        
        
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
        self.robot.set_joint_effort_target(
            self.actions,
            joint_ids=...,
        )
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
        return torch.ones((self.scene.num_envs), device=self.device)
        
        
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