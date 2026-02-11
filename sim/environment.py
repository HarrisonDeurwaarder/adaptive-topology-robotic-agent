import torch
import torch.nn.functional as F
from typing import Any, Sequence

import isaaclab.sim as sim_utils

from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
from isaaclab.sim import SimulationCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from configs.python.scene_cfg import SceneCfg
from configs.python.env_cfg import EnvironmentCfg

from utils.config import config
from utils.math import vect_to_quat


class Environment(DirectRLEnv):
    """
    RL Environment
    """
    def __init__(self,) -> None:
        super().__init__(EnvironmentCfg(), None)
        # Get entities from scene
        self.robot = self.scene["robot"]
        self.cube = self.scene["cube"]
        self._define_markers()
        #self.contact_forces = self.scene["contact_sensor"]
        self.ee_idx = self.robot.find_bodies("panda_link7")[0][0]
        # Local forward for computing orientation quats
        self.local_forward: torch.Tensor = torch.tensor([1.0, 0.0, 0.0], device=self.sim.device,).repeat((self.num_envs, 1),)
        # Reward reporting
        self._episode_rewards = {
            key: torch.zeros((self.num_envs), device=self.sim.device)
            for key in ["orient_alignment", "cube_height", "sparse_height_bonus", "ee_prismatic_vel", "cube_z_vel", "dist_penalty", "passive_penalty"]
        }
        
        
    def _pre_physics_step(
        self,
        actions: torch.Tensor,
    ) -> None:
        """
        Prepares the actions before the decimated physics loop

        Args:
            actions (torch.Tensor): joint efforts
        """
        self.actions = torch.concat((
                actions,
                actions[..., -1:], # Duplicate the ee joint effort (for homogeneous movement across the two fingers)
            ),
            dim=1,
        )
        
        self._compute_intermediate_values()
        
        # Place markers above ee
        marker_pos: torch.Tensor = self.ee_pos
        marker_pos[:, 2] += config["scene"]["marker"]["translation"]
        # Compute marker orientation quat
        target_vect_orient: torch.Tensor = F.normalize(
            (self.cube_pos - self.ee_pos),
            dim=1
        )
        target_quat: torch.Tensor = vect_to_quat(target_vect_orient, self.local_forward)
        
        ee_tool_axis = torch.tensor([0.0, 0.0, 1.0], device=self.sim.device).repeat((self.num_envs, 1))
        ee_tool_vect = quat_apply(self.ee_quat, ee_tool_axis)
        ee_tool_quat = vect_to_quat(ee_tool_vect, ee_tool_axis)
        
        # Update markers
        self.markers.visualize(
            translations=marker_pos.repeat((2, 1),)[0:self.num_envs, :],
            orientations=torch.concat(
                (ee_tool_quat,),
                dim=0,
            ),
        )
        
        
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
        joint_pos: torch.Tensor = self.robot.data.joint_pos[..., :-1]
        joint_vel: torch.Tensor = self.robot.data.joint_vel[..., :-1]
        # Get end-effector contact forces
        #contact_forces: torch.Tensor = self.contact_forces.data.net_forces_w
        return torch.cat((
                joint_pos,
                joint_vel,
                #contact_forces,
            ),
            dim=1,
        )
        
        
    def _compute_intermediate_values(self,) -> None:
        """
        Computes values needed for rewards and marker visualization
        """
        # Compute locations in world frame, then distance
        self.ee_pos: torch.Tensor = self.robot.data.body_pos_w[:, self.ee_idx]
        self.cube_pos = self.cube.data.root_link_pose_w[:, 0:3]
        # Compute quats
        self.ee_quat: torch.Tensor = self.robot.data.body_quat_w[:, self.ee_idx]
        
        self.ee_orient: torch.Tensor = F.normalize(quat_apply(self.ee_quat, self.local_forward,), dim=1,) # Quaternion applied to a unit vector should result in a unit vector, but floating-point errors can accumulate
        self.targ_orient: torch.Tensor = F.normalize((self.cube_pos - self.ee_pos), dim=1,)
        
    
    
    def _get_rewards(self,) -> torch.Tensor:
        """
        Gets the rewards for all environments

        Returns:
            torch.Tensor: Rewards
        """
        # Compute orientation and target orientation of ee
        orient_alignment: torch.Tensor = torch.sum(self.ee_orient * self.targ_orient, dim=1,)
        # Extract height of cube
        dist_from_plane: torch.Tensor = self.cube_pos[:, 2]
        # Apply sparse bonus if height exceeds threshold
        is_above_threshold: torch.Tensor = dist_from_plane >= config["env"]["rewards"]["sparse_height_bonus_threshold"]
        # Get position of ee joint
        ee_joint_vel: torch.Tensor = self.robot.data.joint_vel[:, 2]
        # Extract z velocity of cube
        cube_z_vel: torch.Tensor = self.cube.data.root_lin_vel_w[:, 2]
        
        dist: torch.Tensor = 1.0 / torch.sqrt(
            torch.sum(torch.square(self.ee_pos - self.cube_pos), dim=1)
        )
        # Get episode length
        episode_length: torch.Tensor = self.episode_length_buf
        
        rewards = {
            "orient_alignment": config["env"]["rewards"]["orient_alignment_coef"] * orient_alignment,
            "cube_height": config["env"]["rewards"]["cube_height_coef"] * dist_from_plane,
            "sparse_height_bonus": config["env"]["rewards"]["sparse_height_bonus_coef"] * is_above_threshold,
            "ee_prismatic_vel": config["env"]["rewards"]["ee_prismatic_vel_coef"] * ee_joint_vel,
            "cube_z_vel": config["env"]["rewards"]["cube_z_vel_coef"] * cube_z_vel,
            "dist_penalty": config["env"]["rewards"]["dist_penalty_coef"] * dist,
            "passive_penalty": config["env"]["rewards"]["passive_penalty_coef"] * episode_length,
        }
        # Save rewards for logging
        for key, reward in rewards.items():
            self._episode_rewards[key] += reward
        # Cumulative reward
        return torch.sum(
            torch.stack(tuple(rewards.values()), dim=1,),
            dim=1,
        )
        
        
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
        
    
    def _define_markers(self,) -> None:
        """
        Create orientation markers
        """
        markers_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
            prim_path="/Visuals/goal_marker",
            markers={
                "ee_orientation": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=tuple(config["scene"]["marker"]["scale"]),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
                #"targ_orientation": sim_utils.UsdFileCfg(
                #    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                #    scale=tuple(config["scene"]["marker"]["scale"]),
                #    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                #),
            },
        )
        self.markers = VisualizationMarkers(markers_cfg,)