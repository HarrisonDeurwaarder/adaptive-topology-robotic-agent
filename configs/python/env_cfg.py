from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg

from configs.python.scene_cfg import SceneCfg
from utils.config import config


@configclass
class EnvironmentCfg(DirectRLEnvCfg):
    """
    RL environment configuration
    """
    # Environment config
    decimation: int = config["env"]["decimation"]
    episode_length_s: float = config["env"]["episode_length"]
    action_space: int = config["env"]["action_space"]
    observation_space: int = config["env"]["obs_space"]
    
    # Simulation config
    sim: SimulationCfg = SimulationCfg(
        dt=config["sim"]["dt"],
        render_interval=config["sim"]["render_interval"],
    )
    
    # Scene config
    scene: SceneCfg = SceneCfg(
        num_envs=config["scene"]["num_envs"],
        env_spacing=config["scene"]["env_spacing"],
        replicate_physics=config["scene"]["replicate_physics"],
        clone_in_fabric=config["scene"]["clone_in_fabric"],
    )