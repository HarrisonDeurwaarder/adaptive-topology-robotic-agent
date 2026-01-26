import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

from sim.launch_app import launch_app
from utils.config import load_config

sim_app, args_cli = launch_app()
load_config("configs\\yaml\\train.yaml")

import isaaclab.sim as sim_utils
from isaaclab.controllers import OperationalSpaceController
from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationContext, SimulationCfg

from configs.python.scene_cfg import SceneCfg
from configs.python.env_cfg import EnvironmentCfg
from sim.environment import Environment
from utils.config import config

from rl.ppo import Actor, Critic
from rl.rollout import Rollout
import traceback

def main() -> None:
    """
    Main function ran on file execution
    """
    # Create the environment
    env: Environment = Environment()
    # Load the simulation
    sim: sim_utils.SimulationContext = sim_utils.SimulationContext(env.cfg.sim,)
    sim.reset()
    # Extract the scene and robot
    scene: InteractiveScene = env.scene
    robot: Articulation = scene["robot"]
    device: torch.device = sim.device
    
    sim_dt: float = sim.get_physics_dt()
    robot.update(dt=sim_dt,)
    
    # Create RL objects
    policy: Actor = Actor().to(device=device)
    value: Critic = Critic().to(device=device)
    
    optimizer: Adam = Adam([
            {"params": policy.parameters(), "lr": config["rl"]["ppo"]["policy_lr"]},
            {"params": value.parameters(), "lr": config["rl"]["ppo"]["value_lr"]},
        ],
        maximize=True,
    )
    
    """ Training Loop """
    # Epoch = num rollouts collected
    for iteration in range(config["rl"]["iterations"]):
        # Rollout collection phase
        obs: torch.Tensor = env.reset()[0]
        rollout: Rollout = Rollout(
            initial_obs=obs,
            initial_value_out=value(obs,),
            device=device,
        )
        # Loop until rollout is at capacity
        while len(rollout) < config["rl"]["rollout_length"]:
            # Sample action from policy
            mean, variance = policy(obs,)
            action: torch.Tensor = Normal(mean, variance,).sample()
            # Step in the environment
            obs, rew, term, trunc, _, = env.step(action,)
            # Add to rollout
            rollout.add(
                obs, action, mean, variance, rew, value(obs,), term | trunc,
            )
        
        # Batch rollout
        dataloader: DataLoader = DataLoader(
            dataset=rollout,
            batch_size=config["rl"]["batch_size"],
            shuffle=True,
        )
        
        # Training phase
        rews_h, dones_h, value_outs_h = rollout.get_horizon()
        # Compute advantages
        advantages: torch.Tensor = policy.gae(
            rewards=rews_h,
            dones=dones_h,
            value_outs=value_outs_h,
        )
        rollout.add_advantages(advantages,)
        
        for _ in range(config["rl"]["epochs"]):
            for obs, actions, old_means, old_variances, advantages, old_value_outs in dataloader:
                optimizer.zero_grad()
                # Compute advantages and loss
                value_outs: torch.Tensor = value(obs,).squeeze(2)
                means, variances = policy(obs,)
                policy_dist: Normal = Normal(means, variances)
                
                policy_loss: torch.Tensor = policy.policy_objective(
                    policy_dist=policy_dist,
                    old_policy_dist=Normal(old_means, old_variances),
                    actions=actions,
                    advantages=advantages,
                )
                value_loss: torch.Tensor = value.value_loss(
                    value_outs=value_outs,
                    old_value_outs=old_value_outs,
                    advantages=advantages,
                )
                
                # Backpropagate
                loss: torch.Tensor = policy_loss - config["rl"]["ppo"]["value_coef"] * value_loss + config["rl"]["ppo"]["entropy_coef"] * policy_dist.entropy().mean()
                loss.backward()
                optimizer.step()
                
        print(
            f"Iteration #{iteration+1}/{config['rl']['iterations']} Completed:",
            f"\tMean Reward: {rollout.rewards.mean():.2f}",
            f"\tMean Episode Length: {((1 - rollout.dones).sum()) / rollout.dones.sum():.2f}",
            f"\tMean Advantage: {rollout.advantages.mean():.2f}",
            f"\tMean Value Loss: {Critic.value_loss(value(rollout.obs[:-1],).squeeze(2), rollout.value_outs[:-1], rollout.advantages,).mean():.2f}",
            f"\tMean Policy Loss: {Actor.policy_objective(Normal(*policy(rollout.obs[:-1])), Normal(rollout.means, rollout.variances), rollout.actions, rollout.advantages).mean():.2f}\n",
            sep="\n",
        )
        

if __name__ == "__main__":
    main()