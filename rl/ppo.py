import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import config


class Actor(nn.Module):
    """
    Policy function for PPO
    """
    def __init__(self,) -> None:
        super().__init__()
        self.net: nn.Module = nn.Sequential(
            nn.Linear(18, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 18),
        )
        
        
    def forward(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the action distribution for the policy

        Args:
            obs (torch.Tensor): Observations

        Returns:
            torch.Tensor: Distribution parameters
        """
        out = self.net(obs,)
        # Split last dimension into mean/logstd
        return (
            out[..., 0:9],
            torch.exp(
                torch.clamp(out[..., 9:18], config["rl"]["ppo"]["log_std_min"], config["rl"]["ppo"]["log_std_max"])
            ),
        )
        
    
    @classmethod
    def gae(
        cls,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        value_outs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes GAE-derived advantages

        Args:
            rewards (torch.Tensor): Rewards at every step
            dones (torch.Tensor): Episode termination flags at every step
            value_outs (torch.Tensor): Predicted values of the policy

        Returns:
            torch.Tensor: GAE advantages
        """
        # Compute TD residuals
        td_residuals: torch.Tensor = rewards + config["rl"]["ppo"]["discount_factor"] * (1 - dones) * value_outs[1:, ...] - value_outs[:-1, ...]
        # Compute advantages
        advantages: torch.Tensor = torch.zeros_like(rewards) # T+1
        for t in reversed(range(td_residuals.size(0) - 1)):
            advantages[t, ...] = td_residuals[t, ...] + config["rl"]["ppo"]["discount_factor"] * config["rl"]["ppo"]["gae_decay"] * (1 - dones[t, ...]) * advantages[t, ...]
        
        return advantages
    
    
    @classmethod
    def policy_objective(
        cls,
        policy_dist: torch.distributions.Normal,
        old_policy_dist: torch.distributions.Normal,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the clipped surrogate objective for PPO

        Args:
            policy_dist (torch.distributions.Normal): Output distribution
            old_policy_dist (torch.distributions.Normal): Output distribution of the target policy
            actions (torch.Tensor): Sampled actions
            advantages (torch.Tensor): GAE advantages

        Returns:
            tuple[torch.Tensor, torch.Tensor]: a tuple containing the distribution parameters
        """
        advantages = advantages.detach()
        # Compute policy ratio of selected action
        policy_ratio: torch.Tensor = torch.exp(
            torch.sum(policy_dist.log_prob(actions), dim=2) - torch.sum(old_policy_dist.log_prob(actions).detach(), dim=2),
        )
        # Apply ratio scaling
        policy_objecive: torch.Tensor = torch.minimum(
            advantages * policy_ratio,
            advantages * torch.clip(
                policy_ratio,
                1 - config["rl"]["ppo"]["clipping_param"],
                1 + config["rl"]["ppo"]["clipping_param"],
            ),
        )
        return policy_objecive.mean()
    
    
class Critic(nn.Module):
    """
    Value function for PPO
    """
    def __init__(self,) -> None:
        super().__init__()
        self.net: nn.Module = nn.Sequential(
            nn.Linear(18, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        
        
    def forward(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the expected value for the policy

        Args:
            obs (torch.Tensor): Observations

        Returns:
            torch.Tensor: Expected value
        """
        out: torch.Tensor = self.net(obs,)
        # Split last dimension into mean/logstd
        return out.squeeze(1)
    
    
    @classmethod
    def value_loss(
        cls,
        value_outs: torch.Tensor,
        old_value_outs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the value loss for PPO

        Args:
            value_outs (torch.Tensor): Predicted value of the policy
            old_value_outs (torch.Tensor): Old predicted value
            advantages (torch.Tensor): GAE advantages

        Returns:
            torch.Tensor: Value loss
        """
        return F.mse_loss(
            value_outs,
            (advantages + old_value_outs).detach(),
        )