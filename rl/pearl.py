import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class InferenceNetwork(nn.Module):
    """
    Inference network (probabalistic encoder) for PEARL algorithm
    """
    def __init__(self,) -> None:
        self.net: nn.Sequential = nn.Sequential(
            ...
        )
    
    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        # Context as the input for the inference network
        context: torch.Tensor = torch.cat(state, next_state, action, reward, dim=1)
        # Get output gaussian parameters
        mean, logvar = torch.chunk(
            self.net(context), 2, dim=1
        )
        # Use reparameterization for gradient-friendly sample
        out: torch.Tensor = mean + torch.exp(logvar) * Normal(0.0, 1.0)
        return out