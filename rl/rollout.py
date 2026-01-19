import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset


class Rollout(Dataset):
    """
    Rollout for transition storage
    """
    def __init__(
        self,
        initial_state: torch.Tensor,
    ) -> None:
        super().__init__()
        # Define objects
        self.states = torch.tensor([initial_state])
        self.actions = torch.empty()
        self.distributions = torch.empty()
        self.rewards = torch.empty()
        
        
    def __len__(self,) -> int:
        return self.actions.size(0)
    
    
    def __getitem__(
        self,
        idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.distributions.Normal, float]:
        # Return transition tuple
        return (
            self.states[idx],
            self.states[idx+1],
            self.actions[idx],
            self.distributions[idx],
            self.rewards[idx],
        )