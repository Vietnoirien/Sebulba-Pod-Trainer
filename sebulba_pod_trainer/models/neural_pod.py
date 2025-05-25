import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Any

class PodNetwork(nn.Module):
    """
    Neural network for controlling a pod in the race.
    Takes observations as input and outputs actions.
    """
    def __init__(self, 
                 observation_dim: int = 47,  # Updated for new observation format
                 hidden_layers: List[Dict[str, Any]] = None,
                 policy_hidden_size: int = 12,  # Reduced from 16
                 value_hidden_size: int = 12,   # Reduced from 16
                 special_hidden_size: int = 12): # Reduced from 16
        super().__init__()

        self.input_dim = observation_dim
        
        # Default hidden layers if none provided - SMALLER SIZES
        if hidden_layers is None:
            hidden_layers = [
                {'type': 'Linear+ReLU', 'size': 24},  # Reduced from 32
                {'type': 'Linear+ReLU', 'size': 16}   # Reduced from 32
            ]
        
        # Ensure we have at least one hidden layer
        if len(hidden_layers) == 0:
            hidden_layers = [
                {'type': 'Linear+ReLU', 'size': 24}  # Reduced from 32
            ]
        
        # Build encoder layers dynamically
        encoder_layers = []
        input_size = observation_dim
        
        for layer_config in hidden_layers:
            layer_type = layer_config['type']
            layer_size = layer_config['size']
            
            # Add linear layer
            encoder_layers.append(nn.Linear(input_size, layer_size))
            
            # Add activation function
            if layer_type == 'Linear+ReLU':
                encoder_layers.append(nn.ReLU())
            elif layer_type == 'Linear+Tanh':
                encoder_layers.append(nn.Tanh())
            elif layer_type == 'Linear+Sigmoid':
                encoder_layers.append(nn.Sigmoid())
            # Linear only has no activation
            
            # Update input size for next layer
            input_size = layer_size
        
        # Create encoder sequential module
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Get the output size of the encoder
        encoder_output_size = hidden_layers[-1]['size']
        
        # Policy head (outputs angle and thrust) - SMALLER
        self.policy_head = nn.Sequential(
            nn.Linear(encoder_output_size, policy_hidden_size),
            nn.ReLU(),
            nn.Linear(policy_hidden_size, 2),  # angle and thrust
        )
        
        # Value head (estimates expected reward) - SMALLER
        self.value_head = nn.Sequential(
            nn.Linear(encoder_output_size, value_hidden_size),
            nn.ReLU(),
            nn.Linear(value_hidden_size, 1),
        )
        
        # Special action head (for SHIELD and BOOST decisions) - SMALLER
        self.special_action_head = nn.Sequential(
            nn.Linear(encoder_output_size, special_hidden_size),
            nn.ReLU(),
            nn.Linear(special_hidden_size, 2),  # shield and boost logits
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            x: Observation tensor
            
        Returns:
            actions: Tensor with [angle, thrust] values
            value: Estimated value of the state
            special_probs: Probabilities for special actions [shield_prob, boost_prob]
        """
        # Encode observations
        encoded = self.encoder(x)
        
        # Get policy outputs
        raw_actions = self.policy_head(encoded)
        
        # Process actions
        angle = torch.tanh(raw_actions[:, 0:1])  # Range: [-1, 1]
        thrust = torch.sigmoid(raw_actions[:, 1:2])  # Range: [0, 1]
        
        # Combine into action tensor
        actions = torch.cat([angle, thrust], dim=1)
        
        # Get value estimate
        value = self.value_head(encoded)
        
        # Get special action probabilities
        special_logits = self.special_action_head(encoded)
        special_probs = torch.sigmoid(special_logits)
        
        return actions, value, special_probs
    
    def get_actions(self, 
                   observations: torch.Tensor, 
                   deterministic: bool = False) -> torch.Tensor:
        """
        Get actions from observations, with optional exploration
        
        Args:
            observations: Batch of observations
            deterministic: If True, return deterministic actions, otherwise sample
            
        Returns:
            actions: Tensor with [angle, thrust] values
        """
        actions, _, special_probs = self.forward(observations)
        
        if not deterministic:
            # Add exploration noise to angle
            angle_noise = torch.randn_like(actions[:, 0:1]) * 0.1
            actions[:, 0:1] = torch.clamp(actions[:, 0:1] + angle_noise, -1.0, 1.0)
            
            # Sample special actions
            shield_prob = special_probs[:, 0:1]
            boost_prob = special_probs[:, 1:2]
            
            # Apply shield with probability
            shield_action = torch.bernoulli(shield_prob)
            actions[:, 1:2] = torch.where(
                shield_action > 0.5,
                torch.tensor(-1.0, device=actions.device),
                actions[:, 1:2]
            )
            
            # Apply boost with probability (if not shielding)
            boost_action = torch.bernoulli(boost_prob) * (1.0 - shield_action)
            actions[:, 1:2] = torch.where(
                boost_action > 0.5,
                torch.tensor(1.0, device=actions.device),
                actions[:, 1:2]
            )
        
        return actions
