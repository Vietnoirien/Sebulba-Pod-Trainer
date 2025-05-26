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
                 observation_dim: int = 56,  # Updated for new observation format
                 hidden_layers: List[Dict[str, Any]] = None,
                 policy_hidden_size: int = 12,  # Reduced from 16
                 value_hidden_size: int = 12,   # Reduced from 16
                 action_hidden_size: int = 12,  # Renamed from special_hidden_size
                 special_hidden_size: int = None):  # Keep for backward compatibility
        super().__init__()

        # Handle backward compatibility
        if special_hidden_size is not None:
            action_hidden_size = special_hidden_size

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
        
        # Action head (outputs all 4 action components) - UPDATED
        self.action_head = nn.Sequential(
            nn.Linear(encoder_output_size, action_hidden_size),
            nn.ReLU(),
            nn.Linear(action_hidden_size, 4),  # angle, thrust, shield_prob, boost_prob
        )
        
        # Value head (estimates expected reward) - SMALLER
        self.value_head = nn.Sequential(
            nn.Linear(encoder_output_size, value_hidden_size),
            nn.ReLU(),
            nn.Linear(value_hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            x: Observation tensor
            
        Returns:
            actions: Tensor with [angle, thrust, shield_prob, boost_prob] values
            value: Estimated value of the state
            special_probs: Probabilities for special actions [shield_prob, boost_prob] (for backward compatibility)
        """
        # Encode observations
        encoded = self.encoder(x)
        
        # Get action outputs
        raw_actions = self.action_head(encoded)
        
        # Process actions
        angle = torch.tanh(raw_actions[:, 0:1])  # Range: [-1, 1]
        thrust = torch.sigmoid(raw_actions[:, 1:2])  # Range: [0, 1]
        shield_prob = torch.sigmoid(raw_actions[:, 2:3])  # Range: [0, 1]
        boost_prob = torch.sigmoid(raw_actions[:, 3:4])  # Range: [0, 1]
        
        # Combine into action tensor
        actions = torch.cat([angle, thrust, shield_prob, boost_prob], dim=1)
        
        # Get value estimate
        value = self.value_head(encoded)
        
        # Extract special action probabilities for backward compatibility
        special_probs = torch.cat([shield_prob, boost_prob], dim=1)
        
        return actions, value, special_probs
    
    def get_actions(self, 
                observations: torch.Tensor, 
                deterministic: bool = False) -> torch.Tensor:
        """
        Get actions from observations, with optional exploration
        """
        actions, _, _ = self.forward(observations)
        
        if not deterministic:
            # Add exploration noise to angle
            angle_noise = torch.randn_like(actions[:, 0:1]) * 0.1
            actions[:, 0:1] = torch.clamp(actions[:, 0:1] + angle_noise, -1.0, 1.0)
            
            # Add small exploration noise to thrust
            thrust_noise = torch.randn_like(actions[:, 1:2]) * 0.05
            actions[:, 1:2] = torch.clamp(actions[:, 1:2] + thrust_noise, 0.0, 1.0)
            
            # Add small exploration noise to special action probabilities
            shield_noise = torch.randn_like(actions[:, 2:3]) * 0.05
            actions[:, 2:3] = torch.clamp(actions[:, 2:3] + shield_noise, 0.0, 1.0)
            
            boost_noise = torch.randn_like(actions[:, 3:4]) * 0.05
            actions[:, 3:4] = torch.clamp(actions[:, 3:4] + boost_noise, 0.0, 1.0)
        
        return actions
