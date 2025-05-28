import torch
import numpy as np
import math
from typing import Dict, Tuple, Any
import sys
import io
from contextlib import redirect_stderr
import importlib.util
import os
import re

class HeuristicPodAdapter:
    """
    Adapter to use the heuristic pod with the OptimizedRaceEnvironment.
    Converts between environment observations/actions and heuristic pod format.
    """
    
    def __init__(self, heuristic_path: str = None):
        """Initialize the adapter and load the heuristic controller"""
        self.heuristic_control = None
        self.pod_states = {}  # Track state for each pod
        
        # Load the heuristic controller
        if heuristic_path is None:
            # Default path - adjust as needed
            heuristic_path = "pods/heuristic_pod.py"
        
        self._load_heuristic_controller(heuristic_path)
        
        # Initialize global state tracking for each pod
        self._init_pod_states()
    
    def _load_heuristic_controller(self, heuristic_path: str):
        """Load the heuristic controller from file, avoiding the main loop"""
        try:
            # Read the file and extract only the function definitions and imports
            with open(heuristic_path, 'r') as f:
                content = f.read()
            
            # Find where the main game loop starts and remove it
            # Look for the "# game loop" comment or "while True:"
            lines = content.split('\n')
            filtered_lines = []
            in_main_loop = False
            
            for line in lines:
                # Check if we've reached the main game loop
                if ('# game loop' in line.lower() or 
                    (line.strip().startswith('while True:') and 'game' in ''.join(lines).lower())):
                    in_main_loop = True
                    break
                
                # Skip the main loop execution
                if not in_main_loop:
                    filtered_lines.append(line)
            
            # Create the modified content without the main loop
            modified_content = '\n'.join(filtered_lines)
            
            # Create a temporary module and execute the modified content
            spec = importlib.util.spec_from_file_location("heuristic_pod", heuristic_path)
            heuristic_module = importlib.util.module_from_spec(spec)
            
            # Execute the modified content in the module's namespace
            exec(modified_content, heuristic_module.__dict__)
            
            # Get the control function
            if hasattr(heuristic_module, 'control'):
                self.heuristic_control = heuristic_module.control
                print(f"Successfully loaded heuristic controller from {heuristic_path}")
            else:
                raise AttributeError("No 'control' function found in heuristic module")
                
        except Exception as e:
            print(f"Failed to load heuristic controller: {e}")
            # Fallback to a simple controller
            self.heuristic_control = self._simple_fallback_controller
    
    def _init_pod_states(self):
        """Initialize state tracking for all pods"""
        for player_idx in range(2):
            for pod_idx in range(2):
                pod_key = f"player{player_idx}_pod{pod_idx}"
                self.pod_states[pod_key] = {
                    'last_x': None,
                    'last_y': None,
                    'velocity_x': 0,
                    'velocity_y': 0,
                    'boost_used': False,
                    'shield_used': False,
                    'shield_cooldown': 0,
                    'checkpoint_history': [],
                    'checkpoint_count': 0,
                    'lap_completed': False,
                    'current_checkpoint_index': 0,
                    'last_checkpoint_x': None,
                    'last_checkpoint_y': None
                }
    
    def _extract_pod_state_from_obs(self, obs: torch.Tensor, batch_idx: int = 0) -> Dict[str, Any]:
        """Extract pod state information from environment observation"""
        obs_np = obs[batch_idx].cpu().numpy()
        
        # The observation format from the environment (56 dimensions total):
        # Base observations (48 dims): position, velocity, checkpoints, angles, etc.
        # Role observations (8 dims): role-specific information
        
        # Extract position (first 2 elements are normalized position)
        normalized_pos = obs_np[0:2]  # [-1, 1] range
        # Convert back to game coordinates
        x = (normalized_pos[0] + 1) * 16000 / 2  # WIDTH = 16000
        y = (normalized_pos[1] + 1) * 9000 / 2   # HEIGHT = 9000
        
        # Extract velocity (elements 2-3, normalized)
        normalized_vel = obs_np[2:4]
        velocity_x = normalized_vel[0] * 1000  # Denormalize velocity
        velocity_y = normalized_vel[1] * 1000
        
        # Extract next checkpoint relative position (elements 4-5)
        rel_next_cp = obs_np[4:6]
        # Convert back to absolute position
        next_checkpoint_x = x + rel_next_cp[0] * 16000 / 2
        next_checkpoint_y = y + rel_next_cp[1] * 9000 / 2
        
        # Extract angle information (elements 6-9 are angle sin/cos pairs)
        pod_angle_sin, pod_angle_cos = obs_np[6], obs_np[7]
        pod_angle = math.degrees(math.atan2(pod_angle_sin, pod_angle_cos))
        
        angle_to_cp_sin, angle_to_cp_cos = obs_np[8], obs_np[9]
        angle_to_cp = math.degrees(math.atan2(angle_to_cp_sin, angle_to_cp_cos))
        
        # Calculate angle difference
        next_checkpoint_angle = angle_to_cp - pod_angle
        next_checkpoint_angle = ((next_checkpoint_angle + 180) % 360) - 180
        
        # Extract distance to checkpoint (element 12, normalized)
        distance_normalized = obs_np[12]
        next_checkpoint_dist = distance_normalized * 600  # Denormalize
        
        # Extract shield cooldown (element 14)
        shield_cooldown = int(obs_np[14] * 3)  # Denormalize from [0,1] to [0,3]
        
        # Extract boost availability (element 15)
        boost_available = obs_np[15] > 0.5
        
        # Extract opponent information (starting from element 18)
        # For simplicity, we'll use the first opponent's position
        opponent_rel_pos = obs_np[18:20]  # First opponent relative position
        opponent_x = x + opponent_rel_pos[0] * 16000 / 2
        opponent_y = y + opponent_rel_pos[1] * 9000 / 2
        
        return {
            'x': x,
            'y': y,
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'next_checkpoint_x': next_checkpoint_x,
            'next_checkpoint_y': next_checkpoint_y,
            'next_checkpoint_dist': next_checkpoint_dist,
            'next_checkpoint_angle': next_checkpoint_angle,
            'opponent_x': opponent_x,
            'opponent_y': opponent_y,
            'shield_cooldown': shield_cooldown,
            'boost_available': boost_available,
            'pod_angle': pod_angle
        }
    
    def _simple_fallback_controller(self, next_checkpoint_x, next_checkpoint_y, 
                                  next_checkpoint_dist, next_checkpoint_angle, 
                                  x, y, opponent_x, opponent_y):
        """Simple fallback controller if heuristic loading fails"""
        # Simple logic: aim for checkpoint with speed based on angle
        target_x = int(next_checkpoint_x)
        target_y = int(next_checkpoint_y)
        
        if abs(next_checkpoint_angle) < 30:
            thrust = 100
        elif abs(next_checkpoint_angle) < 90:
            thrust = 50
        else:
            thrust = 20
            
        return f"{target_x} {target_y} {thrust}"
    
    def _parse_heuristic_output(self, output: str, state: Dict[str, Any]) -> torch.Tensor:
        """Parse heuristic output and convert to environment action format"""
        parts = output.strip().split()
        
        if len(parts) != 3:
            # Fallback action
            return torch.tensor([[0.0, 0.5, 0.0, 0.0]], dtype=torch.float32)
        
        target_x = float(parts[0])
        target_y = float(parts[1])
        thrust_str = parts[2]
        
        # Calculate angle adjustment
        current_x = state['x']
        current_y = state['y']
        
        # Calculate desired angle
        dx = target_x - current_x
        dy = target_y - current_y
        desired_angle = math.degrees(math.atan2(dy, dx))
        
        # Calculate angle difference from current pod angle
        current_angle = state['pod_angle']
        angle_diff = desired_angle - current_angle
        angle_diff = ((angle_diff + 180) % 360) - 180
        
        # Normalize to [-1, 1] range (assuming max adjustment of 18 degrees)
        angle_adjustment = np.clip(angle_diff / 18.0, -1.0, 1.0)
        
        # Parse thrust
        shield_prob = 0.0
        boost_prob = 0.0
        thrust_value = 0.0
        
        if thrust_str == "SHIELD":
            shield_prob = 1.0
            thrust_value = 0.0
        elif thrust_str == "BOOST":
            boost_prob = 1.0
            thrust_value = 1.0
        else:
            try:
                thrust_int = int(thrust_str)
                thrust_value = np.clip(thrust_int / 100.0, 0.0, 1.0)
            except ValueError:
                thrust_value = 0.5  # Default
        
        return torch.tensor([[angle_adjustment, thrust_value, shield_prob, boost_prob]], 
                          dtype=torch.float32)
    
    def get_actions(self, observations: Dict[str, torch.Tensor], 
                   env_state: Any = None, batch_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Convert environment observations to heuristic pod actions
        
        Args:
            observations: Dict of observations from environment
            env_state: Additional environment state (optional)
            batch_idx: Which batch item to use for single-pod testing
            
        Returns:
            Dict of actions in environment format
        """
        actions = {}
        
        # Redirect stderr to capture heuristic debug output
        stderr_buffer = io.StringIO()
        
        for pod_key, obs in observations.items():
            try:
                with redirect_stderr(stderr_buffer):
                    # Extract game state from observation
                    game_state = self._extract_pod_state_from_obs(obs, batch_idx)
                    
                    # Update pod state tracking
                    pod_state = self.pod_states[pod_key]
                    if pod_state['last_x'] is not None:
                        pod_state['velocity_x'] = game_state['x'] - pod_state['last_x']
                        pod_state['velocity_y'] = game_state['y'] - pod_state['last_y']
                    
                    pod_state['last_x'] = game_state['x']
                    pod_state['last_y'] = game_state['y']
                    
                    # Update shield cooldown
                    if pod_state['shield_cooldown'] > 0:
                        pod_state['shield_cooldown'] -= 1
                    
                    # Call heuristic controller
                    heuristic_output = self.heuristic_control(
                        next_checkpoint_x=game_state['next_checkpoint_x'],
                        next_checkpoint_y=game_state['next_checkpoint_y'],
                        next_checkpoint_dist=game_state['next_checkpoint_dist'],
                        next_checkpoint_angle=game_state['next_checkpoint_angle'],
                        x=game_state['x'],
                        y=game_state['y'],
                        opponent_x=game_state['opponent_x'],
                        opponent_y=game_state['opponent_y']
                    )
                    
                    # Convert to environment action format
                    actions[pod_key] = self._parse_heuristic_output(heuristic_output, game_state)
                    
            except Exception as e:
                print(f"Error processing {pod_key}: {e}")
                # Fallback action
                actions[pod_key] = torch.tensor([[0.0, 0.5, 0.0, 0.0]], dtype=torch.float32)
        
        return actions

    def reset(self):
        """Reset the adapter state"""
        self._init_pod_states()