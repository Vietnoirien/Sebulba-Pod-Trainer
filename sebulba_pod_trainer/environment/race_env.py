import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from ..models.pod import Pod

# Game constants
WIDTH = 16000
HEIGHT = 9000
MAX_TURNS_WITHOUT_CHECKPOINT = 100


class RaceEnvironment:
    """
    Racing environment that simulates the Mad Pod Racing game using PyTorch tensors.
    Supports batch processing for efficient training on GPU.
    """
    def __init__(self, 
                 num_checkpoints: int = 3,
                 laps: int = 3,
                 batch_size: int = 64,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        self.batch_size = batch_size
        self.num_checkpoints = num_checkpoints
        self.laps = laps
        self.total_checkpoints = num_checkpoints * laps
        
        # Initialize environment state
        self.checkpoints = None
        self.pods = []
        self.turn_count = None
        self.last_checkpoint_turn = None
        self.done = None
        
        # Generate random tracks
        self._generate_tracks()
        
        # Create pods (2 per player, 2 players)
        for _ in range(4):
            self.pods.append(Pod(device=device, batch_size=batch_size))
        
        # Reset environment
        self.reset()
    
    def _generate_tracks(self) -> None:
        """Generate random tracks for each batch - optimized version"""
        # Create random checkpoints for each batch at once
        self.checkpoints = torch.zeros(self.batch_size, self.num_checkpoints, 2, device=self.device)
        
        # Generate first checkpoint for all batches at once
        self.checkpoints[:, 0, 0] = torch.randint(2000, WIDTH-2000, (self.batch_size,), device=self.device).float()
        self.checkpoints[:, 0, 1] = torch.randint(2000, HEIGHT-2000, (self.batch_size,), device=self.device).float()
        
        # Generate remaining checkpoints with vectorized distance constraints
        for i in range(1, self.num_checkpoints):
            # Try up to 10 times to find valid positions (avoid infinite loops)
            for attempt in range(10):
                # Generate candidate positions for all batches at once
                candidates_x = torch.randint(2000, WIDTH-2000, (self.batch_size,), device=self.device).float()
                candidates_y = torch.randint(2000, HEIGHT-2000, (self.batch_size,), device=self.device).float()
                candidates = torch.stack([candidates_x, candidates_y], dim=1)
                
                # Calculate distances from all previous checkpoints in one operation
                valid_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
                
                for j in range(i):
                    # Calculate distances to previous checkpoint j for all batches
                    distances = torch.norm(
                        self.checkpoints[:, j] - candidates,
                        dim=1
                    )
                    # Update valid mask - positions must be far enough from all previous checkpoints
                    valid_mask &= (distances > 3000)
                
                # Update valid positions
                self.checkpoints[valid_mask, i] = candidates[valid_mask]
                
                # If all batches have valid positions, we're done
                if valid_mask.all():
                    break
                
            # For any remaining invalid positions, just place them randomly
            if not valid_mask.all():
                invalid_mask = ~valid_mask
                self.checkpoints[invalid_mask, i, 0] = torch.randint(2000, WIDTH-2000, (invalid_mask.sum(),), device=self.device).float()
                self.checkpoints[invalid_mask, i, 1] = torch.randint(2000, HEIGHT-2000, (invalid_mask.sum(),), device=self.device).float()
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the environment to initial state"""
        self.turn_count = torch.zeros(self.batch_size, 1, dtype=torch.long, device=self.device)
        self.last_checkpoint_turn = torch.zeros(self.batch_size, 4, dtype=torch.long, device=self.device)
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Reset pods
        for pod_idx, pod in enumerate(self.pods):
            # Reset pod state
            pod.position = torch.zeros(self.batch_size, 2, device=self.device)
            pod.velocity = torch.zeros(self.batch_size, 2, device=self.device)
            pod.angle = torch.zeros(self.batch_size, 1, device=self.device)
            pod.current_checkpoint = torch.zeros(self.batch_size, 1, dtype=torch.long, device=self.device)
            pod.shield_cooldown = torch.zeros(self.batch_size, 1, device=self.device)
            pod.boost_available = torch.ones(self.batch_size, 1, dtype=torch.bool, device=self.device)
            pod.mass = torch.ones(self.batch_size, 1, device=self.device)
            
            # Set initial positions
            player_idx = pod_idx // 2  # 0 for player 1, 1 for player 2
            pod_in_team_idx = pod_idx % 2  # 0 for first pod, 1 for second pod
            
            # Position pods near the first checkpoint with offset
            for b in range(self.batch_size):
                start_cp = self.checkpoints[b, 0]
                next_cp = self.checkpoints[b, 1]
                
                # Calculate direction vector from start to next checkpoint
                direction = next_cp - start_cp
                direction_norm = direction / torch.norm(direction)
                
                # Calculate perpendicular vector
                perpendicular = torch.tensor([-direction_norm[1], direction_norm[0]], device=self.device)
                
                # Position offset based on player and pod index
                offset = 500 * (1 if player_idx == 0 else -1) * (1 if pod_in_team_idx == 0 else 2)
                
                # Set pod position
                pod.position[b] = start_cp + perpendicular * offset
                
                # Set pod angle towards next checkpoint
                pod.angle[b] = pod.angle_to(next_cp)[b]
        
        return self._get_observations()
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for all pods"""
        observations = {}
        
        for pod_idx, pod in enumerate(self.pods):
            # Get pod-specific observations
            player_idx = pod_idx // 2
            team_pod_idx = pod_idx % 2
            obs_key = f"player{player_idx}_pod{team_pod_idx}"
            
            # Get next checkpoint position
            next_cp_idx = pod.current_checkpoint % self.num_checkpoints
            next_cp_positions = torch.stack([
                self.checkpoints[b, next_cp_idx[b].item()] 
                for b in range(self.batch_size)
            ])
            
            # Get next-next checkpoint position
            next_next_cp_idx = (pod.current_checkpoint + 1) % self.num_checkpoints
            next_next_cp_positions = torch.stack([
                self.checkpoints[b, next_next_cp_idx[b].item()] 
                for b in range(self.batch_size)
            ])
            
            # Normalize positions
            normalized_position = pod.position.clone()
            normalized_position[:, 0] = normalized_position[:, 0] / WIDTH * 2 - 1  # Scale to [-1, 1]
            normalized_position[:, 1] = normalized_position[:, 1] / HEIGHT * 2 - 1  # Scale to [-1, 1]
            
            # Normalize velocities
            normalized_velocity = pod.velocity / 1000.0  # Typical max speed is around 600-800
            
            # Normalize checkpoint positions
            normalized_next_cp = next_cp_positions.clone()
            normalized_next_cp[:, 0] = normalized_next_cp[:, 0] / WIDTH * 2 - 1
            normalized_next_cp[:, 1] = normalized_next_cp[:, 1] / HEIGHT * 2 - 1
            
            normalized_next_next_cp = next_next_cp_positions.clone()
            normalized_next_next_cp[:, 0] = normalized_next_next_cp[:, 0] / WIDTH * 2 - 1
            normalized_next_next_cp[:, 1] = normalized_next_next_cp[:, 1] / HEIGHT * 2 - 1
            
            # Calculate relative positions to next checkpoint
            rel_next_cp = normalized_next_cp - normalized_position
            rel_next_next_cp = normalized_next_next_cp - normalized_position
            
            # Calculate angle to next checkpoint (normalized to [-1, 1])
            angle_to_next_cp = pod.angle_to(next_cp_positions) - pod.angle
            angle_to_next_cp = (angle_to_next_cp + 180) % 360 - 180  # Normalize to [-180, 180]
            normalized_angle = angle_to_next_cp / 180.0  # Scale to [-1, 1]
            
            # Combine all observations
            observations[obs_key] = torch.cat([
                normalized_position,
                normalized_velocity,
                rel_next_cp,
                rel_next_next_cp,
                normalized_angle,
                pod.current_checkpoint.float() / self.total_checkpoints,  # Progress
                pod.shield_cooldown / 3.0,  # Normalized shield cooldown
                pod.boost_available.float()  # Boost availability
            ], dim=1)
            
            # Add opponent information
            for opp_idx in range(4):
                if opp_idx != pod_idx:  # Skip self
                    opp_pod = self.pods[opp_idx]
                    
                    # Relative position and velocity of opponent
                    rel_pos = (opp_pod.position - pod.position)
                    normalized_rel_pos = rel_pos.clone()
                    normalized_rel_pos[:, 0] = normalized_rel_pos[:, 0] / WIDTH * 2
                    normalized_rel_pos[:, 1] = normalized_rel_pos[:, 1] / HEIGHT * 2
                    
                    rel_vel = opp_pod.velocity - pod.velocity
                    normalized_rel_vel = rel_vel / 1000.0
                    
                    # Distance to opponent (normalized)
                    distance = torch.norm(rel_pos, dim=1, keepdim=True) / (WIDTH/2)
                    
                    observations[obs_key] = torch.cat([
                        observations[obs_key],
                        normalized_rel_pos,
                        normalized_rel_vel,
                        distance
                    ], dim=1)
        
        return observations
    
    def step(self, actions: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            actions: Dictionary mapping pod keys to action tensors
                    Each action is [angle_target, thrust] where:
                    - angle_target is in [-1, 1] representing target angle adjustment
                    - thrust is in [0, 1] representing thrust power
                    - Special values: thrust < -0.9 for SHIELD, thrust > 0.9 for BOOST
        
        Returns:
            observations: New observations after step
            rewards: Rewards for each batch
            dones: Boolean tensor indicating if episodes are done
            info: Additional information
        """
        # Increment turn counter
        self.turn_count += 1
        
        # Process actions for each pod
        for pod_idx, pod in enumerate(self.pods):
            player_idx = pod_idx // 2
            team_pod_idx = pod_idx % 2
            pod_key = f"player{player_idx}_pod{team_pod_idx}"
            
            if pod_key in actions:
                action = actions[pod_key]
                
                # Extract angle and thrust
                angle_target = action[:, 0].unsqueeze(1)  # [-1, 1]
                thrust_value = action[:, 1].unsqueeze(1)  # [0, 1] or special values
                
                # Get next checkpoint position
                next_cp_idx = pod.current_checkpoint % self.num_checkpoints
                next_cp_positions = torch.stack([
                    self.checkpoints[b, next_cp_idx[b].item()] 
                    for b in range(self.batch_size)
                ])
                
                # Calculate target angle
                base_angle = pod.angle_to(next_cp_positions)
                # Convert normalized angle_target [-1, 1] to degree adjustment [-18, 18]
                angle_adjustment = angle_target * 18.0
                target_angle = base_angle + angle_adjustment
                
                # Rotate pod
                pod.rotate(target_angle)
                
                # Apply thrust/shield/boost based on thrust_value
                shield_mask = thrust_value < -0.9
                boost_mask = thrust_value > 0.9
                normal_thrust_mask = ~(shield_mask | boost_mask)
                
                # Apply shield
                if shield_mask.any():
                    pod.apply_shield()
                
                # Apply boost
                if boost_mask.any():
                    pod.apply_boost()
                
                # Apply normal thrust
                if normal_thrust_mask.any():
                    # Convert normalized thrust [0, 1] to game thrust [0, 100]
                    thrust = torch.round(thrust_value[normal_thrust_mask] * 100)
                    # Create a full tensor of zeros and fill in the normal thrust values
                    full_thrust = torch.zeros_like(thrust_value)
                    full_thrust[normal_thrust_mask] = thrust
                    pod.apply_thrust(full_thrust)
        
        # Move pods and handle collisions
        self._simulate_movement()
        
        # Check for checkpoints
        self._check_checkpoints()
        
        # Check for race completion or timeout
        self._update_race_status()
        
        # Get observations, rewards, and done status
        observations = self._get_observations()
        rewards = self._calculate_rewards()
        
        # Additional info
        info = {
            "turn_count": self.turn_count,
            "checkpoint_progress": torch.stack([pod.current_checkpoint for pod in self.pods]),
        }
        
        return observations, rewards, self.done, info
    
    def _simulate_movement(self) -> None:
        """Simulate pod movement with vectorized collision detection"""
        # First, move all pods without collisions
        for pod in self.pods:
            pod.move()
        
        # Vectorized collision detection between all pod pairs
        for i in range(len(self.pods)):
            for j in range(i+1, len(self.pods)):
                pod_i = self.pods[i]
                pod_j = self.pods[j]
                
                # Calculate distance between pods
                distance = torch.norm(pod_i.position - pod_j.position, dim=1)
                collision_mask = distance < 800  # Pod radius is 400, so collision at distance < 800
                
                if collision_mask.any():
                    # Vectorized collision resolution
                    # Get indices where collisions occur
                    collision_indices = torch.where(collision_mask)[0]
                    
                    # Extract position and velocity for colliding pods
                    pos_i = pod_i.position[collision_indices]
                    vel_i = pod_i.velocity[collision_indices]
                    mass_i = pod_i.mass[collision_indices]
                    
                    pos_j = pod_j.position[collision_indices]
                    vel_j = pod_j.velocity[collision_indices]
                    mass_j = pod_j.mass[collision_indices]
                    
                    # Calculate collision response
                    # Direction from pod_i to pod_j
                    collision_dir = pos_j - pos_i
                    collision_dist = torch.norm(collision_dir, dim=1, keepdim=True)
                    collision_dir = collision_dir / (collision_dist + 1e-8)  # Normalize
                    
                    # Project velocities onto collision direction
                    vel_i_proj = torch.sum(vel_i * collision_dir, dim=1, keepdim=True) * collision_dir
                    vel_j_proj = torch.sum(vel_j * collision_dir, dim=1, keepdim=True) * collision_dir
                    
                    # Calculate new velocities (elastic collision)
                    total_mass = mass_i + mass_j
                    vel_i_new = vel_i - (2 * mass_j / total_mass) * (vel_i_proj - vel_j_proj)
                    vel_j_new = vel_j - (2 * mass_i / total_mass) * (vel_j_proj - vel_i_proj)
                    
                    # Update velocities for colliding pods
                    pod_i.velocity[collision_indices] = vel_i_new
                    pod_j.velocity[collision_indices] = vel_j_new
    
    def _check_checkpoints(self) -> None:
        """Vectorized checkpoint checking"""
        for pod_idx, pod in enumerate(self.pods):
            # Get batch-specific next checkpoint positions
            next_cp_idx = pod.current_checkpoint % self.num_checkpoints
            
            # Vectorized gathering of checkpoint positions
            batch_indices = torch.arange(self.batch_size, device=self.device)
            next_cp_positions = self.checkpoints[batch_indices, next_cp_idx.squeeze()]
            
            # Check distances to next checkpoints
            distances = torch.norm(pod.position - next_cp_positions, dim=1)
            reached = distances <= 600  # Checkpoint radius
            
            # Update checkpoint counters and last checkpoint turn
            if reached.any():
                pod.current_checkpoint[reached] += 1
                # Fix the shape mismatch by squeezing the turn_count tensor
                self.last_checkpoint_turn[reached, pod_idx] = self.turn_count[reached].squeeze()

    def _update_race_status(self) -> None:
        """Update race status, checking for completion or timeout"""
        # Check if any pod has completed all checkpoints
        for player_idx in range(2):  # Two players
            player_pods = [self.pods[player_idx*2], self.pods[player_idx*2+1]]
            
            for pod in player_pods:
                # Check if pod has completed all checkpoints
                race_completed = pod.current_checkpoint >= self.total_checkpoints
                
                # Mark races as done where this player has won
                self.done = self.done | race_completed.squeeze()
        
        # Check for timeout (100 turns without reaching a checkpoint)
        for pod_idx, pod in enumerate(self.pods):
            turns_since_checkpoint = self.turn_count - self.last_checkpoint_turn[:, pod_idx].unsqueeze(1)
            timeout = turns_since_checkpoint > MAX_TURNS_WITHOUT_CHECKPOINT
            
            # Mark races as done where timeout occurred
            self.done = self.done | timeout.squeeze()
    
    def _calculate_rewards(self) -> Dict[str, torch.Tensor]:
        """Calculate rewards for each pod"""
        rewards = {}
        
        for pod_idx, pod in enumerate(self.pods):
            player_idx = pod_idx // 2
            team_pod_idx = pod_idx % 2
            pod_key = f"player{player_idx}_pod{team_pod_idx}"
            
            # Base reward is progress through checkpoints
            checkpoint_reward = pod.current_checkpoint.float() / self.total_checkpoints
            
            # Get next checkpoint position
            next_cp_idx = pod.current_checkpoint % self.num_checkpoints
            next_cp_positions = torch.stack([
                self.checkpoints[b, next_cp_idx[b].item()] 
                for b in range(self.batch_size)
            ])
            
            # Distance-based reward (closer to next checkpoint is better)
            distances = pod.distance(next_cp_positions)
            normalized_distances = torch.clamp(1.0 - distances / 5000.0, 0.0, 1.0)
            
            # Velocity-based reward (higher speed towards checkpoint is better)
            direction_to_checkpoint = next_cp_positions - pod.position
            direction_norm = torch.norm(direction_to_checkpoint, dim=1, keepdim=True)
            direction_norm = torch.where(direction_norm > 0, direction_norm, torch.ones_like(direction_norm))
            normalized_direction = direction_to_checkpoint / direction_norm
            
            velocity_alignment = torch.sum(normalized_direction * pod.velocity, dim=1, keepdim=True)
            velocity_reward = torch.clamp(velocity_alignment / 500.0, -1.0, 1.0)
            
            # Combine rewards
            pod_reward = checkpoint_reward + 0.01 * normalized_distances + 0.005 * velocity_reward
            
            # Win/loss reward
            race_completed = pod.current_checkpoint >= self.total_checkpoints
            pod_reward = torch.where(race_completed, pod_reward + 10.0, pod_reward)
            
            # Opponent pods (from other player)
            opponent_player_idx = 1 - player_idx
            opponent_pods = [
                self.pods[opponent_player_idx*2], 
                self.pods[opponent_player_idx*2+1]
            ]
            
            # Penalty if opponent completes race
            for opp_pod in opponent_pods:
                opp_completed = opp_pod.current_checkpoint >= self.total_checkpoints
                pod_reward = torch.where(opp_completed, pod_reward - 5.0, pod_reward)
            
            rewards[pod_key] = pod_reward
        
        return rewards

    def get_checkpoints(self):
        """Return checkpoint positions for visualization"""
        if self.checkpoints is not None:
            # Return first batch's checkpoints
            return self.checkpoints[0].cpu().numpy().tolist()
        return []