import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from ..models.pod import Pod

# Game constants
WIDTH = 16000
HEIGHT = 9000
MAX_TURNS_WITHOUT_CHECKPOINT = 100


class OptimizedRaceEnvironment:
    """
    Optimized racing environment that simulates the Mad Pod Racing game using PyTorch tensors.
    Supports batch processing for efficient training on GPU with performance optimizations.
    """
    def __init__(self, 
                num_checkpoints: int = 3,
                laps: int = 3,
                batch_size: int = 64,
                device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                randomize_checkpoints: bool = True,
                min_checkpoints: int = 3,
                max_checkpoints: int = 6):
        self.device = device
        self.batch_size = batch_size
        self.base_num_checkpoints = num_checkpoints
        self.laps = laps

        # Add randomization parameters
        self.randomize_checkpoints = randomize_checkpoints
        self.min_checkpoints = min_checkpoints
        self.max_checkpoints = max_checkpoints

        # num_checkpoints and total_checkpoints will be set per batch in reset()
        self.num_checkpoints = num_checkpoints
        self.total_checkpoints = num_checkpoints * laps
        
        # Initialize environment state
        self.checkpoints = None
        self.pods = []
        
        # Initialize tensors that will be used in reset
        self.turn_count = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.last_checkpoint_turn = torch.zeros(batch_size, 4, dtype=torch.long, device=device)
        self.done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Pre-allocate tensors for efficiency
        self._setup_cached_tensors()
        
        # Generate random tracks
        self._generate_tracks()
        
        # Create pods (2 per player, 2 players)
        for _ in range(4):
            self.pods.append(Pod(device=device, batch_size=batch_size))
        
        # Reset environment
        self.reset()
    
    def _setup_cached_tensors(self):
        """Pre-allocate tensors that will be reused to avoid repeated allocations"""
        # Cached tensors for observations
        self.normalized_position_cache = torch.zeros(self.batch_size, 2, device=self.device)
        self.normalized_velocity_cache = torch.zeros(self.batch_size, 2, device=self.device)
        self.normalized_next_cp_cache = torch.zeros(self.batch_size, 2, device=self.device)
        self.normalized_next_next_cp_cache = torch.zeros(self.batch_size, 2, device=self.device)
        self.rel_next_cp_cache = torch.zeros(self.batch_size, 2, device=self.device)
        self.rel_next_next_cp_cache = torch.zeros(self.batch_size, 2, device=self.device)
        
        # Cached tensors for collision detection
        self.collision_indices_cache = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Cached tensors for reward calculation
        self.checkpoint_reward_cache = torch.zeros(self.batch_size, 1, device=self.device)
        self.distance_reward_cache = torch.zeros(self.batch_size, 1, device=self.device)
        self.velocity_reward_cache = torch.zeros(self.batch_size, 1, device=self.device)
        
        # Batch indices for efficient indexing
        self.batch_indices = torch.arange(self.batch_size, device=self.device)
    
    def _generate_tracks(self) -> None:
        """Generate random tracks for each batch - with randomized checkpoint counts"""
        # Randomize number of checkpoints per batch if enabled
        if self.randomize_checkpoints:
            # Generate random checkpoint counts for each batch
            checkpoint_counts = torch.randint(
                self.min_checkpoints, 
                self.max_checkpoints + 1, 
                (self.batch_size,), 
                device=self.device
            )
            max_checkpoints_in_batch = checkpoint_counts.max().item()
            self.batch_checkpoint_counts = checkpoint_counts
        else:
            max_checkpoints_in_batch = self.base_num_checkpoints
            self.batch_checkpoint_counts = torch.full(
                (self.batch_size,), 
                self.base_num_checkpoints, 
                device=self.device
            )
        
        # Create checkpoints tensor with maximum size needed
        self.checkpoints = torch.zeros(self.batch_size, max_checkpoints_in_batch, 2, device=self.device)
        
        # Generate checkpoints for each batch individually
        for batch_idx in range(self.batch_size):
            num_cp = self.batch_checkpoint_counts[batch_idx].item()
            
            # Generate first checkpoint
            self.checkpoints[batch_idx, 0, 0] = torch.randint(2000, WIDTH-2000, (1,), device=self.device).float()
            self.checkpoints[batch_idx, 0, 1] = torch.randint(2000, HEIGHT-2000, (1,), device=self.device).float()
            
            # Generate remaining checkpoints with distance constraints
            for i in range(1, num_cp):
                for attempt in range(5):
                    candidate_x = torch.randint(2000, WIDTH-2000, (1,), device=self.device).float()
                    candidate_y = torch.randint(2000, HEIGHT-2000, (1,), device=self.device).float()
                    candidate = torch.tensor([candidate_x, candidate_y], device=self.device)
                    
                    # Check distance from all previous checkpoints
                    valid = True
                    for j in range(i):
                        distance = torch.norm(self.checkpoints[batch_idx, j] - candidate)
                        if distance <= 3000:
                            valid = False
                            break
                    
                    if valid:
                        self.checkpoints[batch_idx, i] = candidate
                        break
                else:
                    # Fallback: place randomly if no valid position found
                    self.checkpoints[batch_idx, i, 0] = torch.randint(2000, WIDTH-2000, (1,), device=self.device).float()
                    self.checkpoints[batch_idx, i, 1] = torch.randint(2000, HEIGHT-2000, (1,), device=self.device).float()
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the environment to initial state"""
        # Reset counters
        self.turn_count.zero_()
        self.last_checkpoint_turn.zero_()
        self.done.zero_()
        
        # Reset pods
        for pod_idx, pod in enumerate(self.pods):
            # Reset pod state
            pod.position.zero_()
            pod.velocity.zero_()
            pod.angle.zero_()
            pod.current_checkpoint.zero_()
            pod.shield_cooldown.zero_()
            pod.boost_available.fill_(1)
            pod.mass.fill_(1)
            
            # Set initial positions
            player_idx = pod_idx // 2  # 0 for player 1, 1 for player 2
            pod_in_team_idx = pod_idx % 2  # 0 for first pod, 1 for second pod
            
            # Position pods near the first checkpoint with offset
            # Vectorized implementation for all batches at once
            start_cp = self.checkpoints[:, 0]
            next_cp = self.checkpoints[:, 1]
            
            # Calculate direction vector from start to next checkpoint
            direction = next_cp - start_cp
            direction_norm = direction / torch.norm(direction, dim=1, keepdim=True)
            
            # Calculate perpendicular vector (vectorized)
            perpendicular = torch.stack([-direction_norm[:, 1], direction_norm[:, 0]], dim=1)
            
            # Position offset based on player and pod index
            offset = 500 * (1 if player_idx == 0 else -1) * (1 if pod_in_team_idx == 0 else 2)
            
            # Set pod position (vectorized)
            pod.position = start_cp + perpendicular * offset
            
            # Set pod angle towards next checkpoint (vectorized)
            pod.angle = pod.angle_to(next_cp)
        
        return self._get_observations()
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for all pods - updated with absolute angles"""
        observations = {}
        
        # Pre-compute next checkpoint indices for all pods (with modulo based on batch-specific counts)
        next_cp_indices = []
        next_next_cp_indices = []
        
        for pod in self.pods:
            next_cp_idx = torch.zeros_like(pod.current_checkpoint)
            next_next_cp_idx = torch.zeros_like(pod.current_checkpoint)
            
            for b in range(self.batch_size):
                num_cp = self.batch_checkpoint_counts[b].item()
                next_cp_idx[b] = pod.current_checkpoint[b] % num_cp
                next_next_cp_idx[b] = (pod.current_checkpoint[b] + 1) % num_cp
            
            next_cp_indices.append(next_cp_idx)
            next_next_cp_indices.append(next_next_cp_idx)
        
        # Pre-compute next checkpoint positions for all pods
        next_cp_positions = []
        next_next_cp_positions = []
        
        for pod_idx in range(4):
            # Get next checkpoint positions (vectorized)
            next_cp_idx = next_cp_indices[pod_idx]
            next_cp_pos = torch.zeros(self.batch_size, 2, device=self.device)
            for b in range(self.batch_size):
                next_cp_pos[b] = self.checkpoints[b, next_cp_idx[b].item()]
            next_cp_positions.append(next_cp_pos)
            
            # Get next-next checkpoint positions (vectorized)
            next_next_cp_idx = next_next_cp_indices[pod_idx]
            next_next_cp_pos = torch.zeros(self.batch_size, 2, device=self.device)
            for b in range(self.batch_size):
                next_next_cp_pos[b] = self.checkpoints[b, next_next_cp_idx[b].item()]
            next_next_cp_positions.append(next_next_cp_pos)
        
        for pod_idx, pod in enumerate(self.pods):
            # Get pod-specific observations
            player_idx = pod_idx // 2
            team_pod_idx = pod_idx % 2
            obs_key = f"player{player_idx}_pod{team_pod_idx}"
            
            # Reuse cached tensors for normalization
            # Normalize positions (in-place operations)
            self.normalized_position_cache.copy_(pod.position)
            self.normalized_position_cache[:, 0].mul_(2.0/WIDTH).sub_(1.0)  # Scale to [-1, 1]
            self.normalized_position_cache[:, 1].mul_(2.0/HEIGHT).sub_(1.0)  # Scale to [-1, 1]
            
            # Normalize velocities (in-place operations)
            self.normalized_velocity_cache.copy_(pod.velocity)
            self.normalized_velocity_cache.div_(1000.0)  # Typical max speed is around 600-800
            
            # Normalize checkpoint positions (in-place operations)
            self.normalized_next_cp_cache.copy_(next_cp_positions[pod_idx])
            self.normalized_next_cp_cache[:, 0].mul_(2.0/WIDTH).sub_(1.0)
            self.normalized_next_cp_cache[:, 1].mul_(2.0/HEIGHT).sub_(1.0)
            
            self.normalized_next_next_cp_cache.copy_(next_next_cp_positions[pod_idx])
            self.normalized_next_next_cp_cache[:, 0].mul_(2.0/WIDTH).sub_(1.0)
            self.normalized_next_next_cp_cache[:, 1].mul_(2.0/HEIGHT).sub_(1.0)
            
            # Calculate relative positions to next checkpoint (in-place operations)
            self.rel_next_cp_cache.copy_(self.normalized_next_cp_cache)
            self.rel_next_cp_cache.sub_(self.normalized_position_cache)
            
            self.rel_next_next_cp_cache.copy_(self.normalized_next_next_cp_cache)
            self.rel_next_next_cp_cache.sub_(self.normalized_position_cache)

            # Pod's absolute angle as sin/cos
            pod_angle_rad = torch.deg2rad(pod.angle)
            pod_angle_sin = torch.sin(pod_angle_rad)
            pod_angle_cos = torch.cos(pod_angle_rad)

            # Angle to next checkpoint as sin/cos
            angle_to_next_cp = pod.angle_to(next_cp_positions[pod_idx])
            angle_to_next_cp_rad = torch.deg2rad(angle_to_next_cp)
            angle_to_next_cp_sin = torch.sin(angle_to_next_cp_rad)
            angle_to_next_cp_cos = torch.cos(angle_to_next_cp_rad)

            # Relative angle (how much to turn)
            relative_angle = angle_to_next_cp - pod.angle
            relative_angle = ((relative_angle + 180) % 360) - 180
            relative_angle_normalized = relative_angle / 180.0

            # Speed and direction metrics
            speed_magnitude = torch.norm(pod.velocity, dim=1, keepdim=True) / 800.0
            distance_to_next_cp = torch.norm(next_cp_positions[pod_idx] - pod.position, dim=1, keepdim=True)
            distance_normalized = distance_to_next_cp / 600.0  # Normalize by checkpoint radius
            
            batch_total_checkpoints = self.batch_checkpoint_counts * self.laps
            progress = pod.current_checkpoint.float() / batch_total_checkpoints.float().unsqueeze(1)

            # Pod-specific observations (18 dimensions)
            pod_specific_obs = torch.cat([
                self.normalized_position_cache,             # Pod position (2)
                self.normalized_velocity_cache,             # Pod velocity (2)  
                self.rel_next_cp_cache,                     # Relative position to next checkpoint (2)
                self.rel_next_next_cp_cache,                # Relative position to next-next checkpoint (2)
                pod_angle_sin, pod_angle_cos,               # Pod angle (sin/cos) (2)
                angle_to_next_cp_sin, angle_to_next_cp_cos, # Angle to target (sin/cos) (2)
                relative_angle_normalized,                  # How much to turn (1)
                speed_magnitude,                            # Current speed (1)
                distance_normalized,                        # Distance to next checkpoint (1)
                progress,                                   # Progress (1)
                pod.shield_cooldown / 3.0,                  # Shield cooldown (1)
                pod.boost_available.float()                 # Boost availability (1)
            ], dim=1)

            # Add opponent information
            opponent_obs_list = []
            for opp_idx in range(4):
                if opp_idx != pod_idx:  # Skip self
                    opp_pod = self.pods[opp_idx]
                    
                    # Relative position and velocity of opponent
                    rel_pos = (opp_pod.position - pod.position)
                    normalized_rel_pos = rel_pos.clone()
                    normalized_rel_pos[:, 0].mul_(2.0/WIDTH)
                    normalized_rel_pos[:, 1].mul_(2.0/HEIGHT)
                    
                    rel_vel = opp_pod.velocity - pod.velocity
                    normalized_rel_vel = rel_vel / 1000.0
                    
                    # Distance to opponent (normalized)
                    distance = torch.norm(rel_pos, dim=1, keepdim=True) / (WIDTH/2)
                    
                    # Opponent's absolute angle
                    opp_normalized_angle = (opp_pod.angle % 360) / 180.0 - 1.0
                    
                    # Opponent's progress
                    batch_total_checkpoints = self.batch_checkpoint_counts * self.laps
                    opp_progress = opp_pod.current_checkpoint.float() / batch_total_checkpoints.float().unsqueeze(1)
                    
                    # Opponent's next checkpoint
                    opp_next_cp_normalized = (opp_pod.current_checkpoint % self.num_checkpoints).float() / self.num_checkpoints
                    
                    # Add to list for single concatenation (10 dimensions per opponent)
                    opponent_obs_list.append(torch.cat([
                        normalized_rel_pos,                    # (2)
                        normalized_rel_vel,                    # (2)
                        distance,                             # (1)
                        opp_normalized_angle,                 # (1)
                        opp_progress,                         # (1)
                        opp_next_cp_normalized,               # (1)
                        opp_pod.shield_cooldown / 3.0,       # (1)
                        opp_pod.boost_available.float()       # (1)
                    ], dim=1))
                            
            # Combine pod-specific and opponent observations
            # Pod-specific: 18 dimensions
            # Opponent data: 3 opponents × 10 dimensions = 30 dimensions
            # Total: 18 + 30 = 48 dimensions
            all_opponent_obs = torch.cat(opponent_obs_list, dim=1)
            observations[obs_key] = torch.cat([pod_specific_obs, all_opponent_obs], dim=1)
        
        return observations
    
    def step(self, actions: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Execute one step in the environment - optimized version with device consistency
        """
        # Increment turn counter
        self.turn_count += 1
        
        # Process actions for each pod
        for pod_idx, pod in enumerate(self.pods):
            player_idx = pod_idx // 2
            team_pod_idx = pod_idx % 2
            pod_key = f"player{player_idx}_pod{team_pod_idx}"
            
            if pod_key in actions:
                action = actions[pod_key].to(self.device)  # Ensure action is on environment's device
                
                # Extract angle and thrust
                angle_target = action[:, 0].unsqueeze(1)  # [-1, 1]
                thrust_value = action[:, 1].unsqueeze(1)  # [0, 1] or special values
                
                # Get next checkpoint position
                next_cp_idx = pod.current_checkpoint % self.num_checkpoints
                
                # Vectorized gathering of checkpoint positions
                next_cp_positions = torch.zeros(self.batch_size, 2, device=self.device)
                for b in range(self.batch_size):
                    next_cp_positions[b] = self.checkpoints[b, next_cp_idx[b].item()]
                
                # Calculate target angle
                base_angle = pod.angle_to(next_cp_positions)
                # Convert normalized angle_target [-1, 1] to degree adjustment [-18, 18]
                angle_adjustment = angle_target * 18.0
                target_angle = base_angle + angle_adjustment
                
                # Rotate pod
                pod.rotate(target_angle)
                
                # Apply thrust/shield/boost based on thrust_value
                # Use in-place operations where possible
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
                    # Use in-place operations for efficiency
                    full_thrust = torch.zeros_like(thrust_value)
                    thrust = torch.round(thrust_value[normal_thrust_mask] * 100)
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
        """Simulate pod movement with proper collision detection and bounce mechanics"""
        # First, move all pods (this calls their move() method)
        for pod in self.pods:
            pod.move()
        
        # Then handle collisions between all pod pairs
        POD_RADIUS = 400.0
        MIN_IMPULSE = 120.0
        
        for i in range(len(self.pods)):
            for j in range(i + 1, len(self.pods)):
                pod_i = self.pods[i]
                pod_j = self.pods[j]
                
                # Calculate distance between pods
                diff = pod_j.position - pod_i.position
                distance = torch.norm(diff, dim=1)
                collision_mask = distance <= (2 * POD_RADIUS)
                
                if collision_mask.any():
                    collision_indices = torch.where(collision_mask)[0]
                    if len(collision_indices) > 0:
                        self._handle_collision_batch(
                            pod_i, pod_j, collision_indices, 
                            POD_RADIUS * 2, MIN_IMPULSE
                        )

    def _handle_collision_batch(self, pod_i, pod_j, collision_indices, min_radius, min_impulse):
        """Handle collision between two pods for specific batch indices"""
        # Extract data for colliding instances
        pos_i = pod_i.position[collision_indices]
        vel_i = pod_i.velocity[collision_indices]
        mass_i = pod_i.mass[collision_indices]
        
        pos_j = pod_j.position[collision_indices]
        vel_j = pod_j.velocity[collision_indices]
        mass_j = pod_j.mass[collision_indices]
        
        # Calculate normal vector (from pod_i to pod_j)
        normal = pos_j - pos_i
        distance = torch.norm(normal, dim=1, keepdim=True)
        
        # Avoid division by zero
        distance = torch.where(distance > 0, distance, torch.ones_like(distance))
        normal = normal / distance
        
        # Calculate relative velocity
        relative_velocity = vel_i - vel_j
        
        # Calculate collision force
        # Using equal masses (mass = 1 for both pods as in original game)
        force = torch.sum(normal * relative_velocity, dim=1, keepdim=True) / (1/mass_i + 1/mass_j)
        
        # Apply minimum impulse if force is too small
        force = torch.where(force < min_impulse, 
                        torch.full_like(force, min_impulse), 
                        force)
        
        # Calculate impulse
        impulse = -normal * force
        
        # Apply impulse to velocities
        vel_i_new = vel_i + impulse * (1 / mass_i)
        vel_j_new = vel_j - impulse * (1 / mass_j)
        
        # Separate overlapping pods
        overlap = min_radius - distance.squeeze(-1)
        separation_mask = overlap > 0
        
        if separation_mask.any():
            # Calculate separation distance
            separation = overlap[separation_mask].unsqueeze(-1) / 2 + 1e-6  # Small epsilon
            separation_normal = normal[separation_mask]
            
            # Move pods apart
            pos_i_new = pos_i.clone()
            pos_j_new = pos_j.clone()
            
            pos_i_new[separation_mask] -= separation_normal * separation
            pos_j_new[separation_mask] += separation_normal * separation
            
            # Update positions for colliding instances
            pod_i.position[collision_indices] = pos_i_new
            pod_j.position[collision_indices] = pos_j_new
        
        # Update velocities for colliding instances
        pod_i.velocity[collision_indices] = vel_i_new
        pod_j.velocity[collision_indices] = vel_j_new
    
    def _check_checkpoints(self) -> None:
        """Optimized checkpoint checking - updated for variable checkpoint counts"""
        for pod_idx, pod in enumerate(self.pods):
            # Get batch-specific next checkpoint positions
            next_cp_positions = torch.zeros(self.batch_size, 2, device=self.device)
            
            for b in range(self.batch_size):
                num_cp = self.batch_checkpoint_counts[b].item()
                next_cp_idx = pod.current_checkpoint[b] % num_cp
                next_cp_positions[b] = self.checkpoints[b, next_cp_idx]
            
            # Check distances to next checkpoints
            diff = pod.position - next_cp_positions
            distances = torch.norm(diff, dim=1)
            reached = distances <= 600  # Checkpoint radius
            
            # Update checkpoint counters and last checkpoint turn
            if reached.any():
                pod.current_checkpoint[reached] += 1
                self.last_checkpoint_turn[reached, pod_idx] = self.turn_count[reached].squeeze()

    def _update_race_status(self) -> None:
        """Update race status - updated for variable checkpoint counts"""
        new_done = torch.zeros_like(self.done)
        
        # Calculate total checkpoints per batch
        batch_total_checkpoints = self.batch_checkpoint_counts * self.laps
        
        # Check if any pod has completed all checkpoints
        for player_idx in range(2):
            player_pods = [self.pods[player_idx*2], self.pods[player_idx*2+1]]
            
            for pod in player_pods:
                # Check if pod has completed all checkpoints (batch-specific)
                race_completed = pod.current_checkpoint.squeeze() >= batch_total_checkpoints
                new_done = new_done | race_completed
        
        # Check for timeout (unchanged)
        for pod_idx, pod in enumerate(self.pods):
            turns_since_checkpoint = self.turn_count - self.last_checkpoint_turn[:, pod_idx].unsqueeze(1)
            timeout = turns_since_checkpoint > MAX_TURNS_WITHOUT_CHECKPOINT
            new_done = new_done | timeout.squeeze()
        
        self.done = new_done
    
    def _calculate_rewards(self) -> Dict[str, torch.Tensor]:
        """Calculate rewards for each pod - optimized version"""
        rewards = {}
        
        # Pre-compute next checkpoint positions for all pods
        next_cp_positions = []
        for pod in self.pods:
            next_cp_idx = pod.current_checkpoint % self.num_checkpoints
            next_cp_pos = torch.zeros(self.batch_size, 2, device=self.device)
            for b in range(self.batch_size):
                next_cp_pos[b] = self.checkpoints[b, next_cp_idx[b].item()]
            next_cp_positions.append(next_cp_pos)
        
        # Pre-compute race completion status for all pods
        race_completed = [pod.current_checkpoint >= self.total_checkpoints for pod in self.pods]
        
        for pod_idx, pod in enumerate(self.pods):
            player_idx = pod_idx // 2
            team_pod_idx = pod_idx % 2
            pod_key = f"player{player_idx}_pod{team_pod_idx}"
            
            # Base reward is progress through checkpoints
            batch_total_checkpoints = self.batch_checkpoint_counts * self.laps
            self.checkpoint_reward_cache.copy_(pod.current_checkpoint.float() / batch_total_checkpoints.float().unsqueeze(1))
            
            # Distance-based reward (closer to next checkpoint is better)
            diff = pod.position - next_cp_positions[pod_idx]
            distances = torch.norm(diff, dim=1, keepdim=True)
            self.distance_reward_cache.copy_(torch.clamp(1.0 - distances / 5000.0, 0.0, 1.0))
            
            # Velocity-based reward (higher speed towards checkpoint is better)
            direction_to_checkpoint = next_cp_positions[pod_idx] - pod.position
            direction_norm = torch.norm(direction_to_checkpoint, dim=1, keepdim=True)
            direction_norm = torch.where(direction_norm > 0, direction_norm, torch.ones_like(direction_norm))
            normalized_direction = direction_to_checkpoint / direction_norm
            
            velocity_alignment = torch.sum(normalized_direction * pod.velocity, dim=1, keepdim=True)
            self.velocity_reward_cache.copy_(torch.clamp(velocity_alignment / 500.0, -1.0, 1.0))
            
            # Combine rewards
            pod_reward = self.checkpoint_reward_cache + 0.01 * self.distance_reward_cache + 0.005 * self.velocity_reward_cache
            
            # Win/loss reward
            pod_reward = torch.where(race_completed[pod_idx], pod_reward + 10.0, pod_reward)
            
            # Opponent pods (from other player)
            opponent_player_idx = 1 - player_idx
            opponent_pods = [
                self.pods[opponent_player_idx*2], 
                self.pods[opponent_player_idx*2+1]
            ]
            
            # Penalty if opponent completes race
            for opp_idx in range(2):
                opp_pod_idx = opponent_player_idx * 2 + opp_idx
                pod_reward = torch.where(race_completed[opp_pod_idx], pod_reward - 5.0, pod_reward)
            
            rewards[pod_key] = pod_reward
        
        return rewards

    def get_checkpoints(self):
        """Return checkpoint positions for visualization"""
        if self.checkpoints is not None:
            # Return first batch's checkpoints
            return self.checkpoints[0].cpu().numpy().tolist()
        return []
    
    def get_pod_states(self):
        """Get current pod states for visualization"""
        pod_states = []
        for pod_idx, pod in enumerate(self.pods):
            # Get first batch item for visualization
            state = {
                'position': pod.position[0].cpu().numpy().tolist(),
                'velocity': pod.velocity[0].cpu().numpy().tolist(),
                'angle': float(pod.angle[0].cpu().numpy()),
                'current_checkpoint': int(pod.current_checkpoint[0].cpu().numpy()),
                'shield_cooldown': int(pod.shield_cooldown[0].cpu().numpy()),
                'boost_available': bool(pod.boost_available[0].cpu().numpy())
            }
            pod_states.append(state)
        return pod_states