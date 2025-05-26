import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from ..models.pod import Pod

# Game constants
WIDTH = 16000
HEIGHT = 9000
MAX_TURNS_WITHOUT_CHECKPOINT = 100  # Keep for reward calculation, not episode termination


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
                max_checkpoints: int = 8,
                timeout_penalty_weight: float = 0.01):
        self.device = device
        self.batch_size = batch_size
        self.base_num_checkpoints = num_checkpoints
        self.laps = laps

        # Add randomization parameters
        self.randomize_checkpoints = randomize_checkpoints
        self.min_checkpoints = min_checkpoints
        self.max_checkpoints = max_checkpoints
        
        # Add timeout penalty weight for reward calculation
        self.timeout_penalty_weight = timeout_penalty_weight

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
        self.timeout_penalty_cache = torch.zeros(self.batch_size, 1, device=self.device)
        
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
        """Get observations with role-specific information"""
        observations = {}
        
        for pod_idx, pod in enumerate(self.pods):
            player_idx = pod_idx // 2
            team_pod_idx = pod_idx % 2
            obs_key = f"player{player_idx}_pod{team_pod_idx}"
            
            # Base observations (current 48 dims)
            base_obs = self._get_base_observations(pod_idx, pod)
            
            # Role-specific observations (8 additional dims)
            role_obs = self._get_role_observations(pod_idx, team_pod_idx)
            
            # Combine to 56 dimensions total
            observations[obs_key] = torch.cat([base_obs, role_obs], dim=1)
        
        return observations

    def _get_role_observations(self, pod_idx, team_pod_idx):
        """Get role-specific observations including timeout information"""
        role_obs = torch.zeros(self.batch_size, 8, device=self.device)
        
        # Role identifier
        role_obs[:, 0] = float(team_pod_idx)  # 0 for runner, 1 for blocker
        
        # Add timeout progress for all pods (normalized)
        turns_since_checkpoint = self.turn_count - self.last_checkpoint_turn[:, pod_idx].unsqueeze(1)
        timeout_progress = torch.clamp(turns_since_checkpoint.float() / MAX_TURNS_WITHOUT_CHECKPOINT, 0.0, 2.0)
        role_obs[:, 1:2] = timeout_progress
        
        if team_pod_idx == 1:  # Blocker-specific observations
            player_idx = pod_idx // 2
            runner_idx = player_idx * 2
            opponent_player_idx = 1 - player_idx
            
            runner = self.pods[runner_idx]
            blocker = self.pods[pod_idx]
            
            # Distance to teammate runner (normalized)
            runner_distance = torch.norm(blocker.position - runner.position, dim=1, keepdim=True)
            role_obs[:, 2:3] = runner_distance / 8000.0
            
            # Runner's progress relative to blocker
            progress_diff = (runner.current_checkpoint - blocker.current_checkpoint).float()
            role_obs[:, 3:4] = torch.clamp(progress_diff / 5.0, -1.0, 1.0)
            
            # Closest opponent distance and relative position
            min_opp_distance = float('inf')
            closest_opp_pos = torch.zeros(self.batch_size, 2, device=self.device)
            
            for opp_idx in range(2):
                opp_pod = self.pods[opponent_player_idx * 2 + opp_idx]
                opp_distance = torch.norm(blocker.position - opp_pod.position, dim=1)
                
                closer_mask = opp_distance < min_opp_distance
                if closer_mask.any():
                    min_opp_distance = torch.where(closer_mask, opp_distance, min_opp_distance)
                    closest_opp_pos[closer_mask] = opp_pod.position[closer_mask]
            
            # Normalized distance to closest opponent
            role_obs[:, 4:5] = min_opp_distance.unsqueeze(1) / 8000.0
            
            # Relative position to closest opponent (normalized)
            rel_opp_pos = (closest_opp_pos - blocker.position) / 8000.0
            role_obs[:, 5:7] = rel_opp_pos
            
            # Blocking opportunity score (how well positioned to block)
            role_obs[:, 7:8] = self._calculate_blocking_opportunity(pod_idx)
            
        else:  # Runner-specific observations
            # For runners, add more race-specific information
            player_idx = pod_idx // 2
            opponent_player_idx = 1 - player_idx
            
            # Distance to closest opponent runner
            closest_opp_runner_distance = float('inf')
            opp_runner_idx = opponent_player_idx * 2  # Opponent's runner
            opp_runner = self.pods[opp_runner_idx]
            
            opp_distance = torch.norm(self.pods[pod_idx].position - opp_runner.position, dim=1, keepdim=True)
            role_obs[:, 2:3] = opp_distance / 8000.0
            
            # Progress relative to opponent runner
            progress_diff = (self.pods[pod_idx].current_checkpoint - opp_runner.current_checkpoint).float()
            role_obs[:, 3:4] = torch.clamp(progress_diff / 5.0, -1.0, 1.0)
            
            # Current speed normalized
            current_speed = torch.norm(self.pods[pod_idx].velocity, dim=1, keepdim=True)
            role_obs[:, 4:5] = torch.clamp(current_speed / 800.0, 0.0, 1.0)
            
            # Remaining dimensions for future use
            role_obs[:, 5:8] = 0.0
        
        return role_obs

    def _calculate_blocking_opportunity(self, blocker_idx):
        """Calculate how well positioned the blocker is to interfere with opponents"""
        blocker = self.pods[blocker_idx]
        player_idx = blocker_idx // 2
        opponent_player_idx = 1 - player_idx
        
        opportunity_score = torch.zeros(self.batch_size, 1, device=self.device)
        
        for opp_idx in range(2):
            opp_pod = self.pods[opponent_player_idx * 2 + opp_idx]
            
            # Get opponent's next checkpoint using batch-specific counts
            opp_next_cp_pos = torch.zeros(self.batch_size, 2, device=self.device)
            for b in range(self.batch_size):
                num_cp = self.batch_checkpoint_counts[b].item()
                opp_next_cp_idx = opp_pod.current_checkpoint[b] % num_cp
                # Ensure the index is within bounds
                opp_next_cp_idx = min(opp_next_cp_idx, num_cp - 1)
                opp_next_cp_pos[b] = self.checkpoints[b, opp_next_cp_idx]

            # Calculate if blocker can intercept opponent's path
            opp_to_cp = opp_next_cp_pos - opp_pod.position
            blocker_to_cp = opp_next_cp_pos - blocker.position
            
            # Simple interception opportunity based on relative distances
            opp_distance_to_cp = torch.norm(opp_to_cp, dim=1, keepdim=True)
            blocker_distance_to_cp = torch.norm(blocker_to_cp, dim=1, keepdim=True)
            
            # Higher score if blocker is closer to opponent's checkpoint
            intercept_score = torch.clamp(1.0 - blocker_distance_to_cp / (opp_distance_to_cp + 1e-6), 0.0, 1.0)
            opportunity_score += intercept_score
        
        return torch.clamp(opportunity_score / 2.0, 0.0, 1.0)  # Average and normalize
   
    def _get_base_observations(self, pod_idx, pod) -> torch.Tensor:
        """Get base observations for a specific pod - updated with absolute angles"""
        
        # Pre-compute next checkpoint indices for this pod (with modulo based on batch-specific counts)
        next_cp_idx = torch.zeros_like(pod.current_checkpoint)
        next_next_cp_idx = torch.zeros_like(pod.current_checkpoint)
        
        for b in range(self.batch_size):
            num_cp = self.batch_checkpoint_counts[b].item()
            total_cp = num_cp * self.laps
            
            if pod.current_checkpoint[b] < total_cp:
                next_cp_idx[b] = pod.current_checkpoint[b] % num_cp
                next_next_cp_idx[b] = (pod.current_checkpoint[b] + 1) % num_cp
            else:
                # Race finished, use last checkpoint
                next_cp_idx[b] = (num_cp - 1)
                next_next_cp_idx[b] = (num_cp - 1)
        
        # Get next checkpoint positions (vectorized)
        next_cp_pos = torch.zeros(self.batch_size, 2, device=self.device)
        for b in range(self.batch_size):
            idx = min(next_cp_idx[b].item(), self.batch_checkpoint_counts[b].item() - 1)
            next_cp_pos[b] = self.checkpoints[b, idx]
        
        # Get next-next checkpoint positions (vectorized)
        next_next_cp_pos = torch.zeros(self.batch_size, 2, device=self.device)
        for b in range(self.batch_size):
            idx = min(next_next_cp_idx[b].item(), self.batch_checkpoint_counts[b].item() - 1)
            next_next_cp_pos[b] = self.checkpoints[b, idx]
        
        # Reuse cached tensors for normalization
        # Normalize positions (in-place operations)
        self.normalized_position_cache.copy_(pod.position)
        self.normalized_position_cache[:, 0].mul_(2.0/WIDTH).sub_(1.0)  # Scale to [-1, 1]
        self.normalized_position_cache[:, 1].mul_(2.0/HEIGHT).sub_(1.0)  # Scale to [-1, 1]
        
        # Normalize velocities (in-place operations)
        self.normalized_velocity_cache.copy_(pod.velocity)
        self.normalized_velocity_cache.div_(1000.0)  # Typical max speed is around 600-800
        
        # Normalize checkpoint positions (in-place operations)
        self.normalized_next_cp_cache.copy_(next_cp_pos)
        self.normalized_next_cp_cache[:, 0].mul_(2.0/WIDTH).sub_(1.0)
        self.normalized_next_cp_cache[:, 1].mul_(2.0/HEIGHT).sub_(1.0)
        
        self.normalized_next_next_cp_cache.copy_(next_next_cp_pos)
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
        angle_to_next_cp = pod.angle_to(next_cp_pos)
        angle_to_next_cp_rad = torch.deg2rad(angle_to_next_cp)
        angle_to_next_cp_sin = torch.sin(angle_to_next_cp_rad)
        angle_to_next_cp_cos = torch.cos(angle_to_next_cp_rad)

        # Relative angle (how much to turn)
        relative_angle = angle_to_next_cp - pod.angle
        relative_angle = ((relative_angle + 180) % 360) - 180
        relative_angle_normalized = relative_angle / 180.0

        # Speed and direction metrics
        speed_magnitude = torch.norm(pod.velocity, dim=1, keepdim=True) / 800.0
        distance_to_next_cp = torch.norm(next_cp_pos - pod.position, dim=1, keepdim=True)
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
                opp_next_cp_normalized = torch.zeros(self.batch_size, 1, device=self.device)
                for b in range(self.batch_size):
                    num_cp = self.batch_checkpoint_counts[b].item()
                    opp_next_cp_normalized[b] = (opp_pod.current_checkpoint[b] % num_cp).float() / num_cp

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
        base_observations = torch.cat([pod_specific_obs, all_opponent_obs], dim=1)
        
        return base_observations
    
    def step(self, actions: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Execute one step in the environment - UPDATED to accept 4-output format from model
        
        Actions format: Dict[str, torch.Tensor] where each tensor has shape [batch_size, 4]
        - action[:, 0] = angle adjustment [-1, 1]
        - action[:, 1] = thrust [0, 1]
        - action[:, 2] = shield probability [0, 1]
        - action[:, 3] = boost probability [0, 1]
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
                
                # Validate action format
                if action.shape[1] != 4:
                    raise ValueError(f"Invalid action shape: {action.shape}. Expected [batch_size, 4]")
                
                # Extract action components
                angle_target = action[:, 0].unsqueeze(1)  # [-1, 1]
                thrust_value = action[:, 1].unsqueeze(1)  # [0, 1]
                shield_prob = action[:, 2].unsqueeze(1)   # [0, 1]
                boost_prob = action[:, 3].unsqueeze(1)    # [0, 1]
                
                # Determine action type based on probabilities
                shield_mask = shield_prob > 0.5
                boost_mask = (boost_prob > 0.5) & pod.boost_available.bool().unsqueeze(1)
                normal_thrust_mask = ~(shield_mask | boost_mask)
                
                # Get next checkpoint position using batch-specific counts
                next_cp_positions = torch.zeros(self.batch_size, 2, device=self.device)
                for b in range(self.batch_size):
                    num_cp = self.batch_checkpoint_counts[b].item()
                    next_cp_idx = pod.current_checkpoint[b] % num_cp
                    # Ensure the index is within bounds
                    next_cp_idx = min(next_cp_idx.item(), num_cp - 1)
                    next_cp_positions[b] = self.checkpoints[b, next_cp_idx]
                
                # Calculate target angle
                base_angle = pod.angle_to(next_cp_positions)
                # Convert normalized angle_target [-1, 1] to degree adjustment [-18, 18]
                angle_adjustment = angle_target * 18.0
                target_angle = base_angle + angle_adjustment
                
                # Rotate pod
                pod.rotate(target_angle)
                
                # Apply actions based on determined masks
                # Apply shield
                if shield_mask.any():
                    shield_indices = shield_mask.squeeze(-1)
                    if shield_indices.any():
                        pod.apply_shield_selective(shield_indices)
                
                # Apply boost
                if boost_mask.any():
                    boost_indices = boost_mask.squeeze(-1)
                    if boost_indices.any():
                        pod.apply_boost_selective(boost_indices)
                
                # Apply normal thrust
                if normal_thrust_mask.any():
                    # Check shield cooldown - pods can't thrust while shield is cooling down
                    can_thrust_mask = pod.shield_cooldown == 0
                    normal_thrust_mask_squeezed = normal_thrust_mask.squeeze(-1)
                    can_thrust_mask_squeezed = can_thrust_mask.squeeze(-1)
                    final_thrust_mask = normal_thrust_mask_squeezed & can_thrust_mask_squeezed

                    if final_thrust_mask.any():
                        # Convert normalized thrust [0, 1] to game thrust [0, 100]
                        thrust_to_apply = torch.zeros(self.batch_size, 1, device=self.device)
                        
                        # FIX: Use direct indexing instead of torch.where to avoid broadcasting issues
                        # Get indices where thrust should be applied
                        thrust_indices = torch.where(final_thrust_mask)[0]
                        
                        if len(thrust_indices) > 0:
                            # Apply thrust only to the selected indices
                            thrust_values_to_apply = thrust_value[thrust_indices] * 100.0
                            thrust_to_apply[thrust_indices] = thrust_values_to_apply
                        
                        pod.apply_thrust(thrust_to_apply)
                        
        # Move pods and handle collisions
        self._simulate_movement()
        
        # Check for checkpoints
        self._check_checkpoints()
        
        # Check for race completion (REMOVED timeout-based termination)
        self._update_race_status()
        
        # Get observations, rewards, and done status
        observations = self._get_observations()
        rewards = self._calculate_rewards()
        
        # Additional info
        info = {
            "turn_count": self.turn_count,
            "checkpoint_progress": torch.stack([pod.current_checkpoint for pod in self.pods]),
            "timeout_penalties": self._get_timeout_penalties(),  # Add timeout info for debugging
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
                total_cp = num_cp * self.laps
                
                # Only check checkpoints if race isn't finished for this pod
                if pod.current_checkpoint[b] < total_cp:
                    next_cp_idx = pod.current_checkpoint[b] % num_cp
                    next_cp_positions[b] = self.checkpoints[b, next_cp_idx]
                else:
                    # Race finished, set dummy position (won't be used)
                    next_cp_positions[b] = self.checkpoints[b, 0]
            
            # Check distances to next checkpoints only for unfinished races
            diff = pod.position - next_cp_positions
            distances = torch.norm(diff, dim=1)
            
            # Create mask for pods that haven't finished the race
            batch_total_checkpoints = self.batch_checkpoint_counts * self.laps
            not_finished = pod.current_checkpoint.squeeze() < batch_total_checkpoints
            
            # Only check checkpoint collision for unfinished pods
            reached = (distances <= 600) & not_finished  # Checkpoint radius
            
            # Update checkpoint counters and last checkpoint turn
            if reached.any():
                pod.current_checkpoint[reached] += 1
                self.last_checkpoint_turn[reached, pod_idx] = self.turn_count[reached].squeeze()

    def _get_next_checkpoint_positions(self) -> Dict[int, torch.Tensor]:
        """Get next checkpoint positions for all pods"""
        next_cp_positions = {}
        
        for pod_idx, pod in enumerate(self.pods):
            # Get batch-specific next checkpoint positions
            positions = torch.zeros(self.batch_size, 2, device=self.device)
            
            for b in range(self.batch_size):
                num_cp = self.batch_checkpoint_counts[b].item()
                next_cp_idx = pod.current_checkpoint[b] % num_cp
                # Ensure the index is within bounds
                next_cp_idx = min(next_cp_idx, num_cp - 1)
                positions[b] = self.checkpoints[b, next_cp_idx]
            
            next_cp_positions[pod_idx] = positions
        
        return next_cp_positions

    def _update_race_status(self) -> None:
        """Update race status - MODIFIED to only end on race completion, not timeout"""
        new_done = torch.zeros_like(self.done)
        
        # Calculate total checkpoints per batch 
        # We need to complete all laps AND cross the finish line (checkpoint 0) one final time
        batch_total_checkpoints = self.batch_checkpoint_counts * self.laps
        
        # Check if any pod has completed all checkpoints
        for player_idx in range(2):
            player_pods = [self.pods[player_idx*2], self.pods[player_idx*2+1]]
            
            for pod_idx_in_team, pod in enumerate(player_pods):
                global_pod_idx = player_idx * 2 + pod_idx_in_team
                
                # Check if pod has completed all checkpoints including final finish line crossing
                # For 3 CP, 3 laps: need to reach checkpoint 9 (3*3 = 9 total checkpoints)
                # print(f"Pod {global_pod_idx} current checkpoint: {pod.current_checkpoint.squeeze().item()}, total checkpoints: {batch_total_checkpoints[0].item()}")
                race_completed = pod.current_checkpoint.squeeze() >= batch_total_checkpoints
                new_done = new_done | race_completed
        
        # REMOVED: Timeout-based episode termination
        # Episodes now only end when race is actually completed
        
        self.done = new_done
            
    def _get_timeout_penalties(self) -> Dict[int, torch.Tensor]:
        """Get timeout penalties for each pod (for debugging/info)"""
        timeout_penalties = {}
        for pod_idx in range(4):
            turns_since_checkpoint = self.turn_count - self.last_checkpoint_turn[:, pod_idx].unsqueeze(1)
            timeout_penalty = torch.clamp(turns_since_checkpoint.float() / MAX_TURNS_WITHOUT_CHECKPOINT, 0.0, 2.0)
            timeout_penalties[pod_idx] = timeout_penalty
        return timeout_penalties
    
    def get_reward_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get detailed reward breakdown for debugging"""
        if not hasattr(self, '_last_reward_breakdown'):
            return {}
        return self._last_reward_breakdown

    def _calculate_rewards(self) -> Dict[str, torch.Tensor]:
        """Calculate role-specific rewards for each pod - UPDATED with timeout penalties"""
        rewards = {}
        self._last_reward_breakdown = {}
        
        # Pre-compute positions and states
        next_cp_positions = self._get_next_checkpoint_positions()
        
        for pod_idx, pod in enumerate(self.pods):
            player_idx = pod_idx // 2
            team_pod_idx = pod_idx % 2
            pod_key = f"player{player_idx}_pod{team_pod_idx}"
            
            if team_pod_idx == 0:  # Runner pod
                rewards[pod_key] = self._calculate_runner_reward(pod_idx, pod, next_cp_positions)
                # Store breakdown for debugging
                self._last_reward_breakdown[pod_key] = {
                    'type': 'runner',
                    'checkpoint_progress': float(pod.current_checkpoint[0]),
                    'total_reward': float(rewards[pod_key][0])
                }
            else:  # Blocker pod
                rewards[pod_key] = self._calculate_blocker_reward(pod_idx, pod, next_cp_positions)
                # Store breakdown for debugging
                self._last_reward_breakdown[pod_key] = {
                    'type': 'blocker', 
                    'checkpoint_progress': float(pod.current_checkpoint[0]),
                    'total_reward': float(rewards[pod_key][0])
                }
        
        return rewards

    def _calculate_timeout_penalty(self, pod_idx):
        """Calculate timeout penalty for a specific pod"""
        turns_since_checkpoint = self.turn_count - self.last_checkpoint_turn[:, pod_idx].unsqueeze(1)
        
        # Progressive penalty that increases with time since last checkpoint
        timeout_progress = turns_since_checkpoint.float() / MAX_TURNS_WITHOUT_CHECKPOINT
        
        # Exponential penalty - gets severe after threshold
        penalty = torch.where(
            timeout_progress > 1.0,
            torch.exp(timeout_progress - 1.0) - 1.0,  # Exponential penalty after threshold
            timeout_progress ** 2 * 0.1  # Quadratic penalty before threshold
        )
        
        return -penalty * self.timeout_penalty_weight

    def _calculate_runner_reward(self, pod_idx, pod, next_cp_positions):
        """Reward runner for racing performance - UPDATED with timeout penalty"""
        # Checkpoint progress reward (normalized)
        batch_total_checkpoints = (self.batch_checkpoint_counts * self.laps).float().unsqueeze(1)
        checkpoint_reward = pod.current_checkpoint.float() / batch_total_checkpoints
        
        # Distance to next checkpoint (more significant weight)
        diff = pod.position - next_cp_positions[pod_idx]
        distances = torch.norm(diff, dim=1, keepdim=True)
        distance_reward = torch.clamp(1.0 - distances / 5000.0, 0.0, 1.0)
        
        # Speed towards checkpoint (more significant weight)
        direction_to_checkpoint = next_cp_positions[pod_idx] - pod.position
        direction_norm = torch.norm(direction_to_checkpoint, dim=1, keepdim=True)
        direction_norm = torch.where(direction_norm > 0, direction_norm, torch.ones_like(direction_norm))
        normalized_direction = direction_to_checkpoint / direction_norm
        velocity_alignment = torch.sum(normalized_direction * pod.velocity, dim=1, keepdim=True)
        velocity_reward = torch.clamp(velocity_alignment / 500.0, -1.0, 1.0)
        
        # Base speed reward (encourage maintaining speed)
        speed_magnitude = torch.norm(pod.velocity, dim=1, keepdim=True)
        speed_reward = torch.clamp(speed_magnitude / 600.0, 0.0, 1.0)
        
        # Penalty for being far from optimal racing line
        optimal_position_reward = distance_reward  # Reuse distance calculation
        
        # ADDED: Timeout penalty to discourage stalling
        timeout_penalty = self._calculate_timeout_penalty(pod_idx)
        
        # REBALANCED: More immediate rewards, less dependence on checkpoint progress
        total_reward = (
            checkpoint_reward * 1.0 +        # Reduced from 2.0
            distance_reward * 0.3 +          # Increased from 0.02  
            velocity_reward * 0.2 +          # Increased from 0.01
            speed_reward * 0.1 +             # New: encourage speed
            optimal_position_reward * 0.1 +  # New: encourage good positioning
            timeout_penalty                  # NEW: penalize stalling
        )
        
        return total_reward

    def _calculate_blocker_reward(self, pod_idx, pod, next_cp_positions):
        """Reward blocker for blocking and supporting runner - UPDATED with timeout penalty"""
        player_idx = pod_idx // 2
        opponent_player_idx = 1 - player_idx
        runner_idx = player_idx * 2  # Teammate runner
        
        # Base progress reward (same normalization as runner)
        batch_total_checkpoints = (self.batch_checkpoint_counts * self.laps).float().unsqueeze(1)
        checkpoint_reward = pod.current_checkpoint.float() / batch_total_checkpoints
        
        # Blocking rewards
        blocking_reward = self._calculate_blocking_reward(pod_idx, opponent_player_idx)
        
        # Support runner reward
        support_reward = self._calculate_support_reward(pod_idx, runner_idx)
        
        # ADDED: Timeout penalty to discourage excessive shielding/stalling
        timeout_penalty = self._calculate_timeout_penalty(pod_idx)
        
        # ADDED: Anti-shield spam penalty
        shield_spam_penalty = self._calculate_shield_spam_penalty(pod_idx)
        
        # REBALANCED: Reduce blocking emphasis, normalize total reward scale
        total_reward = (
            checkpoint_reward * 0.3 +        # Reduced from 0.5
            blocking_reward * 0.4 +          # Reduced from 1.0
            support_reward * 0.3 +           # Reduced from 0.5
            timeout_penalty +                # NEW: penalize stalling
            shield_spam_penalty              # NEW: penalize shield spam
        )
        
        return total_reward

    def _calculate_shield_spam_penalty(self, pod_idx):
        """Penalize excessive shield usage that leads to stalling"""
        pod = self.pods[pod_idx]
        
        # Count how often shield is on cooldown (indicating recent shield use)
        shield_usage_ratio = (pod.shield_cooldown > 0).float()
        
        # Calculate turns since last checkpoint
        turns_since_checkpoint = self.turn_count - self.last_checkpoint_turn[:, pod_idx].unsqueeze(1)
        timeout_progress = torch.clamp(turns_since_checkpoint.float() / MAX_TURNS_WITHOUT_CHECKPOINT, 0.0, 2.0)
        
        # Penalty increases if shield is used frequently while not progressing
        shield_penalty = shield_usage_ratio * timeout_progress * 0.05
        
        return -shield_penalty
    
    def _calculate_blocking_reward(self, blocker_idx, opponent_player_idx):
        """Reward for blocking opponent pods - UPDATED to discourage pure stalling"""
        blocker = self.pods[blocker_idx]
        opponent_pods = [self.pods[opponent_player_idx * 2], self.pods[opponent_player_idx * 2 + 1]]
        
        total_blocking_reward = torch.zeros(self.batch_size, 1, device=self.device)
        
        for opp_pod in opponent_pods:
            # Get opponent's next checkpoint using batch-specific counts
            opp_next_cp_pos = torch.zeros(self.batch_size, 2, device=self.device)
            for b in range(self.batch_size):
                num_cp = self.batch_checkpoint_counts[b].item()
                opp_next_cp_idx = opp_pod.current_checkpoint[b] % num_cp
                # Ensure the index is within bounds
                opp_next_cp_idx = min(opp_next_cp_idx.item(), num_cp - 1)
                opp_next_cp_pos[b] = self.checkpoints[b, opp_next_cp_idx]

            # Reward for being between opponent and their checkpoint
            opp_to_cp = opp_next_cp_pos - opp_pod.position
            opp_to_blocker = blocker.position - opp_pod.position
            
            # Calculate if blocker is in opponent's path
            opp_to_cp_norm = torch.norm(opp_to_cp, dim=1, keepdim=True)
            opp_to_cp_norm = torch.where(opp_to_cp_norm > 0, opp_to_cp_norm, torch.ones_like(opp_to_cp_norm))
            opp_direction = opp_to_cp / opp_to_cp_norm
            
            # Project blocker position onto opponent's path
            projection = torch.sum(opp_to_blocker * opp_direction, dim=1, keepdim=True)
            
            # Reward for being in front of opponent on their path
            path_blocking = torch.clamp(projection / 1000.0, 0.0, 1.0)
            
            # Distance-based blocking reward
            blocker_to_opp = torch.norm(blocker.position - opp_pod.position, dim=1, keepdim=True)
            proximity_reward = torch.clamp(1.0 - blocker_to_opp / 2000.0, 0.0, 1.0)
            
            # ADDED: Reward active blocking (when blocker is also moving)
            blocker_speed = torch.norm(blocker.velocity, dim=1, keepdim=True)
            activity_multiplier = torch.clamp(blocker_speed / 200.0, 0.1, 1.0)  # Min 10% reward
            
            blocking_component = (0.7 * path_blocking + 0.3 * proximity_reward) * activity_multiplier
            total_blocking_reward += blocking_component
        
        return total_blocking_reward

    def _calculate_support_reward(self, blocker_idx, runner_idx):
        """Reward blocker for supporting runner - UPDATED to encourage active support"""
        blocker = self.pods[blocker_idx]
        runner = self.pods[runner_idx]
        
        # Reward for not being too far from runner (coordination)
        distance_to_runner = torch.norm(blocker.position - runner.position, dim=1, keepdim=True)
        coordination_reward = torch.clamp(1.0 - distance_to_runner / 8000.0, 0.0, 1.0)
        
        # Small penalty if blocker is ahead of runner (runner should lead)
        progress_diff = runner.current_checkpoint - blocker.current_checkpoint
        leadership_reward = torch.clamp(progress_diff.float() * 0.1, -0.5, 0.5)
        
        # ADDED: Reward when both pods are making progress
        runner_turns_since_cp = self.turn_count - self.last_checkpoint_turn[:, runner_idx].unsqueeze(1)
        blocker_turns_since_cp = self.turn_count - self.last_checkpoint_turn[:, blocker_idx].unsqueeze(1)
        
        # Bonus when team is making consistent progress
        team_progress_bonus = torch.clamp(
            1.0 - (runner_turns_since_cp.float() + blocker_turns_since_cp.float()) / (2 * MAX_TURNS_WITHOUT_CHECKPOINT),
            0.0, 0.2
        )
        
        return 0.3 * coordination_reward + 0.2 * leadership_reward + team_progress_bonus

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
