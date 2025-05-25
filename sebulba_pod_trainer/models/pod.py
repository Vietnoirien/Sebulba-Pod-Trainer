import torch
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

# Constants from the game
EPSILON = 0.00001
POD_RADIUS = 400
CHECKPOINT_RADIUS = 600
FRICTION = 0.85
MIN_IMPULSE = 120.0
MAX_THRUST = 100
BOOST_THRUST = 650
MAX_ROTATION = 18  # degrees per turn
SHIELD_MASS_MULTIPLIER = 10


class Pod:
    """
    Represents a pod in the race with PyTorch tensor-based physics.
    All operations are vectorized to support batch processing for efficient training.
    """
    def __init__(self, 
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 batch_size: int = 1):
        # Position, velocity, and angle are represented as tensors
        self.device = device
        self.batch_size = batch_size
        
        # Initialize tensors with zeros
        self.position = torch.zeros(batch_size, 2, device=device)  # x, y
        self.velocity = torch.zeros(batch_size, 2, device=device)  # vx, vy
        self.angle = torch.zeros(batch_size, 1, device=device)     # angle in degrees
        self.current_checkpoint = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        # Game state
        self.shield_cooldown = torch.zeros(batch_size, 1, device=device)
        self.boost_available = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        self.mass = torch.ones(batch_size, 1, device=device)
        
    def distance(self, other_position: torch.Tensor) -> torch.Tensor:
        """Calculate distance to another point or batch of points"""
        return torch.norm(self.position - other_position, dim=1, keepdim=True)
    
    def angle_to(self, target_position: torch.Tensor) -> torch.Tensor:
        """Calculate angle to target in degrees"""
        delta = target_position - self.position
        # Calculate angle in radians and convert to degrees
        angle_rad = torch.atan2(delta[:, 1], delta[:, 0])
        return torch.rad2deg(angle_rad).unsqueeze(1)
    
    def rotate(self, target_angle: torch.Tensor) -> None:
        """Rotate pod towards target angle with maximum rotation constraint"""
        angle_diff = (target_angle - self.angle) % 360
        # Ensure we rotate the shortest way
        mask = angle_diff > 180
        angle_diff[mask] = angle_diff[mask] - 360
        
        # Apply maximum rotation constraint
        angle_diff = torch.clamp(angle_diff, -MAX_ROTATION, MAX_ROTATION)
        self.angle = (self.angle + angle_diff) % 360
    
    def apply_thrust(self, thrust: torch.Tensor) -> None:
        """Apply thrust in the direction the pod is facing"""
        # Convert angle to radians
        angle_rad = torch.deg2rad(self.angle)
        
        # Calculate thrust vector
        thrust_vector = torch.cat([
            torch.cos(angle_rad) * thrust,
            torch.sin(angle_rad) * thrust
        ], dim=1)
        
        # Apply thrust to velocity
        self.velocity += thrust_vector
    
    def apply_shield(self) -> None:
        """Activate shield if available"""
        available_mask = self.shield_cooldown == 0
        self.mass[available_mask] = SHIELD_MASS_MULTIPLIER
        self.shield_cooldown[available_mask] = 3  # Shield lasts for 3 turns
    
    def apply_boost(self) -> None:
        """Apply boost if available"""
        available_mask = self.boost_available
        self.boost_available[available_mask] = False
        # Apply boost as thrust
        self.apply_thrust(torch.where(
            available_mask,
            torch.tensor(BOOST_THRUST, device=self.device),
            torch.tensor(MAX_THRUST, device=self.device)
        ))
    
    def move(self) -> None:
        """Move pod according to its velocity"""
        self.position += self.velocity
        # Apply friction
        self.velocity *= FRICTION
        # Round position and truncate velocity as per game rules
        self.position = torch.round(self.position)
        self.velocity = torch.trunc(self.velocity)
        
        # Decrease shield cooldown if active
        self.shield_cooldown = torch.clamp(self.shield_cooldown - 1, min=0)
        # Reset mass if shield is no longer active
        mask = self.shield_cooldown == 0
        self.mass[mask] = 1.0
    
    def check_checkpoint(self, checkpoints: torch.Tensor) -> torch.Tensor:
        """
        Check if pod has reached its next checkpoint
        
        Args:
            checkpoints: Tensor of shape [num_checkpoints, 2] with checkpoint coordinates
            
        Returns:
            Tensor of booleans indicating which pods reached their next checkpoint
        """
        next_checkpoint_indices = self.current_checkpoint % checkpoints.shape[0]
        next_checkpoint_positions = checkpoints[next_checkpoint_indices.squeeze(-1)]
        
        distances = self.distance(next_checkpoint_positions)
        reached = distances <= CHECKPOINT_RADIUS
        
        # Update current checkpoint for pods that reached their next checkpoint
        self.current_checkpoint[reached] += 1
        
        return reached
    
    def get_collision_time(self, other_pod: 'Pod') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate time of collision with another pod
        
        Returns:
            Tuple of (collision_time, collision_mask) where collision_mask indicates
            which pod pairs will collide
        """
        # Check instant collision
        distance = self.distance(other_pod.position)
        instant_collision = distance <= 2 * POD_RADIUS
        
        # Calculate relative position and velocity
        rel_pos = self.position - other_pod.position
        rel_vel = self.velocity - other_pod.velocity
        
        # Check if pods have same velocity
        same_vel = torch.all(rel_vel == 0, dim=1, keepdim=True)
        
        # Calculate quadratic equation coefficients
        a = torch.sum(rel_vel * rel_vel, dim=1, keepdim=True)
        b = 2.0 * torch.sum(rel_pos * rel_vel, dim=1, keepdim=True)
        c = torch.sum(rel_pos * rel_pos, dim=1, keepdim=True) - (2 * POD_RADIUS) ** 2
        
        # Calculate discriminant
        delta = b * b - 4.0 * a * c
        
        # Check if collision will occur
        collision_possible = (a > 0) & (delta >= 0) & ~same_vel
        
        # Calculate collision time
        collision_time = torch.ones_like(a) * float('inf')
        valid_collision = collision_possible & ~instant_collision
        
        if valid_collision.any():
            t = (-b - torch.sqrt(delta)) / (2.0 * a)
            # Only consider future collisions
            valid_t = (t > 0) & valid_collision
            collision_time[valid_t] = t[valid_t]
        
        # For instant collisions, time is 0
        collision_time[instant_collision] = 0.0
        
        return collision_time, collision_possible | instant_collision
    
    def resolve_collision(self, other_pod: 'Pod') -> None:
        """Resolve collision with another pod using elastic collision physics"""
        # Calculate normal vector
        normal = self.position - other_pod.position
        distance = torch.norm(normal, dim=1, keepdim=True)
        
        # Avoid division by zero
        mask = distance > 0
        # Fix: Properly broadcast the mask to match the normal tensor shape
        normal_mask = mask.expand(-1, normal.size(1))
        normal[normal_mask] = normal[normal_mask] / distance[mask].repeat(1, normal.size(1))
        
        # Calculate relative velocity
        rel_vel = self.velocity - other_pod.velocity
        
        # Calculate impulse
        impulse_numerator = torch.sum(normal * rel_vel, dim=1, keepdim=True)
        impulse_denominator = 1/self.mass + 1/other_pod.mass
        impulse = impulse_numerator / impulse_denominator
        
        # Apply minimum impulse
        impulse = torch.maximum(impulse, torch.tensor(MIN_IMPULSE, device=self.device))
        
        # Apply impulse to velocities
        impulse_vector = -normal * impulse
        self.velocity += impulse_vector / self.mass
        other_pod.velocity -= impulse_vector / other_pod.mass
        
        # Ensure minimum separation
        overlap = 2 * POD_RADIUS - distance
        overlap_mask = overlap > 0
        if overlap_mask.any():
            separation = normal * (overlap / 2 + EPSILON)
            # Fix: Properly broadcast the overlap mask
            overlap_mask_expanded = overlap_mask.expand(-1, separation.size(1))
            self.position[overlap_mask_expanded] -= separation[overlap_mask_expanded]
            other_pod.position[overlap_mask_expanded] += separation[overlap_mask_expanded]
