import torch
from sebulba_pod_trainer.environment.race_env import RaceEnvironment

# Monkey patch the _check_checkpoints method to fix the indexing issue
def fixed_check_checkpoints(self):
    """Check if pods have reached their next checkpoints"""
    for pod_idx, pod in enumerate(self.pods):
        # Get batch-specific next checkpoint positions
        next_cp_idx = pod.current_checkpoint % self.num_checkpoints
        next_cp_positions = torch.stack([
            self.checkpoints[b, next_cp_idx[b].item()] 
            for b in range(self.batch_size)
        ])
        
        # Check distances to next checkpoints
        distances = pod.distance(next_cp_positions)
        reached = distances <= 600  # Checkpoint radius
        
        # Update checkpoint counters and last checkpoint turn
        if reached.any():
            pod.current_checkpoint[reached] += 1
            # Fix: Handle the indexing properly for the last_checkpoint_turn tensor
            for b in range(self.batch_size):
                if reached[b]:
                    self.last_checkpoint_turn[b, pod_idx] = self.turn_count[b]

# Apply the monkey patch
RaceEnvironment._check_checkpoints = fixed_check_checkpoints