import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any
import time
import os
import threading
from pathlib import Path
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from ..models.neural_pod import PodNetwork
from ..environment.optimized_race_env import OptimizedRaceEnvironment

class PPOTrainer:
    """
    Proximal Policy Optimization trainer for pod racing using OptimizedRaceEnvironment
    """
    def __init__(self,
                env: OptimizedRaceEnvironment,
                learning_rate: float = 3e-4,
                batch_size: int = 64,
                mini_batch_size: int = 16,
                ppo_epochs: int = 10,
                clip_param: float = 0.2,
                value_coef: float = 0.5,
                entropy_coef: float = 0.01,
                max_grad_norm: float = 0.5,
                gamma: float = 0.99,
                gae_lambda: float = 0.95,
                device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                multi_gpu: bool = False,
                devices: List[int] = None,
                use_mixed_precision: bool = True,
                network_config: Dict = None):
        
        print(f"Initializing PPOTrainer with device={device}, multi_gpu={multi_gpu}, devices={devices}")
        
        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.multi_gpu = multi_gpu
        self.devices = devices if devices else []
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()

        # Use network_config if provided, otherwise use defaults compatible with PodNetwork
        if network_config is None:
            self.network_config = {
                'observation_dim': 56,
                'hidden_layers': [
                    {'type': 'Linear+ReLU', 'size': 24},
                    {'type': 'Linear+ReLU', 'size': 16}
                ],
                'policy_hidden_size': 12,
                'value_hidden_size': 12,
                'action_hidden_size': 12
            }
        else:
            self.network_config = network_config
            
        # Initialize mixed precision tools if enabled
        if self.use_mixed_precision:
            try:
                self.scaler = GradScaler()
                print("Mixed precision training enabled")
            except Exception as e:
                print(f"Error initializing GradScaler: {e}")
                self.use_mixed_precision = False
                print("Mixed precision disabled due to errors")
        
        # Force CUDA to initialize on all selected GPUs
        if self.multi_gpu and len(self.devices) > 1:
            print(f"Initializing {len(self.devices)} GPUs: {self.devices}")
            for device_id in self.devices:
                try:
                    dummy_tensor = torch.zeros(1, device=f"cuda:{device_id}")
                    print(f"Successfully initialized GPU {device_id}")
                except Exception as e:
                    print(f"Error initializing GPU {device_id}: {e}")
        
        # Create networks for each pod
        self.pod_networks = {}
        self.optimizers = {}
        
        print("Creating pod networks...")
        # Initialize networks for all 4 pods (2 per player)
        for player_idx in range(2):
            for pod_idx in range(2):
                pod_key = f"player{player_idx}_pod{pod_idx}"
                
                # Determine device for this pod
                if self.multi_gpu and len(self.devices) > 1:
                    # Distribute pods across GPUs
                    gpu_idx = (player_idx * 2 + pod_idx) % len(self.devices)
                    pod_device = torch.device(f"cuda:{self.devices[gpu_idx]}")
                else:
                    pod_device = self.device
                
                print(f"Creating network for {pod_key} on device {pod_device}")
                
                # Create network with updated configuration
                network = PodNetwork(
                    observation_dim=self.network_config.get('observation_dim', 56),
                    hidden_layers=self.network_config.get('hidden_layers', []),
                    policy_hidden_size=self.network_config.get('policy_hidden_size', 12),
                    value_hidden_size=self.network_config.get('value_hidden_size', 12),
                    action_hidden_size=self.network_config.get('action_hidden_size', 12)
                ).to(pod_device)
                
                self.pod_networks[pod_key] = network
                self.optimizers[pod_key] = optim.Adam(
                    network.parameters(),
                    lr=learning_rate
                )
                print(f"Successfully created {pod_key}")
        
        # Pre-compute tensor constants for entropy calculation
        print("Setting up tensor constants...")
        self._setup_tensor_constants()
        
        # Storage for trajectory data
        print("Initializing storage...")
        self.reset_storage()
        print("PPOTrainer initialization complete")
    
    def _setup_tensor_constants(self):
        """Pre-compute tensor constants to avoid float operations during training"""
        # Constants for entropy calculation
        self.angle_entropy_constant = torch.tensor(-0.5 * np.log(2 * np.pi * 0.01), device=self.device)
        self.thrust_entropy_constant = torch.tensor(-0.5 * np.log(2 * np.pi * 0.01), device=self.device)
        self.log_eps = torch.tensor(1e-8, device=self.device)
        self.pi_tensor = torch.tensor(np.pi, device=self.device)
        self.two_pi_tensor = torch.tensor(2 * np.pi, device=self.device)
        
        # Standard deviations as tensors
        self.angle_std_tensor = torch.tensor(0.1, device=self.device)
        self.thrust_std_tensor = torch.tensor(0.1, device=self.device)
    
    def reset_storage(self):
        """Reset storage for new trajectories"""
        self.storage = {
            pod_key: {
                'observations': [],
                'actions': [],
                'rewards': [],
                'values': [],
                'log_probs': [],
                'dones': []
            } for pod_key in self.pod_networks.keys()
        }
    
    def get_action_log_prob(self, network, observation, action):
        """Calculate log probability of an action for the 4-action format - fully tensorized with NaN protection"""
        # Handle DataParallel models
        if isinstance(network, nn.DataParallel):
            module = network.module
        else:
            module = network

        # Use autocast for mixed precision if enabled
        with autocast('cuda', enabled=self.use_mixed_precision):
            actions, _, _ = module(observation)
        
        # Convert to float32 for numerical stability
        actions = actions.float()
        action = action.float()
        
        # Check for NaN values and replace them
        if torch.isnan(actions).any():
            print(f"Warning: NaN detected in network output, replacing with zeros")
            actions = torch.where(torch.isnan(actions), torch.zeros_like(actions), actions)
        
        # Extract action components
        angle_pred = actions[:, 0:1]      # [-1, 1]
        thrust_pred = actions[:, 1:2]     # [0, 1]
        shield_prob = actions[:, 2:3]     # [0, 1]
        boost_prob = actions[:, 3:4]      # [0, 1]
        
        angle_actual = action[:, 0:1]
        thrust_actual = action[:, 1:2]
        shield_actual = action[:, 2:3]
        boost_actual = action[:, 3:4]
        
        # Clamp predictions to valid ranges to prevent NaN
        angle_pred = torch.clamp(angle_pred, -1.0, 1.0)
        thrust_pred = torch.clamp(thrust_pred, 0.0, 1.0)
        shield_prob = torch.clamp(shield_prob, 1e-6, 1.0 - 1e-6)
        boost_prob = torch.clamp(boost_prob, 1e-6, 1.0 - 1e-6)
        
        # Calculate log probabilities for each component
        
        # Angle: continuous with small standard deviation
        angle_std = torch.full_like(angle_pred, self.angle_std_tensor.item())
        angle_dist = Normal(angle_pred, angle_std)
        angle_log_prob = angle_dist.log_prob(angle_actual)
        
        # Thrust: continuous with small standard deviation
        thrust_std = torch.full_like(thrust_pred, self.thrust_std_tensor.item())
        thrust_dist = Normal(thrust_pred, thrust_std)
        thrust_log_prob = thrust_dist.log_prob(thrust_actual)
        
        # Shield: Bernoulli-like probability
        shield_log_prob = shield_actual * torch.log(shield_prob + self.log_eps) + \
                        (1 - shield_actual) * torch.log(1 - shield_prob + self.log_eps)
        
        # Boost: Bernoulli-like probability
        boost_log_prob = boost_actual * torch.log(boost_prob + self.log_eps) + \
                        (1 - boost_actual) * torch.log(1 - boost_prob + self.log_eps)
        
        # Total log probability
        total_log_prob = angle_log_prob + thrust_log_prob + shield_log_prob + boost_log_prob
        
        # Check for NaN in final result
        if torch.isnan(total_log_prob).any():
            print(f"Warning: NaN detected in log_prob calculation, replacing with large negative value")
            total_log_prob = torch.where(torch.isnan(total_log_prob), 
                                    torch.full_like(total_log_prob, -10.0), 
                                    total_log_prob)
        
        return total_log_prob
    
    def collect_trajectories(self, num_steps: int = 128, visualization_panel=None):
        """Collect trajectory data by running the environment"""
        print(f"Starting trajectory collection for {num_steps} steps...")
        
        try:
            print("Resetting environment...")
            observations = self.env.reset()
            print(f"Environment reset complete. Got {len(observations)} observations")
            
            # Force synchronization across all devices at the start
            if self.multi_gpu and len(self.devices) > 1:
                for device_id in self.devices:
                    torch.cuda.synchronize(device_id)
            
            for step in range(num_steps):
                if step % 50 == 0:  # Progress indicator
                    print(f"Trajectory collection step {step}/{num_steps}")
                
                # Get actions from all pod networks
                actions = {}
                values = {}
                log_probs = {}
                
                try:
                    with torch.no_grad():
                        for pod_key, network in self.pod_networks.items():
                            # Get device for this pod's network
                            device = next(network.parameters()).device
                            
                            # Ensure observation is on the correct device
                            if pod_key in observations:
                                obs = observations[pod_key].to(device)
                                
                                # Use autocast for mixed precision if enabled
                                with autocast('cuda', enabled=self.use_mixed_precision):
                                    # Get action using the network's get_actions method
                                    action = network.get_actions(obs, deterministic=False)
                                    
                                    # Get value
                                    _, value, _ = network(obs)
                                    log_prob = self.get_action_log_prob(network, obs, action)
                                
                                # Store the results
                                actions[pod_key] = action
                                values[pod_key] = value
                                log_probs[pod_key] = log_prob
                            else:
                                print(f"Warning: {pod_key} not found in observations")
                
                except Exception as e:
                    print(f"Error getting actions at step {step}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                # Convert all actions to env's device before stepping
                try:
                    env_actions = {}
                    env_device = next(iter(observations.values())).device
                    for pod_key, action in actions.items():
                        env_actions[pod_key] = action.to(env_device)
                    
                    # Step the environment
                    if step % 100 == 0:
                        print(f"Stepping environment at step {step}")
                    
                    next_observations, rewards, dones, info = self.env.step(env_actions)
                    
                    if step % 100 == 0:
                        print(f"Environment step {step} complete")
                
                except Exception as e:
                    print(f"Error stepping environment at step {step}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise

                # Update visualization if provided
                if visualization_panel is not None and step % 5 == 0:
                    self.send_race_state_to_visualization(info, visualization_panel)   

                # Store trajectory data
                try:
                    for pod_key in self.pod_networks.keys():
                        if pod_key in observations:
                            device = next(self.pod_networks[pod_key].parameters()).device
                            
                            # Ensure all data is on the correct device before storing
                            self.storage[pod_key]['observations'].append(observations[pod_key].to(device))
                            self.storage[pod_key]['actions'].append(actions[pod_key].to(device))
                            self.storage[pod_key]['rewards'].append(rewards[pod_key].to(device))
                            self.storage[pod_key]['values'].append(values[pod_key].to(device))
                            self.storage[pod_key]['log_probs'].append(log_probs[pod_key].to(device))
                            
                            # Handle dones tensor
                            if isinstance(dones, dict):
                                pod_done = dones[pod_key].to(device)
                                if pod_done.dim() == 0:
                                    pod_done = pod_done.unsqueeze(0).unsqueeze(1)
                                elif pod_done.dim() == 1:
                                    pod_done = pod_done.unsqueeze(1)
                                self.storage[pod_key]['dones'].append(pod_done)
                            else:
                                dones_device = dones.to(device)
                                if dones_device.dim() == 1:
                                    self.storage[pod_key]['dones'].append(dones_device.unsqueeze(1))
                                elif dones_device.dim() == 2:
                                    self.storage[pod_key]['dones'].append(dones_device[:, 0:1])
                                else:
                                    sliced_dones = dones_device[:, 0, 0].unsqueeze(1)
                                    self.storage[pod_key]['dones'].append(sliced_dones)
                
                except Exception as e:
                    print(f"Error storing trajectory data at step {step}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                observations = next_observations
                
                # Explicit synchronization between steps for multi-GPU
                if self.multi_gpu and len(self.devices) > 1:
                    for device_id in self.devices:
                        torch.cuda.synchronize(device_id)
                
                # Reset environment if all episodes are done
                if dones.all():
                    print(f"All episodes done at step {step}, resetting environment")
                    observations = self.env.reset()
            
            print("Trajectory collection complete")
            
        except Exception as e:
            print(f"Error in collect_trajectories: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def compute_returns_and_advantages(self):
        """Compute returns and advantages for all pods"""
        print("Computing returns and advantages...")
        
        try:
            for pod_key, data in self.storage.items():
                print(f"Processing {pod_key}...")
                
                # Get network for this pod
                network = self.pod_networks[pod_key]
                device = next(network.parameters()).device
                
                # Convert lists to tensors and ensure they're on the correct device
                observations = torch.cat([obs.to(device) for obs in data['observations']])
                actions = torch.cat([act.to(device) for act in data['actions']])
                rewards = torch.cat([rew.to(device) for rew in data['rewards']])
                values = torch.cat([val.to(device) for val in data['values']])
                log_probs = torch.cat([lp.to(device) for lp in data['log_probs']])
                dones = torch.cat([done.to(device) for done in data['dones']])
                
                print(f"{pod_key}: obs={observations.shape}, rewards={rewards.shape}, values={values.shape}")
                
                # Ensure dones has the right shape [batch, 1]
                if dones.dim() > 2:
                    dones = dones[:, :, 0]
                
                # Compute returns and advantages using GAE
                returns = torch.zeros_like(rewards, device=device)
                advantages = torch.zeros_like(rewards, device=device)
                
                # Get final value estimate for the last observation
                with torch.no_grad():
                    last_obs = observations[-1].unsqueeze(0)
                    
                    with autocast('cuda', enabled=self.use_mixed_precision):
                        _, next_value, _ = network(last_obs)
                    
                    next_value = next_value.squeeze()
                    if next_value.dim() == 0:
                        next_value = next_value.unsqueeze(0)
                    
                    next_value = next_value.to(device)
                
                # Initialize for GAE calculation
                gae = torch.zeros(1, device=device)
                for t in reversed(range(len(rewards))):
                    if t == len(rewards) - 1:
                        next_non_terminal = 1.0 - dones[t].float()
                        next_value_t = next_value.to(device)
                    else:
                        next_non_terminal = 1.0 - dones[t].float()
                        next_value_t = values[t + 1].to(device)
                    
                    next_non_terminal = next_non_terminal.to(device)
                    rewards_t = rewards[t].to(device)
                    values_t = values[t].to(device)
                    
                    if next_value_t.shape != next_non_terminal.shape:
                        if next_value_t.numel() == 1:
                            next_value_t = next_value_t.view(-1).expand_as(next_non_terminal)
                            
                    delta = rewards_t + self.gamma * next_value_t * next_non_terminal - values_t
                    gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                    
                    returns[t] = gae + values_t
                    advantages[t] = gae
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Store processed data
                data['returns'] = returns
                data['advantages'] = advantages
                
                print(f"{pod_key}: returns computed, mean advantage = {advantages.mean().item():.4f}")
            
            print("Returns and advantages computation complete")
            
        except Exception as e:
            print(f"Error in compute_returns_and_advantages: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def update_policy(self):
        """Update policy using PPO"""
        print("Starting policy update...")
        
        try:
            total_policy_loss = 0
            total_value_loss = 0
            pod_count = 0
        
            # Process each pod
            for pod_key, data in self.storage.items():
                print(f"Updating policy for {pod_key}...")
                losses = self._update_pod(pod_key, data)
                total_policy_loss += losses.get('policy_loss', 0)
                total_value_loss += losses.get('value_loss', 0)
                pod_count += 1
                print(f"{pod_key}: policy_loss={losses.get('policy_loss', 0):.4f}, value_loss={losses.get('value_loss', 0):.4f}")

            # Calculate average losses
            avg_policy_loss = total_policy_loss / max(1, pod_count)
            avg_value_loss = total_value_loss / max(1, pod_count)
            
            print(f"Policy update complete: avg_policy_loss={avg_policy_loss:.4f}, avg_value_loss={avg_value_loss:.4f}")
            
            return avg_policy_loss, avg_value_loss
            
        except Exception as e:
            print(f"Error in update_policy: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _update_pod(self, pod_key, data):
        """Update a single pod - fully tensorized entropy calculation with NaN protection"""
        try:
            # Get network and optimizer for this pod
            network = self.pod_networks[pod_key]
            optimizer = self.optimizers[pod_key]
            
            # Get device for this pod's network
            device = next(network.parameters()).device
            
            # Get data and ensure it's on the correct device
            if isinstance(data['observations'], list):
                observations = torch.cat([obs.to(device) for obs in data['observations']])
                actions = torch.cat([act.to(device) for act in data['actions']])
                old_log_probs = torch.cat([lp.to(device) for lp in data['log_probs']])
                returns = data['returns'].to(device)
                advantages = data['advantages'].to(device)
            else:
                observations = data['observations'].to(device)
                actions = data['actions'].to(device)
                old_log_probs = data['old_log_probs'].to(device)
                returns = data['returns'].to(device)
                advantages = data['advantages'].to(device)

            # Check for NaN in input data
            if torch.isnan(observations).any():
                print(f"Warning: NaN detected in observations for {pod_key}")
                observations = torch.where(torch.isnan(observations), torch.zeros_like(observations), observations)
            
            if torch.isnan(actions).any():
                print(f"Warning: NaN detected in actions for {pod_key}")
                actions = torch.where(torch.isnan(actions), torch.zeros_like(actions), actions)

            # Track total losses
            total_policy_loss = 0
            total_value_loss = 0
            total_batches = 0
            
            # PPO update
            for epoch in range(self.ppo_epochs):
                # Generate random indices for mini-batches
                indices = torch.randperm(observations.size(0), device=device)
                
                # Process mini-batches
                for start_idx in range(0, observations.size(0), self.mini_batch_size):
                    end_idx = start_idx + self.mini_batch_size
                    mb_indices = indices[start_idx:end_idx]
                    
                    # Get mini-batch data
                    mb_observations = observations[mb_indices]
                    mb_actions = actions[mb_indices]
                    mb_old_log_probs = old_log_probs[mb_indices]
                    mb_returns = returns[mb_indices]
                    mb_advantages = advantages[mb_indices]
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass with mixed precision if enabled
                    with autocast('cuda', enabled=self.use_mixed_precision):
                        # Forward pass
                        mb_pred_actions, mb_values, _ = network(mb_observations)
                        
                        # Check for NaN in network output
                        if torch.isnan(mb_pred_actions).any() or torch.isnan(mb_values).any():
                            print(f"Warning: NaN detected in network output for {pod_key}, skipping this batch")
                            continue
                        
                        # Calculate new log probabilities
                        mb_new_log_probs = self.get_action_log_prob(network, mb_observations, mb_actions)
                        
                        # Check for NaN in log probabilities
                        if torch.isnan(mb_new_log_probs).any():
                            print(f"Warning: NaN detected in log probabilities for {pod_key}, skipping this batch")
                            continue
                        
                        # Calculate ratio and clipped ratio
                        ratio = torch.exp(mb_new_log_probs - mb_old_log_probs)
                        
                        # Clamp ratio to prevent extreme values
                        ratio = torch.clamp(ratio, 0.1, 10.0)
                        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                        
                        # Calculate surrogate losses
                        surrogate1 = ratio * mb_advantages
                        surrogate2 = clipped_ratio * mb_advantages
                        policy_loss = -torch.min(surrogate1, surrogate2).mean()
                        
                        # Calculate value loss
                        value_loss = F.mse_loss(mb_values, mb_returns)
                        
                        # Calculate entropy bonus for exploration - FULLY TENSORIZED
                        # For the 4-action format, calculate entropy for each component
                        
                        # Entropy for shield and boost probabilities
                        shield_probs = mb_pred_actions[:, 2:3]
                        boost_probs = mb_pred_actions[:, 3:4]
                        
                        # Clamp probabilities to avoid log(0)
                        shield_probs_clamped = torch.clamp(shield_probs, self.log_eps, 1.0 - self.log_eps)
                        boost_probs_clamped = torch.clamp(boost_probs, self.log_eps, 1.0 - self.log_eps)
                        
                        shield_entropy = -(shield_probs_clamped * torch.log(shield_probs_clamped) + 
                                        (1 - shield_probs_clamped) * torch.log(1 - shield_probs_clamped)).mean()
                        boost_entropy = -(boost_probs_clamped * torch.log(boost_probs_clamped) + 
                                        (1 - boost_probs_clamped) * torch.log(1 - boost_probs_clamped)).mean()
                        
                        # For angle and thrust (continuous actions), use constant entropy approximation
                        # This is valid since we're using fixed standard deviations
                        angle_entropy = self.angle_entropy_constant.to(device)
                        thrust_entropy = self.thrust_entropy_constant.to(device)
                        
                        # Total entropy
                        entropy = shield_entropy + boost_entropy + angle_entropy + thrust_entropy
                        
                        # Check for NaN in loss components
                        if torch.isnan(policy_loss) or torch.isnan(value_loss) or torch.isnan(entropy):
                            print(f"Warning: NaN detected in loss calculation for {pod_key}, skipping this batch")
                            print(f"  policy_loss: {policy_loss}, value_loss: {value_loss}, entropy: {entropy}")
                            continue
                        
                        # Total loss
                        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    
                    # Check for NaN in total loss
                    if torch.isnan(loss):
                        print(f"Warning: NaN detected in total loss for {pod_key}, skipping this batch")
                        continue
                    
                    # Backward pass with mixed precision if enabled
                    if self.use_mixed_precision:
                        self.scaler.scale(loss).backward()
                        
                        # Check for NaN in gradients before unscaling
                        has_nan_grad = False
                        for param in network.parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                has_nan_grad = True
                                break
                        
                        if has_nan_grad:
                            print(f"Warning: NaN detected in gradients for {pod_key}, skipping this batch")
                            optimizer.zero_grad()
                            continue
                        
                        self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        
                        # Check for NaN in gradients
                        has_nan_grad = False
                        for param in network.parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                has_nan_grad = True
                                break
                        
                        if has_nan_grad:
                            print(f"Warning: NaN detected in gradients for {pod_key}, skipping this batch")
                            optimizer.zero_grad()
                            continue
                        
                        nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)
                        optimizer.step()

                    # Track losses
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_batches += 1
            
            # Calculate average losses
            avg_policy_loss = total_policy_loss / max(1, total_batches)
            avg_value_loss = total_value_loss / max(1, total_batches)
            
            return {
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss
            }
            
        except Exception as e:
            print(f"Error in _update_pod for {pod_key}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def train(self, num_iterations: int = 1000, steps_per_iteration: int = 128, save_interval: int = 50, save_dir: str = 'models', visualization_panel=None):
        """Train the pod networks"""
        print(f"Starting training with {num_iterations} iterations...")
        
        try:
            # Create save directory if it doesn't exist
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # Create log file for training metrics
            log_file = save_path / "training_log.txt"
            
            # Initialize pod trails for visualization
            if visualization_panel is not None:
                self.pod_trails = {pod_key: [] for pod_key in self.pod_networks.keys()}
            
            # Training loop
            print("Starting training loop...")
            print("Training with the following parameters:")
            print(f"num_iterations: {num_iterations}")
            print(f"steps_per_iteration: {steps_per_iteration}")
            print(f"save_interval: {save_interval}")
            print(f"save_dir: {save_dir}")
            
            for iteration in range(1, num_iterations + 1):
                print(f"\n=== Starting iteration {iteration}/{num_iterations} ===")
                start_time = time.time()
                
                try:
                    # Check for NaN in network parameters before training
                    self._check_and_fix_network_parameters()
                    
                    # Collect trajectories
                    print("Collecting trajectories...")
                    self.collect_trajectories(steps_per_iteration, visualization_panel)
                    print("Trajectory collection complete")
                    
                    # Compute returns and advantages
                    print("Computing returns and advantages...")
                    self.compute_returns_and_advantages()
                    print("Returns and advantages computation complete")
                    
                    # Update policy
                    print("Updating policy...")
                    policy_loss, value_loss = self.update_policy()
                    print("Policy update complete")

                    # Check for NaN in network parameters after training
                    self._check_and_fix_network_parameters()

                    # Calculate role-specific average rewards
                    runner_reward = 0
                    blocker_reward = 0
                    runner_count = 0
                    blocker_count = 0
                    
                    for pod_key in self.pod_networks.keys():
                        if 'rewards' in self.storage[pod_key]:
                            rewards = torch.cat([r for r in self.storage[pod_key]['rewards']])
                            avg_pod_reward = rewards.mean().item()
                            
                            # Determine if this is a runner or blocker
                            if 'pod0' in pod_key:  # Runner
                                runner_reward += avg_pod_reward
                                runner_count += 1
                            elif 'pod1' in pod_key:  # Blocker
                                blocker_reward += avg_pod_reward
                                blocker_count += 1
                    
                    # Calculate averages
                    avg_runner_reward = runner_reward / max(1, runner_count)
                    avg_blocker_reward = blocker_reward / max(1, blocker_count)
                    avg_total_reward = (runner_reward + blocker_reward) / max(1, runner_count + blocker_count)
                    
                    # Reset storage for next iteration
                    self.reset_storage()
                    
                    # Log progress with role-specific information
                    elapsed_time = time.time() - start_time
                    print(f"Iteration {iteration}/{num_iterations} completed in {elapsed_time:.2f}s")
                    print(f"  Total reward: {avg_total_reward:.4f}")
                    print(f"  Runner reward: {avg_runner_reward:.4f}")
                    print(f"  Blocker reward: {avg_blocker_reward:.4f}")
                    print(f"  Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}")

                    # Log metrics to file with role breakdown
                    with open(log_file, 'a') as f:
                        f.write(f"iteration={iteration},total_reward={avg_total_reward:.4f},"
                            f"runner_reward={avg_runner_reward:.4f},blocker_reward={avg_blocker_reward:.4f},"
                            f"policy_loss={policy_loss:.4f},value_loss={value_loss:.4f},time={elapsed_time:.2f}\n")
                    
                    # Send enhanced metrics to visualization panel
                    if visualization_panel is not None:
                        metrics = {
                            'iteration': iteration,
                            'total_reward': avg_total_reward,
                            'runner_reward': avg_runner_reward,
                            'blocker_reward': avg_blocker_reward,
                            'policy_loss': policy_loss,
                            'value_loss': value_loss
                        }
                        self.send_metrics_to_visualization(iteration, metrics, visualization_panel)       

                    # Save models periodically
                    if iteration % save_interval == 0:
                        print(f"Saving models at iteration {iteration}...")
                        for pod_key, network in self.pod_networks.items():
                            torch.save(network.state_dict(), save_path / f"{pod_key}_iter{iteration}.pt")
                            # Also save as the latest model
                            torch.save(network.state_dict(), save_path / f"{pod_key}_latest.pt")
                        print(f"Models saved at iteration {iteration}")
                    
                    # Add explicit synchronization to ensure all GPUs are active
                    if self.multi_gpu and len(self.devices) > 1:
                        for device_id in self.devices:
                            torch.cuda.synchronize(device_id)
                            
                except Exception as e:
                    print(f"Error in training iteration {iteration}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next iteration instead of crashing
                    continue
            
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Error in train method: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _check_and_fix_network_parameters(self):
        """Check for NaN in network parameters and reset if necessary"""
        for pod_key, network in self.pod_networks.items():
            has_nan = False
            for name, param in network.named_parameters():
                if torch.isnan(param).any():
                    print(f"Warning: NaN detected in {pod_key} parameter {name}")
                    has_nan = True
            
            if has_nan:
                print(f"Resetting network parameters for {pod_key} due to NaN values")
                # Reinitialize the network
                device = next(network.parameters()).device
                network.apply(self._init_weights)
            
    def _init_weights(self, module):
        """Initialize weights with Xavier/Glorot initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def save_models(self, save_dir: str = 'models'):
        """Save all pod networks"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for pod_key, network in self.pod_networks.items():
            torch.save(network.state_dict(), save_path / f"{pod_key}.pt")
        
        print(f"All models saved to {save_dir}")
    
    def load_models(self, load_dir: str = 'models'):
        """Load all pod networks"""
        load_path = Path(load_dir)
        
        for pod_key, network in self.pod_networks.items():
            model_path = load_path / f"{pod_key}.pt"
            if model_path.exists():
                network.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded model for {pod_key}")
            else:
                print(f"No model found for {pod_key}, using random initialization")
    
    def get_pod_network(self, pod_key: str):
        """Get the network for the specified pod"""
        return self.pod_networks[pod_key]

    def send_metrics_to_visualization(self, iteration, metrics, visualization_panel=None):
        """Send metrics to visualization panel if available"""
        if visualization_panel is None:
            return
            
        # Only send metrics occasionally to avoid performance impact
        if iteration % 5 != 0:  # Only send every 5 iterations
            return
            
        # Create a metrics dictionary
        metric_dict = {
            'iteration': iteration,
            'reward': metrics.get('total_reward', 0),
            'runner_reward': metrics.get('runner_reward', 0),
            'blocker_reward': metrics.get('blocker_reward', 0),
            'policy_loss': metrics.get('policy_loss', 0),
            'value_loss': metrics.get('value_loss', 0)
        }
        
        # Send to visualization panel
        try:
            visualization_panel.add_metric(metric_dict)
        except Exception as e:
            # Don't let visualization errors affect training
            print(f"Error sending metrics to visualization: {e}")
            
    def send_race_state_to_visualization(self, info, visualization_panel=None):
        """Send race state to visualization panel if available"""
        if visualization_panel is None:
            return
            
        # Only send race state occasionally to avoid performance impact
        if not hasattr(self, '_vis_counter'):
            self._vis_counter = 0
        
        self._vis_counter += 1
        if self._vis_counter % 5 != 0:  # Only send every 5 steps
            return
        
        # Create a simplified race state for visualization
        vis_race_state = {
            'checkpoints': self.env.get_checkpoints() if hasattr(self.env, 'get_checkpoints') else [],
            'pods': [],
            'worker_id': 'main'  # Use 'main' as the worker ID for the standard trainer
        }
        
        # Add pod information
        for pod_key in self.pod_networks.keys():
            if hasattr(info, 'get') and 'positions' in info and 'angles' in info:
                if pod_key in info['positions'] and pod_key in info['angles']:
                    position = info['positions'][pod_key][0].tolist()
                    angle = info['angles'][pod_key][0].item()
                    
                    # Get trail if available
                    trail = []
                    if hasattr(self, 'pod_trails') and pod_key in self.pod_trails:
                        trail = self.pod_trails[pod_key]
                    
                    pod_info = {
                        'position': position,
                        'angle': angle,
                        'trail': trail
                    }
                    
                    vis_race_state['pods'].append(pod_info)
        
        # Send to visualization panel
        try:
            visualization_panel.add_metric({'race_state': vis_race_state})
        except Exception as e:
            # Don't let visualization errors affect training
            print(f"Error sending race state to visualization: {e}")
