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
from ..environment.race_env import RaceEnvironment

class PPOTrainer:
    """
    Proximal Policy Optimization trainer for pod racing
    """
    def __init__(self,
                env: RaceEnvironment,
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
        self.network_config = network_config

        # Use network_config if provided, otherwise use defaults
        if self.network_config is None:
            self.network_config = {
                    'observation_dim': 56,
                    'hidden_layers': [
                        {'type': 'Linear+ReLU', 'size': 24},
                        {'type': 'Linear+ReLU', 'size': 16}
                    ],
                    'policy_hidden_size': 12,
                    'value_hidden_size': 12,
                    'special_hidden_size': 12
                }
            
        # Initialize mixed precision tools if enabled
        if self.use_mixed_precision:
            try:
                # Fix the deprecated GradScaler initialization
                self.scaler = GradScaler()  # Remove the 'cuda' parameter
                print("Mixed precision training enabled with standard GradScaler")
            except Exception as e:
                print(f"Error initializing GradScaler: {e}")
                # Try alternative initialization if the above fails
                try:
                    self.scaler = GradScaler('cuda')
                    print("Mixed precision training enabled with 'cuda' parameter")
                except Exception as e2:
                    print(f"Error initializing GradScaler with 'cuda' parameter: {e2}")
                    self.use_mixed_precision = False
                    print("Mixed precision disabled due to errors")
        
        # Force CUDA to initialize on all selected GPUs
        if self.multi_gpu and len(self.devices) > 1:
            print(f"Initializing {len(self.devices)} GPUs: {self.devices}")
            for device_id in self.devices:
                try:
                    # Allocate a small tensor on each GPU to ensure they're visible in nvidia-smi
                    dummy_tensor = torch.zeros(1, device=f"cuda:{device_id}")
                    print(f"Successfully initialized GPU {device_id}")
                except Exception as e:
                    print(f"Error initializing GPU {device_id}: {e}")
        
        # Create networks for each pod
        self.pod_networks = {}
        self.optimizers = {}
        
        # Distribute pod networks across available GPUs if multi-GPU is enabled
        if self.multi_gpu and len(self.devices) > 1:
            # Create a mapping to distribute networks evenly across GPUs
            pod_device_map = {}
            player_pods = [f"player0_pod{i}" for i in range(2)] + [f"player1_pod{i}" for i in range(2)]
            
            for i, pod_key in enumerate(player_pods):
                # Distribute pods across available devices in a round-robin fashion
                device_idx = i % len(self.devices)
                gpu_id = self.devices[device_idx]  # Get the actual GPU ID from the devices list
                pod_device_map[pod_key] = gpu_id
                
            # Initialize networks for player pods with specific GPU assignments
            for i in range(2):
                for player_idx in range(2):
                    pod_key = f"player{player_idx}_pod{i}"
                    gpu_id = pod_device_map[pod_key]
                    # Explicitly create the network on the specific GPU
                    network = PodNetwork(
                        observation_dim=self.network_config.get('observation_dim', 56),
                        hidden_layers=self.network_config.get('hidden_layers', []),
                        policy_hidden_size=self.network_config.get('policy_hidden_size', 12),
                        value_hidden_size=self.network_config.get('value_hidden_size', 12),
                        special_hidden_size=self.network_config.get('special_hidden_size', 12)
                    ).to(f"cuda:{gpu_id}")
                    
                    self.pod_networks[pod_key] = network
                    self.optimizers[pod_key] = optim.Adam(
                        self.pod_networks[pod_key].parameters(),
                        lr=learning_rate
                    )
        else:
            # Original initialization code for single GPU or CPU
            # Initialize networks for player 1's pods
            for i in range(2):
                pod_key = f"player0_pod{i}"
                network = PodNetwork(
                    observation_dim=network_config.get('observation_dim', 56),
                    hidden_layers=network_config.get('hidden_layers', []),
                    policy_hidden_size=network_config.get('policy_hidden_size', 12),
                    value_hidden_size=network_config.get('value_hidden_size', 12),
                    special_hidden_size=network_config.get('special_hidden_size', 12)
                ).to(device)
                
                # Wrap with DataParallel if using multiple GPUs
                if self.multi_gpu and len(self.devices) > 1:
                    network = nn.DataParallel(network, device_ids=self.devices)
                    
                self.pod_networks[pod_key] = network
                self.optimizers[pod_key] = optim.Adam(
                    self.pod_networks[pod_key].parameters(),
                    lr=learning_rate
                )
            
            # For player 2, we can either train adversarial pods or use fixed strategy
            # Here we'll train adversarial pods
            for i in range(2):
                pod_key = f"player1_pod{i}"
                network = PodNetwork(
                    observation_dim=network_config.get('observation_dim', 56),
                    hidden_layers=network_config.get('hidden_layers', []),
                    policy_hidden_size=network_config.get('policy_hidden_size', 12),
                    value_hidden_size=network_config.get('value_hidden_size', 12),
                    special_hidden_size=network_config.get('special_hidden_size', 12)
                ).to(device)
                
                # Wrap with DataParallel if using multiple GPUs
                if self.multi_gpu and len(self.devices) > 1:
                    network = nn.DataParallel(network, device_ids=self.devices)
                    
                self.pod_networks[pod_key] = network
                self.optimizers[pod_key] = optim.Adam(
                    self.pod_networks[pod_key].parameters(),
                    lr=learning_rate
                )
        
        # Storage for trajectory data
        self.reset_storage()
    
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
        """Calculate log probability of an action with mixed precision safety"""
        # Handle DataParallel models by accessing the module directly if needed
        if isinstance(network, nn.DataParallel):
            module = network.module
        else:
            module = network

        # Use autocast for mixed precision if enabled
        with autocast('cuda', enabled=self.use_mixed_precision):
            raw_actions, _, special_probs = module(observation)
        
        # IMPORTANT: Convert to float32 for probability calculations to avoid NaN values
        # This is crucial for numerical stability when using mixed precision
        angle_mean = raw_actions[:, 0:1].float()  # Explicitly convert to float32
        angle_std = torch.ones_like(angle_mean) * 0.1
        
        # Ensure action is also float32 for distribution calculations
        action_float = action.float()
        
        # Calculate log prob for angle (continuous)
        angle_dist = Normal(angle_mean, angle_std)
        angle_log_prob = angle_dist.log_prob(action_float[:, 0:1])
        
        # Calculate log prob for thrust and special actions
        thrust_value = action_float[:, 1:2]
        
        # Shield action (thrust < -0.9)
        shield_mask = thrust_value < -0.9
        special_probs_float = special_probs.float()  # Convert to float32 for stability
        shield_log_prob = torch.log(special_probs_float[:, 0:1] + 1e-8) * shield_mask + \
                        torch.log(1 - special_probs_float[:, 0:1] + 1e-8) * (~shield_mask)
        
        # Boost action (thrust > 0.9)
        boost_mask = thrust_value > 0.9
        boost_log_prob = torch.log(special_probs_float[:, 1:2] + 1e-8) * boost_mask + \
                        torch.log(1 - special_probs_float[:, 1:2] + 1e-8) * (~boost_mask)
        
        # Normal thrust action
        normal_thrust_mask = ~(shield_mask | boost_mask)
        thrust_mean = raw_actions[:, 1:2].float()  # Convert to float32
        thrust_std = torch.ones_like(thrust_mean) * 0.1
        thrust_dist = Normal(thrust_mean, thrust_std)
        normal_thrust_log_prob = thrust_dist.log_prob(thrust_value) * normal_thrust_mask
        
        # Combine log probs
        thrust_log_prob = shield_log_prob + boost_log_prob + normal_thrust_log_prob
        
        # Total log prob
        log_prob = angle_log_prob + thrust_log_prob
        
        return log_prob
    
    def collect_trajectories(self, num_steps: int = 128, visualization_panel=None):
        """Collect trajectory data by running the environment with device safety"""
        observations = self.env.reset()
        
        # Force synchronization across all devices at the start
        if self.multi_gpu and len(self.devices) > 1:
            for device_id in self.devices:
                torch.cuda.synchronize(device_id)
        
        for step in range(num_steps):
            # Get actions from all pod networks
            actions = {}
            values = {}
            log_probs = {}
            
            with torch.no_grad():
                for pod_key, network in self.pod_networks.items():
                    # Get device for this pod's network
                    device = next(network.parameters()).device
                    
                    # Ensure observation is on the correct device
                    if pod_key in observations:
                        obs = observations[pod_key].to(device)
                        
                        # Use autocast for mixed precision if enabled
                        with autocast('cuda', enabled=self.use_mixed_precision):
                            action = network.get_actions(obs, deterministic=False)
                            
                            # Get value and log probability
                            _, value, _ = network(obs)
                            log_prob = self.get_action_log_prob(network, obs, action)
                        
                        # Store the results (keep on network's device)
                        actions[pod_key] = action
                        values[pod_key] = value
                        log_probs[pod_key] = log_prob
            
            # Convert all actions to env's device before stepping
            env_actions = {}
            env_device = next(iter(observations.values())).device
            for pod_key, action in actions.items():
                env_actions[pod_key] = action.to(env_device)
            
            # Step the environment
            next_observations, rewards, dones, _ = self.env.step(env_actions)

            # Update pod trails for visualization
            if visualization_panel is not None and hasattr(self, 'pod_trails'):
                for pod_key in self.pod_networks.keys():
                    if pod_key in info['positions']:
                        pos = info['positions'][pod_key][0].tolist()
                        self.pod_trails[pod_key].append(pos)
                        # Keep trail length manageable
                        if len(self.pod_trails[pod_key]) > 50:
                            self.pod_trails[pod_key] = self.pod_trails[pod_key][-50:]
            
            # Send race state to visualization occasionally
            if visualization_panel is not None and step % 5 == 0:
                self.send_race_state_to_visualization(info, visualization_panel)   

            # Store trajectory data
            for pod_key in self.pod_networks.keys():
                if pod_key in observations:
                    device = next(self.pod_networks[pod_key].parameters()).device
                    
                    # Ensure all data is on the correct device before storing
                    self.storage[pod_key]['observations'].append(observations[pod_key].to(device))
                    self.storage[pod_key]['actions'].append(actions[pod_key].to(device))
                    self.storage[pod_key]['rewards'].append(rewards[pod_key].to(device))
                    self.storage[pod_key]['values'].append(values[pod_key].to(device))
                    self.storage[pod_key]['log_probs'].append(log_probs[pod_key].to(device))
                    
                    # Handle dones tensor carefully
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
            
            observations = next_observations
            
            # Explicit synchronization between steps for multi-GPU
            if self.multi_gpu and len(self.devices) > 1:
                for device_id in self.devices:
                    torch.cuda.synchronize(device_id)
            
            # Reset environment if all episodes are done
            if dones.all():
                observations = self.env.reset()

    def compute_returns_and_advantages(self):
        """Compute returns and advantages for all pods with device safety"""
        for pod_key, data in self.storage.items():
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
            
            # Ensure dones has the right shape [batch, 1]
            if dones.dim() > 2:
                dones = dones[:, :, 0]
            
            # Compute returns and advantages using GAE
            returns = torch.zeros_like(rewards, device=device)
            advantages = torch.zeros_like(rewards, device=device)
            
            # Get final value estimate for the last observation
            with torch.no_grad():
                last_obs = observations[-1].unsqueeze(0)
                
                # Use autocast for mixed precision if enabled
                with autocast('cuda', enabled=self.use_mixed_precision):
                    _, next_value, _ = network(last_obs)
                
                next_value = next_value.squeeze()
                if next_value.dim() == 0:  # scalar
                    next_value = next_value.unsqueeze(0)
                
                # Explicitly move to the correct device
                next_value = next_value.to(device)
            
            # Initialize for GAE calculation
            gae = torch.zeros(1, device=device)
            for t in reversed(range(len(rewards))):
                # For the last step, use the estimated next value
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t].float()
                    next_value_t = next_value.to(device)  # Ensure next_value is on the correct device
                else:
                    next_non_terminal = 1.0 - dones[t].float()
                    next_value_t = values[t + 1].to(device)  # Ensure the value is on the correct device
                
                # Explicitly ensure all tensors are on the same device
                next_non_terminal = next_non_terminal.to(device)
                rewards_t = rewards[t].to(device)
                values_t = values[t].to(device)
                
                # Make sure the shapes match for operations
                if next_value_t.shape != next_non_terminal.shape:
                    if next_value_t.numel() == 1:
                        next_value_t = next_value_t.view(-1).expand_as(next_non_terminal)
                        
                # Calculate delta and GAE with explicit device management
                delta = rewards_t + self.gamma * next_value_t * next_non_terminal - values_t
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                
                # Store returns and advantages
                returns[t] = gae + values_t
                advantages[t] = gae
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Store processed data
            data['returns'] = returns
            data['advantages'] = advantages

    def update_policy(self):
        """Update policy using PPO with improved multi-GPU parallelism"""
        total_policy_loss = 0
        total_value_loss = 0
        pod_count = 0
    
        # If using multiple GPUs, we can parallelize updates across pods
        if self.multi_gpu and len(self.devices) > 1:
            # Group pod networks by device to optimize GPU utilization
            device_pod_groups = {}
            for pod_key, network in self.pod_networks.items():
                # Get the device of this network's parameters
                device = next(network.parameters()).device
                
                # Add this pod to the list of pods on this device
                if device not in device_pod_groups:
                    device_pod_groups[device] = []
                device_pod_groups[device].append(pod_key)
            
            # Process each device group in parallel (through Python threads)
            threads = []
            results = {}
            for device, pod_keys in device_pod_groups.items():
                # Create a thread to update all pods on this device
                thread = threading.Thread(
                    target=self._update_pods_on_device,
                    args=(pod_keys, results)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to finish
            for thread in threads:
                thread.join()

            # Collect loss values
            for pod_key, result in results.items():
                total_policy_loss += result.get('policy_loss', 0)
                total_value_loss += result.get('value_loss', 0)
                pod_count += 1
        else:
            # Original single-threaded update
            for pod_key, data in self.storage.items():
                losses = self._update_pod(pod_key, data)
                total_policy_loss += losses.get('policy_loss', 0)
                total_value_loss += losses.get('value_loss', 0)
                pod_count += 1

        # Calculate average losses
        avg_policy_loss = total_policy_loss / max(1, pod_count)
        avg_value_loss = total_value_loss / max(1, pod_count)
        
        return avg_policy_loss, avg_value_loss
    
    def _update_pods_on_device(self, pod_keys, results):
        """Update all pods assigned to a specific device"""
        for pod_key in pod_keys:
            data = self.storage[pod_key]
            
            # Get network and optimizer for this pod
            network = self.pod_networks[pod_key]
            optimizer = self.optimizers[pod_key]
            
            # Make sure we're operating on the correct device
            device = next(network.parameters()).device
            
            # Process data on the correct device
            observations = torch.cat([obs.to(device) for obs in data['observations']])
            actions = torch.cat([act.to(device) for act in data['actions']])
            old_log_probs = torch.cat([lp.to(device) for lp in data['log_probs']])
            returns = data['returns'].to(device)
            advantages = data['advantages'].to(device)
            
            # Proceed with PPO updates on this device
            losses = self._update_pod(pod_key, {
                'observations': observations,
                'actions': actions,
                'old_log_probs': old_log_probs,
                'returns': returns,
                'advantages': advantages
            })
            
            # Store results
            results[pod_key] = losses
    
    def _update_pod(self, pod_key, data):
        """Update a single pod with proper device management"""
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

        # Track total losses
        total_policy_loss = 0
        total_value_loss = 0
        total_batches = 0
        
        # PPO update
        for _ in range(self.ppo_epochs):
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
                    mb_raw_actions, mb_values, mb_special_probs = network(mb_observations)
                    
                    # Calculate new log probabilities
                    mb_new_log_probs = self.get_action_log_prob(network, mb_observations, mb_actions)
                    
                    # Calculate ratio and clipped ratio
                    ratio = torch.exp(mb_new_log_probs - mb_old_log_probs)
                    clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    
                    # Calculate surrogate losses
                    surrogate1 = ratio * mb_advantages
                    surrogate2 = clipped_ratio * mb_advantages
                    policy_loss = -torch.min(surrogate1, surrogate2).mean()
                    
                    # Calculate value loss
                    value_loss = F.mse_loss(mb_values, mb_returns)
                    
                    # Calculate entropy bonus (to encourage exploration)
                    angle_mean = mb_raw_actions[:, 0:1]
                    angle_std = torch.ones_like(angle_mean) * 0.1
                    angle_dist = Normal(angle_mean, angle_std)
                    angle_entropy = angle_dist.entropy().mean()
                    
                    # Entropy from special actions (shield and boost)
                    special_entropy = -(mb_special_probs * torch.log(mb_special_probs + 1e-8) + 
                                    (1 - mb_special_probs) * torch.log(1 - mb_special_probs + 1e-8)).mean()
                    
                    entropy = angle_entropy + special_entropy
                    
                    # Total loss
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass with mixed precision if enabled
                if self.use_mixed_precision:
                    # Use the scaler to handle the backward pass and gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Standard backward pass
                    loss.backward()
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
    
    def train(self, num_iterations: int = 1000, steps_per_iteration: int = 128, save_interval: int = 50, save_dir: str = 'models', visualization_panel=None):
        """Train the pod networks"""
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
            start_time = time.time()
            
            # Collect trajectories
            self.collect_trajectories(steps_per_iteration, visualization_panel)
            
            # Compute returns and advantages
            self.compute_returns_and_advantages()
            
            # Update policy
            policy_loss, value_loss = self.update_policy()

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
                for pod_key, network in self.pod_networks.items():
                    torch.save(network.state_dict(), save_path / f"{pod_key}_iter{iteration}.pt")
                    # Also save as the latest model
                    torch.save(network.state_dict(), save_path / f"{pod_key}_latest.pt")
                print(f"Models saved at iteration {iteration}")
            
            # Add explicit synchronization to ensure all GPUs are active
            if self.multi_gpu and len(self.devices) > 1:
                print(f"Training using multiple GPUs: {self.devices}")
                for device_id in self.devices:
                    torch.cuda.synchronize(device_id)
    
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
    
    def _ensure_same_device(self, tensor1, tensor2):
        """Ensure both tensors are on the same device by moving tensor2 to tensor1's device if needed"""
        if tensor1.device != tensor2.device:
            return tensor2.to(tensor1.device)
        return tensor2

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
            'reward': metrics.get('reward', 0),
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
        # This is controlled by a counter to reduce frequency
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
