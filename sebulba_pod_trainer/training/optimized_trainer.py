import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from ..training.trainer import PPOTrainer

class OptimizedPPOTrainer(PPOTrainer):
    """Optimized version of PPOTrainer with better performance"""
    
    def collect_trajectories(self, num_steps: int = 128, visualization_panel=None):
        """Optimized trajectory collection with fewer device transfers"""
        # Reset storage
        self.reset_storage()
        
        # Get initial observations
        observations = self.env.reset()

        # Initialize pod trails for visualization
        if visualization_panel is not None and not hasattr(self, 'pod_trails'):
            self.pod_trails = {pod_key: [] for pod_key in self.pod_networks.keys()}
        
        # Pre-allocate tensors for storage to avoid repeated allocations
        for pod_key in self.pod_networks.keys():
            device = next(self.pod_networks[pod_key].parameters()).device
            batch_size = observations[pod_key].shape[0]
            obs_dim = observations[pod_key].shape[1]
            
            # Pre-allocate storage with correct sizes
            self.storage[pod_key]['observations'] = []
            self.storage[pod_key]['actions'] = []
            self.storage[pod_key]['rewards'] = []
            self.storage[pod_key]['values'] = []
            self.storage[pod_key]['log_probs'] = []
            self.storage[pod_key]['dones'] = []
        
        # Collect trajectories
        for step in range(num_steps):
            # Get actions from all pod networks
            actions = {}
            values = {}
            log_probs = {}
            
            with torch.no_grad():
                for pod_key, network in self.pod_networks.items():
                    device = next(network.parameters()).device
                    obs = observations[pod_key].to(device)
                    
                    with autocast('cuda', enabled=self.use_mixed_precision):
                        action = network.get_actions(obs, deterministic=False)
                        _, value, _ = network(obs)
                        log_prob = self.get_action_log_prob(network, obs, action)
                    
                    actions[pod_key] = action
                    values[pod_key] = value
                    log_probs[pod_key] = log_prob
            
            # Step the environment
            next_observations, rewards, dones, _ = self.env.step(actions)

            # Update pod trails for visualization
            if visualization_panel is not None and hasattr(self, 'pod_trails'):
                # Extract positions from observations or info
                # Most racing envs include position in the observation
                for pod_key in self.pod_networks.keys():
                    if pod_key in observations:
                        # Extract position from observation - assuming x, y are first 2 values
                        pos = observations[pod_key][0, 0:2].tolist()
                        if not hasattr(self, 'pod_trails'):
                            self.pod_trails = {pk: [] for pk in self.pod_networks.keys()}
                        self.pod_trails[pod_key].append(pos)
                        # Keep trail length manageable
                        if len(self.pod_trails[pod_key]) > 50:
                            self.pod_trails[pod_key] = self.pod_trails[pod_key][-50:]
            
            # Send race state to visualization occasionally
            if visualization_panel is not None and step % 20 == 0:
                # Create race state for visualization
                race_state = {
                    'checkpoints': self.env.get_checkpoints() if hasattr(self.env, 'get_checkpoints') else [],
                    'pods': []
                }
                
                # Add pod information from observations
                for pod_key in self.pod_networks.keys():
                    if pod_key in observations:
                        # Extract position and angle from observation
                        # Assuming first 2 values are position, 3rd value is angle
                        obs = observations[pod_key][0]
                        position = obs[0:2].tolist()
                        angle = float(obs[2]) if len(obs) > 2 else 0.0
                        
                        pod_info = {
                            'position': position,
                            'angle': angle,
                            'trail': self.pod_trails.get(pod_key, []) if hasattr(self, 'pod_trails') else []
                        }
                        
                        race_state['pods'].append(pod_info)
            
            # Send to visualization panel
            visualization_panel.add_metric({'race_state': race_state})           
            # Store trajectory data
            for pod_key in self.pod_networks.keys():
                device = next(self.pod_networks[pod_key].parameters()).device
                
                # Store data in lists (compatible with parent class)
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
                    else:
                        self.storage[pod_key]['dones'].append(dones_device[:, 0:1])
            
            observations = next_observations
            
            # Reset environment if all episodes are done
            if isinstance(dones, torch.Tensor) and dones.all():
                observations = self.env.reset()
    
    def compute_returns_and_advantages(self):
        """Optimized computation of returns and advantages"""
        for pod_key, data in self.storage.items():
            network = self.pod_networks[pod_key]
            device = next(network.parameters()).device
            
            # Convert lists to tensors first
            observations = torch.cat([obs.to(device) for obs in data['observations']])
            rewards = torch.cat([rew.to(device) for rew in data['rewards']])
            values = torch.cat([val.to(device) for val in data['values']])
            dones = torch.cat([done.to(device) for done in data['dones']])
            
            # Get final value estimate
            with torch.no_grad():
                last_obs = observations[-1].unsqueeze(0)
                with autocast('cuda', enabled=self.use_mixed_precision):
                    _, next_value, _ = network(last_obs)
            
            # Compute returns and advantages using vectorized operations
            returns = torch.zeros_like(rewards, device=device)
            advantages = torch.zeros_like(rewards, device=device)
            
            # Initialize for GAE calculation
            gae = torch.zeros((1,), device=device)
            next_value = next_value.to(device)
            
            # Vectorized GAE calculation
            for t in reversed(range(rewards.size(0))):
                if t == rewards.size(0) - 1:
                    next_non_terminal = 1.0 - dones[t].float()
                    next_value_t = next_value
                else:
                    next_non_terminal = 1.0 - dones[t].float()
                    next_value_t = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                returns[t] = gae + values[t]
                advantages[t] = gae
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Store processed data
            data['returns'] = returns
            data['advantages'] = advantages
