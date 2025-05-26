from pathlib import Path
import queue
import threading
import time
import traceback
import torch
import torch.multiprocessing as mp
from typing import List, Dict
import numpy as np
from copy import deepcopy
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import random

from ..environment.race_env import RaceEnvironment
from ..environment.optimized_race_env import OptimizedRaceEnvironment
from .trainer import PPOTrainer
from .optimized_trainer import OptimizedPPOTrainer

class SharedExperienceBuffer:
    """Shared experience buffer across workers"""
    def __init__(self, manager, buffer_size=10000):
        self.buffer = manager.list()
        self.buffer_size = buffer_size
        self.lock = manager.Lock()
    
    def add_experience(self, experience):
        with self.lock:
            self.buffer.append(experience)
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)  # Remove oldest experience
    
    def sample_experiences(self, batch_size=64):
        with self.lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)
            
            return random.sample(list(self.buffer), batch_size)

class ParallelPPOTrainer:
    """
    Parallelized PPO Trainer that runs multiple environments across available GPUs
    with model synchronization for improved convergence
    """
    def __init__(self, env_config, trainer_args, envs_per_gpu=8):
        # Set the start method for multiprocessing to 'spawn'
        # This is required for CUDA to work with multiprocessing
        try:
            mp.set_start_method('spawn', force=True)
            print("Set multiprocessing start method to 'spawn'")
        except RuntimeError:
            # If it's already set, this will raise a RuntimeError
            print("Multiprocessing start method already set")
    
        # Make deep copies of input dictionaries to avoid reference issues
        self.env_config = deepcopy(env_config)
        self.trainer_args = deepcopy(trainer_args)
        self.envs_per_gpu = envs_per_gpu
        self.devices = self.trainer_args.get('devices', [0])
        self.num_gpus = len(self.devices)
        self.processes = []
        self.use_mixed_precision = self.trainer_args.get('use_mixed_precision', True)

        # Add synchronization parameters
        self.sync_interval = self.trainer_args.get('sync_interval', 5)  # Sync every N iterations
        self.use_parameter_server = self.trainer_args.get('use_parameter_server', True)
        self.shared_experience_buffer = self.trainer_args.get('shared_experience_buffer', False)
        self.gradient_aggregation = self.trainer_args.get('gradient_aggregation', True)
        self.adaptive_lr = self.trainer_args.get('adaptive_lr', True)

        # Add timeout penalty configuration
        if 'timeout_penalty_weight' not in self.env_config:
            self.env_config['timeout_penalty_weight'] = 0.01  # Default timeout penalty weight
        
        print(f"ParallelPPOTrainer initialized with args: {self.trainer_args}")
        print(f"Training iterations: {self.trainer_args.get('num_iterations', 'NOT FOUND')}")
        print(f"Steps per iteration: {self.trainer_args.get('steps_per_iteration', 'NOT FOUND')}")
        print(f"Using devices: {self.devices}")
        print(f"Number of GPUs: {self.num_gpus}")
        print(f"Environments per GPU: {self.envs_per_gpu}")
        print(f"Sync interval: {self.sync_interval}")
        print(f"Parameter server: {self.use_parameter_server}")
        print(f"Shared experience: {self.shared_experience_buffer}")
        print(f"Gradient aggregation: {self.gradient_aggregation}")
        print(f"Timeout penalty weight: {self.env_config['timeout_penalty_weight']}")
        
        # Calculate total number of environments
        self.total_envs = self.num_gpus * self.envs_per_gpu
        print(f"Total environments: {self.total_envs}")
    
    @staticmethod
    def _worker(gpu_id, env_indices, env_config, trainer_args, total_envs, envs_per_gpu, 
                use_mixed_precision, shared_results, vis_queue=None, shared_models=None, 
                shared_experience=None, shared_metrics=None, sync_interval=5, 
                gradient_aggregation=True, adaptive_lr=True, num_gpus=1):
        """Enhanced worker process with model synchronization and experience sharing"""
        try:
            print(f"Worker process for GPU {gpu_id} starting with timeout penalty prevention...")
            # Set CUDA device for this process
            try:
                torch.cuda.set_device(gpu_id)
                device = torch.device(f'cuda:{gpu_id}')
                print(f"Worker set device to cuda:{gpu_id}")
            except Exception as e:
                print(f"Error setting device for GPU {gpu_id}: {e}")
                device = torch.device('cpu')
                print(f"Falling back to CPU for worker {gpu_id}")
            
            # Create environments for this GPU
            envs = []
            for i in range(len(env_indices)):
                try:
                    env_config_copy = deepcopy(env_config)
                    
                    # Remove use_optimized_env from config before passing to environment constructor
                    use_optimized_env = env_config_copy.pop('use_optimized_env', True)
                    
                    # Set device for this environment
                    env_config_copy['device'] = device
                    
                    print(f"Creating environment {i} on device {device} with config: {env_config_copy}")
                    print(f"Using optimized environment: {use_optimized_env}")
                    print(f"Timeout penalty weight: {env_config_copy.get('timeout_penalty_weight', 'NOT SET')}")
                    
                    # Choose environment class based on use_optimized_env flag
                    if use_optimized_env:
                        from ..environment.optimized_race_env import OptimizedRaceEnvironment
                        env = OptimizedRaceEnvironment(**env_config_copy)
                        print(f"Successfully created OptimizedRaceEnvironment {i} on device {device}")
                    else:
                        from ..environment.race_env import RaceEnvironment
                        env = RaceEnvironment(**env_config_copy)
                        print(f"Successfully created RaceEnvironment {i} on device {device}")
                    
                    envs.append(env)
                    
                except Exception as e:
                    print(f"Error creating environment {i} on device {device}: {e}")
                    raise
            
            # Create a PPOTrainer with modified batch_size
            # Separate trainer initialization args from training args
            trainer_init_args = {
                'learning_rate': trainer_args.get('learning_rate', 3e-4),
                'batch_size': trainer_args.get('batch_size', 64) // total_envs * len(env_indices),
                'device': device,
                'multi_gpu': False,  # Each worker uses a single GPU
                'use_mixed_precision': False,  # Start with mixed precision disabled to be safe
                'network_config': trainer_args.get('network_config', {})
            }
            
            # Add PPO-specific parameters that go to the trainer constructor
            ppo_init_params = [
                'mini_batch_size', 'ppo_epochs', 'clip_param', 'value_coef', 
                'entropy_coef', 'max_grad_norm', 'gamma', 'gae_lambda'
            ]
            
            for param in ppo_init_params:
                if param in trainer_args:
                    trainer_init_args[param] = trainer_args[param]
            
            # Adjust learning rate for parallel training
            if 'learning_rate' in trainer_init_args:
                original_lr = trainer_init_args['learning_rate']
                # Scale learning rate based on number of workers
                trainer_init_args['learning_rate'] = original_lr * np.sqrt(num_gpus)
                print(f"Scaling learning rate from {original_lr} to {trainer_init_args['learning_rate']} for {num_gpus} workers")
            
            print(f"Creating trainer on device {device} with batch_size={trainer_init_args['batch_size']}")
            
            # Create trainer with the first environment - choose trainer type based on environment type
            try:
                if use_optimized_env:
                    from ..training.optimized_trainer import OptimizedPPOTrainer
                    trainer = OptimizedPPOTrainer(envs[0], **trainer_init_args)
                    print(f"Successfully created OptimizedPPOTrainer on device {device}")
                else:
                    from ..training.trainer import PPOTrainer
                    trainer = PPOTrainer(envs[0], **trainer_init_args)
                    print(f"Successfully created PPOTrainer on device {device}")
            except Exception as e:
                print(f"Error creating trainer on device {device}: {e}")
                raise
            
            # Extract training parameters from trainer_args
            training_args = {
                'num_iterations': trainer_args.get('num_iterations', 1000),
                'steps_per_iteration': trainer_args.get('steps_per_iteration', 600),
                'save_interval': trainer_args.get('save_interval', 50),
                'save_dir': trainer_args.get('save_dir', 'trained_models')
            }

            print(f"Worker on GPU {gpu_id} training with: iterations={training_args['num_iterations']}, steps={training_args['steps_per_iteration']}")
            print(f"Worker on GPU {gpu_id} timeout penalty enabled - no more timeout exploitation!")
            
            # Initialize mixed precision scaler if enabled
            scaler = None
            if use_mixed_precision:
                try:
                    scaler = GradScaler()
                    print("Mixed precision training enabled with standard GradScaler")
                except Exception as e:
                    print(f"Error initializing GradScaler: {e}")
                    use_mixed_precision = False
                    print("Mixed precision disabled due to errors")
            
            # Model synchronization functions
            def sync_models_with_shared():
                """Synchronize model parameters with shared models by role"""
                if shared_models is not None:
                    try:
                        for pod_key, network in trainer.pod_networks.items():
                            # Extract role from pod_key (pod0 = runner, pod1 = blocker)
                            parts = pod_key.split('_')
                            team_pod_idx = int(parts[1].replace('pod', ''))
                            role = "runner" if team_pod_idx == 0 else "blocker"
                            
                            # Use role-based key for shared models
                            role_key = f"{role}_network"
                            
                            if role_key in shared_models and len(shared_models[role_key]) > 0:
                                # Load shared parameters for this role
                                shared_state = dict(shared_models[role_key])
                                current_state = network.state_dict()
                                
                                # Weighted average: favor shared knowledge but keep some local adaptation
                                alpha = 0.8  # Weight for shared parameters
                                for param_name in current_state:
                                    if param_name in shared_state:
                                        shared_param = shared_state[param_name].to(device)
                                        current_state[param_name] = (
                                            alpha * shared_param + 
                                            (1 - alpha) * current_state[param_name]
                                        )
                                
                                network.load_state_dict(current_state)
                                print(f"GPU {gpu_id} - Synchronized {pod_key} ({role}) parameters")
                    except Exception as e:
                        print(f"GPU {gpu_id} - Error synchronizing models: {e}")

            def update_shared_models():
                """Update shared model parameters with local improvements by role"""
                if shared_models is not None:
                    try:
                        for pod_key, network in trainer.pod_networks.items():
                            # Extract role from pod_key
                            parts = pod_key.split('_')
                            team_pod_idx = int(parts[1].replace('pod', ''))
                            role = "runner" if team_pod_idx == 0 else "blocker"
                            
                            # Use role-based key for shared models
                            role_key = f"{role}_network"
                            
                            if role_key not in shared_models:
                                shared_models[role_key] = {}
                            
                            # Update shared parameters with local state for this role
                            local_state = network.state_dict()
                            for param_name, param_value in local_state.items():
                                shared_models[role_key][param_name] = param_value.cpu()
                            
                            print(f"GPU {gpu_id} - Updated shared {role} parameters from {pod_key}")
                    except Exception as e:
                        print(f"GPU {gpu_id} - Error updating shared models: {e}")
            
            def aggregate_gradients():
                """Aggregate gradients across workers by role"""
                nonlocal gradient_aggregation

                if not gradient_aggregation or shared_metrics is None:
                    return
                    
                try:
                    # Store gradients in shared metrics for aggregation by role
                    gradient_key = f"gradients_gpu_{gpu_id}"
                    
                    # Organize gradients by role
                    role_gradients = {"runner": {}, "blocker": {}}
                    
                    for pod_key, network in trainer.pod_networks.items():
                        # Extract role from pod_key
                        parts = pod_key.split('_')
                        team_pod_idx = int(parts[1].replace('pod', ''))
                        role = "runner" if team_pod_idx == 0 else "blocker"
                        
                        for name, param in network.named_parameters():
                            if param.grad is not None:
                                role_gradients[role][name] = param.grad.cpu()
                    
                    # Store the role-organized structure
                    shared_metrics[gradient_key] = role_gradients
                    
                    # Wait for other workers
                    time.sleep(0.3)
                    
                    # Check if we have gradients from all workers
                    gradient_keys = [k for k in shared_metrics.keys() if k.startswith("gradients_gpu_")]
                    if len(gradient_keys) >= num_gpus:
                        # Average gradients by role
                        for pod_key, network in trainer.pod_networks.items():
                            parts = pod_key.split('_')
                            team_pod_idx = int(parts[1].replace('pod', ''))
                            role = "runner" if team_pod_idx == 0 else "blocker"
                            
                            for name, param in network.named_parameters():
                                if param.grad is not None:
                                    grad_sum = torch.zeros_like(param.grad)
                                    count = 0
                                    
                                    for grad_worker_key in gradient_keys:
                                        try:
                                            worker_gradients = shared_metrics.get(grad_worker_key, {})
                                            if role in worker_gradients and name in worker_gradients[role]:
                                                grad_tensor = worker_gradients[role][name].to(device)
                                                grad_sum += grad_tensor
                                                count += 1
                                        except Exception:
                                            continue
                                    
                                    if count > 0:
                                        param.grad = grad_sum / count
                        
                        # Clear gradient storage
                        for grad_key in list(gradient_keys):
                            try:
                                del shared_metrics[grad_key]
                            except (KeyError, TypeError):
                                pass

                except Exception as e:
                    print(f"GPU {gpu_id} - Gradient aggregation disabled due to error: {e}")
                    gradient_aggregation = False
            
            def coordinate_learning_rates(avg_reward):
                """Coordinate learning rates based on global performance"""
                if adaptive_lr and shared_metrics is not None:
                    try:
                        # Update global metrics
                        shared_metrics[f'gpu_{gpu_id}_reward'] = avg_reward
                        
                        # Calculate global average reward
                        total_reward = 0
                        worker_count = 0
                        for i in range(num_gpus):
                            reward_key = f'gpu_{i}_reward'
                            if reward_key in shared_metrics:
                                total_reward += shared_metrics[reward_key]
                                worker_count += 1
                        
                        if worker_count > 0:
                            global_avg_reward = total_reward / worker_count
                            
                            # Adjust learning rate based on global performance
                            best_reward = shared_metrics.get('best_global_reward', float('-inf'))
                            if global_avg_reward > best_reward:
                                # Performance improving, maintain or slightly increase LR
                                lr_multiplier = 1.02
                                shared_metrics['best_global_reward'] = global_avg_reward
                                shared_metrics['stagnation_count'] = 0
                            else:
                                # Performance stagnating, track and potentially reduce LR
                                stagnation = shared_metrics.get('stagnation_count', 0) + 1
                                shared_metrics['stagnation_count'] = stagnation
                                
                                if stagnation > 5:  # Reduce LR after 5 iterations of stagnation
                                    lr_multiplier = 0.95
                                    shared_metrics['stagnation_count'] = 0  # Reset counter
                                else:
                                    lr_multiplier = 1.0  # Keep current LR
                            
                            # Update learning rates
                            for pod_key, optimizer in trainer.optimizers.items():
                                for param_group in optimizer.param_groups:
                                    old_lr = param_group['lr']
                                    param_group['lr'] *= lr_multiplier
                                    if lr_multiplier != 1.0:
                                        print(f"GPU {gpu_id} - Updated LR for {pod_key}: {old_lr:.6f} -> {param_group['lr']:.6f}")
                                        
                    except Exception as e:
                        print(f"GPU {gpu_id} - Error coordinating learning rates: {e}")
            
            def add_experience_to_shared_buffer(observations, actions, rewards, values, log_probs, dones):
                """Add experience to shared buffer for cross-worker learning"""
                if shared_experience is not None:
                    try:
                        for pod_key in trainer.pod_networks.keys():
                            if pod_key in observations:
                                experience = {
                                    'pod_key': pod_key,
                                    'observation': observations[pod_key].cpu(),
                                    'action': actions[pod_key].cpu(),
                                    'reward': rewards[pod_key].cpu(),
                                    'value': values[pod_key].cpu(),
                                    'log_prob': log_probs[pod_key].cpu(),
                                    'done': dones[pod_key].cpu() if isinstance(dones, dict) else dones.cpu(),
                                    'worker_id': gpu_id
                                }
                                shared_experience.add_experience(experience)
                    except Exception as e:
                        print(f"GPU {gpu_id} - Error adding experience to shared buffer: {e}")
            
            def sample_shared_experiences():
                """Sample experiences from shared buffer to augment local training"""
                if shared_experience is not None:
                    try:
                        shared_experiences = shared_experience.sample_experiences(batch_size=32)
                        if len(shared_experiences) > 0:
                            # Organize experiences by pod_key
                            shared_data = {}
                            for exp in shared_experiences:
                                pod_key = exp['pod_key']
                                if pod_key not in shared_data:
                                    shared_data[pod_key] = {
                                        'observations': [],
                                        'actions': [],
                                        'rewards': [],
                                        'values': [],
                                        'log_probs': [],
                                        'dones': []
                                    }
                                
                                shared_data[pod_key]['observations'].append(exp['observation'].to(device))
                                shared_data[pod_key]['actions'].append(exp['action'].to(device))
                                shared_data[pod_key]['rewards'].append(exp['reward'].to(device))
                                shared_data[pod_key]['values'].append(exp['value'].to(device))
                                shared_data[pod_key]['log_probs'].append(exp['log_prob'].to(device))
                                shared_data[pod_key]['dones'].append(exp['done'].to(device))
                            
                            # Add shared experiences to local storage
                            for pod_key, data in shared_data.items():
                                if pod_key in trainer.storage:
                                    for key, values in data.items():
                                        trainer.storage[pod_key][key].extend(values)
                            
                            print(f"GPU {gpu_id} - Added {len(shared_experiences)} shared experiences to training")
                            
                    except Exception as e:
                        print(f"GPU {gpu_id} - Error sampling shared experiences: {e}")
            
            # Add gradient clipping to prevent NaN values
            original_update_pod = trainer._update_pod
            
            def update_pod_with_nan_check(pod_key, data):
                # Add gradient clipping to prevent NaN values
                for param in trainer.pod_networks[pod_key].parameters():
                    if param.grad is not None:
                        # Check for NaN in gradients
                        if torch.isnan(param.grad).any():
                            print(f"NaN detected in gradients for {pod_key}. Zeroing out gradients.")
                            param.grad.zero_()
                
                # Call original update method
                return original_update_pod(pod_key, data)
            
            # Replace the trainer's _update_pod method
            trainer._update_pod = update_pod_with_nan_check
            
            # Override parallel_collect_trajectories with improved device handling and NaN checking
            def parallel_collect_trajectories(num_steps=600):
                # Reset storage
                trainer.reset_storage()
                
                # Initialize observations for all environments
                all_observations = [env.reset() for env in envs]
                
                for step in range(num_steps // len(envs)):
                    for env_idx, env in enumerate(envs):
                        observations = all_observations[env_idx]
                        
                        # Get actions from all pod networks
                        actions = {}
                        values = {}
                        log_probs = {}
                        
                        with torch.no_grad():
                            for pod_key, network in trainer.pod_networks.items():
                                # Ensure observation is on the correct device
                                if pod_key in observations:
                                    obs = observations[pod_key].to(device)
                                    
                                    # Check for NaN in observations
                                    if torch.isnan(obs).any():
                                        print(f"NaN detected in observations for {pod_key}. Resetting environment.")
                                        all_observations[env_idx] = env.reset()
                                        observations = all_observations[env_idx]
                                        obs = observations[pod_key].to(device)
                                    
                                    # Use autocast for mixed precision if enabled
                                    with autocast('cuda', enabled=use_mixed_precision):
                                        action = network.get_actions(obs, deterministic=False)
                                        
                                        # Get value and log probability
                                        _, value, _ = network(obs)
                                        log_prob = trainer.get_action_log_prob(network, obs, action)
                                    
                                    # Check for NaN in outputs
                                    if torch.isnan(action).any() or torch.isnan(value).any() or torch.isnan(log_prob).any():
                                        print(f"NaN detected in network outputs for {pod_key}. Using zero actions.")
                                        action = torch.zeros_like(action)
                                        value = torch.zeros_like(value)
                                        log_prob = torch.zeros_like(log_prob)
                                    
                                    actions[pod_key] = action
                                    values[pod_key] = value
                                    log_probs[pod_key] = log_prob
                        
                        # Step the environment
                        next_observations, rewards, dones, info = env.step(actions)
                        
                        # Check for NaN in rewards
                        for pod_key in rewards:
                            if torch.isnan(rewards[pod_key]).any():
                                print(f"NaN detected in rewards for {pod_key}. Replacing with zeros.")
                                rewards[pod_key] = torch.zeros_like(rewards[pod_key])
                        
                        # Log timeout penalties for debugging if available
                        if 'timeout_penalties' in info:
                            timeout_info = info['timeout_penalties']
                            if step % 100 == 0:  # Log every 100 steps
                                timeout_values = [float(timeout_info[i][0]) for i in range(4)]
                                print(f"GPU {gpu_id} - Step {step} timeout penalties: {timeout_values}")
                        
                        # Add experience to shared buffer
                        add_experience_to_shared_buffer(observations, actions, rewards, values, log_probs, dones)
                        
                        # Update pod trails for visualization
                        if vis_queue is not None and not hasattr(trainer, 'pod_trails'):
                            trainer.pod_trails = {pod_key: [] for pod_key in trainer.pod_networks.keys()}
                        
                        if vis_queue is not None and hasattr(trainer, 'pod_trails'):
                            for pod_key in trainer.pod_networks.keys():
                                if pod_key in observations:
                                    # Extract position from observation (first 2 values)
                                    obs = observations[pod_key][0]  # Get first batch item
                                    if len(obs) >= 2:
                                        pos = obs[0:2].tolist()
                                        if not hasattr(trainer.pod_trails, pod_key):
                                            trainer.pod_trails[pod_key] = []
                                        trainer.pod_trails[pod_key].append(pos)
                                        # Keep trail length manageable
                                        if len(trainer.pod_trails[pod_key]) > 50:
                                            trainer.pod_trails[pod_key] = trainer.pod_trails[pod_key][-50:]
                        
                        # Send race state to visualization occasionally
                        if vis_queue is not None and step % 10 == 0:
                            try:
                                # Create a unique worker ID that includes both GPU ID and environment index
                                worker_id = f"gpu_{gpu_id}_env_{env_indices[env_idx]}"
                                
                                # Get checkpoints from environment
                                checkpoints = []
                                if hasattr(env, 'get_checkpoints'):
                                    checkpoints = env.get_checkpoints()
                                elif hasattr(env, 'checkpoints') and isinstance(env.checkpoints, torch.Tensor):
                                    # For optimized environment
                                    # Return first batch's checkpoints
                                    checkpoints = env.checkpoints[0].cpu().numpy().tolist()

                                # Create race state for visualization
                                vis_race_state = {
                                    'checkpoints': checkpoints,
                                    'pods': [],
                                    'worker_id': worker_id  # Include both GPU ID and environment index
                                }
                                
                                # Add pod information from observations
                                for pod_key in trainer.pod_networks.keys():
                                    if pod_key in observations:
                                        obs = observations[pod_key][0]  # Get first batch item
                                        
                                        # Extract position (first 2 values) and angle (3rd value) from observation
                                        if len(obs) >= 3:
                                            # Convert from normalized [-1,1] back to world coordinates if needed
                                            position = obs[0:2].tolist()
                                            
                                            # Check if position values are normalized
                                            if abs(position[0]) <= 1.0 and abs(position[1]) <= 1.0:
                                                position[0] = position[0] * 8000 + 8000  # WIDTH/2 = 8000
                                                position[1] = position[1] * 4500 + 4500  # HEIGHT/2 = 4500
                                            
                                            angle = float(obs[2])
                                            # Convert to degrees if normalized between -1 and 1
                                            if abs(angle) <= 1.0:
                                                angle = angle * 180
                                            
                                            # Get trail if available
                                            trail = []
                                            if hasattr(trainer, 'pod_trails') and pod_key in trainer.pod_trails:
                                                raw_trail = trainer.pod_trails[pod_key]
                                                # Convert trail positions if they're normalized
                                                trail = []
                                                for pos in raw_trail:
                                                    if abs(pos[0]) <= 1.0 and abs(pos[1]) <= 1.0:
                                                        # Convert from normalized [-1,1] back to world coordinates
                                                        trail_pos = [
                                                            pos[0] * 8000 + 8000,  # WIDTH/2 = 8000
                                                            pos[1] * 4500 + 4500   # HEIGHT/2 = 4500
                                                        ]
                                                        trail.append(trail_pos)
                                                    else:
                                                        trail.append(pos)
                                            
                                            pod_info = {
                                                'position': position,
                                                'angle': angle,
                                                'trail': trail
                                            }
                                            
                                            vis_race_state['pods'].append(pod_info)
                                
                                # Put in queue if we have pod data
                                if len(vis_race_state['pods']) > 0 and len(vis_race_state['checkpoints']) > 0:
                                    vis_queue.put({'race_state': vis_race_state})
                            except Exception as e:
                                # Don't let visualization errors affect training
                                print(f"Error sending race state to visualization: {e}")
                                print(traceback.format_exc())

                        # Store trajectory data with explicit device management and NaN checking
                        for pod_key in trainer.pod_networks.keys():
                            if pod_key not in observations:
                                continue

                            # Check for NaN in data before storing
                            obs_ok = not torch.isnan(observations[pod_key]).any()
                            act_ok = not torch.isnan(actions[pod_key]).any()
                            rew_ok = not torch.isnan(rewards[pod_key]).any()
                            val_ok = not torch.isnan(values[pod_key]).any()
                            log_ok = not torch.isnan(log_probs[pod_key]).any()
                            
                            if not (obs_ok and act_ok and rew_ok and val_ok and log_ok):
                                print(f"NaN detected in trajectory data for {pod_key}. Skipping storage.")
                                continue

                            trainer.storage[pod_key]['observations'].append(observations[pod_key].to(device))
                            trainer.storage[pod_key]['actions'].append(actions[pod_key].to(device))
                            trainer.storage[pod_key]['rewards'].append(rewards[pod_key].to(device))
                            trainer.storage[pod_key]['values'].append(values[pod_key].to(device))
                            trainer.storage[pod_key]['log_probs'].append(log_probs[pod_key].to(device))
                            
                            # Handle dones properly
                            if isinstance(dones, dict):
                                pod_done = dones[pod_key].to(device)
                                if pod_done.dim() == 0:
                                    pod_done = pod_done.unsqueeze(0).unsqueeze(1)
                                elif pod_done.dim() == 1:
                                    pod_done = pod_done.unsqueeze(1)
                                trainer.storage[pod_key]['dones'].append(pod_done)
                            else:
                                dones_device = dones.to(device)
                                if dones_device.dim() == 1:
                                    trainer.storage[pod_key]['dones'].append(dones_device.unsqueeze(1))
                                elif dones_device.dim() == 2:
                                    trainer.storage[pod_key]['dones'].append(dones_device[:, 0:1])
                                else:
                                    sliced_dones = dones_device[:, 0, 0].unsqueeze(1)
                                    trainer.storage[pod_key]['dones'].append(sliced_dones)
                        
                        all_observations[env_idx] = next_observations
                        
                        # Reset environment if all episodes are done
                        # NOTE: Episodes now only reset when race is completed, not on timeout
                        if dones.all():
                            all_observations[env_idx] = env.reset()
                            print(f"GPU {gpu_id} - Environment {env_idx} reset due to race completion")
                        
                        # Ensure synchronization between operations
                        if device.type == 'cuda':
                            torch.cuda.synchronize(device)
                
                # Sample shared experiences to augment training data
                sample_shared_experiences()

            # Replace the trainer's collect_trajectories method
            trainer.collect_trajectories = parallel_collect_trajectories
            
            # Override compute_returns_and_advantages with NaN checking
            original_compute_returns = trainer.compute_returns_and_advantages
            
            def compute_returns_with_nan_check():
                # Call original method
                original_compute_returns()
                
                # Check for NaN in returns and advantages
                for pod_key in trainer.storage.keys():
                    if 'returns' in trainer.storage[pod_key]:
                        # Check if returns is a list of tensors or a single tensor
                        if isinstance(trainer.storage[pod_key]['returns'], list):
                            # It's a list of tensors, so we can concatenate
                            try:
                                returns_tensor = torch.cat(trainer.storage[pod_key]['returns'])
                                if torch.isnan(returns_tensor).any():
                                    print(f"NaN detected in returns for {pod_key}. Replacing with zeros.")
                                    for i in range(len(trainer.storage[pod_key]['returns'])):
                                        trainer.storage[pod_key]['returns'][i] = torch.zeros_like(trainer.storage[pod_key]['returns'][i])
                            except Exception as e:
                                print(f"Error checking returns for NaN: {e}")
                                # If there's an error, just zero out all returns to be safe
                                for i in range(len(trainer.storage[pod_key]['returns'])):
                                    trainer.storage[pod_key]['returns'][i] = torch.zeros_like(trainer.storage[pod_key]['returns'][i])
                        else:
                            # It's a single tensor
                            if torch.isnan(trainer.storage[pod_key]['returns']).any():
                                print(f"NaN detected in returns for {pod_key}. Replacing with zeros.")
                                trainer.storage[pod_key]['returns'] = torch.zeros_like(trainer.storage[pod_key]['returns'])
                    
                    if 'advantages' in trainer.storage[pod_key]:
                        # Check if advantages is a list of tensors or a single tensor
                        if isinstance(trainer.storage[pod_key]['advantages'], list):
                            # It's a list of tensors, so we can concatenate
                            try:
                                advantages_tensor = torch.cat(trainer.storage[pod_key]['advantages'])
                                if torch.isnan(advantages_tensor).any():
                                    print(f"NaN detected in advantages for {pod_key}. Replacing with zeros.")
                                    for i in range(len(trainer.storage[pod_key]['advantages'])):
                                        trainer.storage[pod_key]['advantages'][i] = torch.zeros_like(trainer.storage[pod_key]['advantages'][i])
                            except Exception as e:
                                print(f"Error checking advantages for NaN: {e}")
                                # If there's an error, just zero out all advantages to be safe
                                for i in range(len(trainer.storage[pod_key]['advantages'])):
                                    trainer.storage[pod_key]['advantages'][i] = torch.zeros_like(trainer.storage[pod_key]['advantages'][i])
                        else:
                            # It's a single tensor
                            if torch.isnan(trainer.storage[pod_key]['advantages']).any():
                                print(f"NaN detected in advantages for {pod_key}. Replacing with zeros.")
                                trainer.storage[pod_key]['advantages'] = torch.zeros_like(trainer.storage[pod_key]['advantages'])

            # Replace the trainer's compute_returns_and_advantages method
            trainer.compute_returns_and_advantages = compute_returns_with_nan_check

            # Override update_policy to return loss values and handle NaN with gradient aggregation
            original_update_policy = trainer.update_policy

            def update_policy_with_losses_and_nan_check():
                total_policy_loss = 0
                total_value_loss = 0
                pod_count = 0
                
                # Process each pod
                for pod_key, data in trainer.storage.items():
                    try:
                        losses = trainer._update_pod(pod_key, data)
                        
                        # Check for NaN in losses
                        policy_loss = losses.get('policy_loss', 0)
                        value_loss = losses.get('value_loss', 0)
                        
                        if isinstance(policy_loss, torch.Tensor) and torch.isnan(policy_loss).any():
                            print(f"NaN detected in policy loss for {pod_key}. Using zero loss.")
                            policy_loss = 0.0
                        
                        if isinstance(value_loss, torch.Tensor) and torch.isnan(value_loss).any():
                            print(f"NaN detected in value loss for {pod_key}. Using zero loss.")
                            value_loss = 0.0
                        
                        total_policy_loss += policy_loss
                        total_value_loss += value_loss
                        pod_count += 1
                    except Exception as e:
                        print(f"Error updating policy for {pod_key}: {e}")
                        print(traceback.format_exc())
                        # Continue with other pods
                
                # Aggregate gradients across workers before optimizer step
                aggregate_gradients()
                
                # Calculate average losses
                avg_policy_loss = total_policy_loss / max(1, pod_count)
                avg_value_loss = total_value_loss / max(1, pod_count)
                
                return avg_policy_loss, avg_value_loss

            # Replace the trainer's update_policy method
            trainer.update_policy = update_policy_with_losses_and_nan_check

            # Create log file for training metrics
            save_dir = Path(training_args['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            log_file = save_dir / f"training_log_gpu_{gpu_id}.txt"

            # Train for the specified number of iterations
            print(f"Starting training with num_iterations={training_args['num_iterations']}, steps_per_iteration={training_args['steps_per_iteration']}")

            # Add a learning rate scheduler for stability
            initial_lr = trainer_init_args.get('learning_rate', 3e-4)
            min_lr = initial_lr * 0.1  # Minimum learning rate is 10% of initial

            # Function to check and fix NaN in network parameters
            def check_and_fix_network_parameters():
                for pod_key, network in trainer.pod_networks.items():
                    for name, param in network.named_parameters():
                        if torch.isnan(param).any():
                            print(f"NaN detected in {pod_key} network parameter {name}. Reinitializing.")
                            if 'weight' in name:
                                torch.nn.init.xavier_uniform_(param)
                            elif 'bias' in name:
                                torch.nn.init.zeros_(param)

            # Check networks before starting training
            check_and_fix_network_parameters()

            # Initialize shared models if this is the first worker
            if shared_models is not None and gpu_id == 0:
                print(f"GPU {gpu_id} - Initializing shared model parameters")
                update_shared_models()

            # Training metrics tracking for timeout prevention
            timeout_penalty_stats = {
                'total_penalties': 0,
                'max_penalty_per_iteration': 0,
                'episodes_with_penalties': 0
            }

            for iteration in range(1, training_args['num_iterations'] + 1):
                start_time = time.time()
                
                # Synchronize models periodically
                if iteration % sync_interval == 0 and iteration > 1 and shared_models is not None:
                    sync_models_with_shared()
                
                # Adjust learning rate based on iteration (simple linear decay)
                progress = iteration / training_args['num_iterations']
                current_lr = max(min_lr, initial_lr * (1 - 0.9 * progress))  # Linear decay to 10% of initial
                
                # Update learning rate in optimizer (will be further adjusted by coordinate_learning_rates)
                base_lr = current_lr
                for pod_key, optimizer in trainer.optimizers.items():
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = base_lr
                
                try:
                    # Collect trajectories
                    trainer.collect_trajectories(training_args['steps_per_iteration'])
                    
                    # Compute returns and advantages
                    trainer.compute_returns_and_advantages()
                    
                    # Update policy
                    policy_loss, value_loss = trainer.update_policy()
                    
                    # Check for NaN in network parameters after update
                    check_and_fix_network_parameters()
                    
                    # Update shared models after successful training step
                    if iteration % sync_interval == 0 and shared_models is not None:
                        update_shared_models()
                    
                    # Calculate elapsed time since start of iteration
                    elapsed_time = time.time() - start_time
                    
                    # Calculate role-specific average rewards and timeout penalties
                    runner_reward = 0
                    blocker_reward = 0
                    runner_count = 0
                    blocker_count = 0
                    total_timeout_penalties = 0
                    max_timeout_penalty = 0
                    episodes_with_timeout_penalty = 0

                    for pod_key in trainer.pod_networks.keys():
                        if 'rewards' in trainer.storage[pod_key]:
                            rewards = torch.cat([r for r in trainer.storage[pod_key]['rewards']])
                            avg_pod_reward = rewards.mean().item()
                            
                            # Check for timeout penalties in rewards (negative values beyond normal reward range)
                            timeout_penalties = rewards[rewards < -0.1]  # Assuming normal rewards are >= -0.1
                            if len(timeout_penalties) > 0:
                                total_timeout_penalties += len(timeout_penalties)
                                max_timeout_penalty = max(max_timeout_penalty, abs(timeout_penalties.min().item()))
                                episodes_with_timeout_penalty += 1
                            
                            # Determine if this is a runner or blocker
                            if 'pod0' in pod_key:  # Runner
                                runner_reward += avg_pod_reward
                                runner_count += 1
                            elif 'pod1' in pod_key:  # Blocker
                                blocker_reward += avg_pod_reward
                                blocker_count += 1

                    avg_runner_reward = runner_reward / max(1, runner_count)
                    avg_blocker_reward = blocker_reward / max(1, blocker_count)
                    avg_total_reward = (runner_reward + blocker_reward) / max(1, runner_count + blocker_count)

                    # Update timeout penalty statistics
                    timeout_penalty_stats['total_penalties'] += total_timeout_penalties
                    timeout_penalty_stats['max_penalty_per_iteration'] = max(
                        timeout_penalty_stats['max_penalty_per_iteration'], 
                        max_timeout_penalty
                    )
                    timeout_penalty_stats['episodes_with_penalties'] += episodes_with_timeout_penalty

                    # Coordinate learning rates based on performance
                    coordinate_learning_rates(avg_total_reward)
                    
                    # Get current learning rate after coordination
                    current_lr = trainer.optimizers[next(iter(trainer.optimizers))].param_groups[0]['lr']

                    # Log progress with role-specific information and timeout prevention metrics
                    print(f"GPU {gpu_id} - Iteration {iteration}/{training_args['num_iterations']} completed in {elapsed_time:.2f}s")
                    print(f"  Total reward: {avg_total_reward:.4f}")
                    print(f"  Runner reward: {avg_runner_reward:.4f}")
                    print(f"  Blocker reward: {avg_blocker_reward:.4f}")
                    print(f"  Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}, lr={current_lr:.6f}")
                    
                    # Log timeout prevention metrics
                    if total_timeout_penalties > 0:
                        print(f"  TIMEOUT PENALTIES: {total_timeout_penalties} occurrences, max penalty: {max_timeout_penalty:.4f}")
                        print(f"  Episodes with penalties: {episodes_with_timeout_penalty}")
                    else:
                        print(f"  NO TIMEOUT PENALTIES - agents actively racing!")

                    # Log metrics to file with role breakdown and timeout prevention
                    with open(log_file, 'a') as f:
                        f.write(f"iteration={iteration},total_reward={avg_total_reward:.4f},"
                            f"runner_reward={avg_runner_reward:.4f},blocker_reward={avg_blocker_reward:.4f},"
                            f"policy_loss={policy_loss:.4f},value_loss={value_loss:.4f},"
                            f"time={elapsed_time:.2f},lr={current_lr:.6f},"
                            f"timeout_penalties={total_timeout_penalties},max_penalty={max_timeout_penalty:.4f},"
                            f"episodes_with_penalties={episodes_with_timeout_penalty}\n")

                    # Send enhanced metrics to visualization if enabled
                    if vis_queue is not None:
                        metrics = {
                            'iteration': iteration,
                            'total_reward': avg_total_reward,
                            'runner_reward': avg_runner_reward,
                            'blocker_reward': avg_blocker_reward,
                            'policy_loss': policy_loss,
                            'value_loss': value_loss,
                            'learning_rate': current_lr,
                            'timeout_penalties': total_timeout_penalties,
                            'max_timeout_penalty': max_timeout_penalty,
                            'episodes_with_penalties': episodes_with_timeout_penalty,
                            'worker_id': f"gpu_{gpu_id}"
                        }
                        vis_queue.put(metrics)
                        print(f"GPU {gpu_id} sent metrics for iteration {iteration}")
                    
                    # Save models periodically
                    if iteration % training_args['save_interval'] == 0:
                        for pod_key, network in trainer.pod_networks.items():
                            # Check for NaN before saving
                            has_nan = False
                            for param in network.parameters():
                                if torch.isnan(param).any():
                                    has_nan = True
                                    break
                            
                            if not has_nan:
                                # Extract player and pod indices from pod_key
                                # pod_key format: "player{player_idx}_pod{team_pod_idx}"
                                parts = pod_key.split('_')
                                player_idx = parts[0].replace('player', '')
                                team_pod_idx = int(parts[1].replace('pod', ''))
                                
                                # Determine role based on team_pod_idx
                                role = "runner" if team_pod_idx == 0 else "blocker"
                                
                                # Create descriptive filename with role and timeout prevention info
                                base_filename = f"player{player_idx}_{role}_notimeout_gpu{gpu_id}"
                                
                                # Save with iteration number
                                torch.save(network.state_dict(), save_dir / f"{base_filename}_iter{iteration}.pt")
                                # Also save as the latest model
                                torch.save(network.state_dict(), save_dir / f"{base_filename}.pt")
                                print(f"GPU {gpu_id} - Models saved at iteration {iteration} as {base_filename}")
                            else:
                                print(f"GPU {gpu_id} - Not saving model for {pod_key} due to NaN parameters")
                    
                    # Report timeout prevention progress every 10 iterations
                    if iteration % 10 == 0:
                        avg_penalties_per_iter = timeout_penalty_stats['total_penalties'] / iteration
                        print(f"GPU {gpu_id} - TIMEOUT PREVENTION REPORT (Iteration {iteration}):")
                        print(f"  Average penalties per iteration: {avg_penalties_per_iter:.2f}")
                        print(f"  Max penalty seen: {timeout_penalty_stats['max_penalty_per_iteration']:.4f}")
                        print(f"  Episodes with penalties: {timeout_penalty_stats['episodes_with_penalties']}")
                        
                        # Success metric: low penalty rate indicates agents are actively racing
                        if avg_penalties_per_iter < 1.0:
                            print(f"  SUCCESS: Low penalty rate indicates agents are actively racing!")
                        elif avg_penalties_per_iter > 5.0:
                            print(f"  WARNING: High penalty rate - agents may still be exploiting timeouts")
                    
                    # Try enabling mixed precision after a few successful iterations if it was requested
                    if use_mixed_precision and not trainer.use_mixed_precision and iteration == 10:
                        print(f"GPU {gpu_id} - Attempting to enable mixed precision after 10 successful iterations")
                        try:
                            # Test mixed precision with a small forward pass
                            test_input = torch.randn(4, trainer.pod_networks[next(iter(trainer.pod_networks))].input_dim, 
                                                    device=device)
                            with autocast('cuda', enabled=True):
                                test_output = trainer.pod_networks[next(iter(trainer.pod_networks))](test_input)
                                if not torch.isnan(test_output[0]).any() and not torch.isnan(test_output[1]).any():
                                    trainer.use_mixed_precision = True
                                    print(f"GPU {gpu_id} - Mixed precision enabled successfully")
                                else:
                                    print(f"GPU {gpu_id} - Mixed precision test produced NaN values. Keeping it disabled.")
                        except Exception as e:
                            print(f"GPU {gpu_id} - Error testing mixed precision: {e}. Keeping it disabled.")
                
                except Exception as e:
                    print(f"GPU {gpu_id} - Error in iteration {iteration}: {e}")
                    print(traceback.format_exc())
                    # Continue with next iteration instead of crashing

            # Store results in shared dictionary with timeout prevention metrics
            shared_results[f'gpu_{gpu_id}'] = {
                'completed': True,
                'env_indices': env_indices,
                'final_rewards': {
                    'total': avg_total_reward,
                    'runner': avg_runner_reward,
                    'blocker': avg_blocker_reward
                },
                'final_losses': {
                    'policy': policy_loss,
                    'value': value_loss
                },
                'timeout_prevention_metrics': {
                    'total_penalties': timeout_penalty_stats['total_penalties'],
                    'avg_penalties_per_iteration': timeout_penalty_stats['total_penalties'] / training_args['num_iterations'],
                    'max_penalty': timeout_penalty_stats['max_penalty_per_iteration'],
                    'episodes_with_penalties': timeout_penalty_stats['episodes_with_penalties']
                }
            }

        except Exception as e:
            print(f"Error in worker process for GPU {gpu_id}: {str(e)}")
            print(traceback.format_exc())
            shared_results[f'gpu_{gpu_id}'] = {
                'completed': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def train(self, visualization_panel=None):
        """Start parallel training across GPUs with improved error handling and timeout prevention"""
        print("Starting parallel training with timeout penalty system...")
        print(f"Training parameters: num_iterations={self.trainer_args.get('num_iterations')}, steps_per_iteration={self.trainer_args.get('steps_per_iteration')}")
        print(f"Timeout prevention enabled with penalty weight: {self.env_config.get('timeout_penalty_weight', 0.01)}")
        print(f"Synchronization settings:")
        print(f"  - Sync interval: {self.sync_interval}")
        print(f"  - Parameter server: {self.use_parameter_server}")
        print(f"  - Shared experience: {self.shared_experience_buffer}")
        print(f"  - Gradient aggregation: {self.gradient_aggregation}")
        print(f"  - Adaptive learning rate: {self.adaptive_lr}")
        
        # Create environment indices for each GPU
        all_env_indices = np.arange(self.total_envs).reshape(self.num_gpus, self.envs_per_gpu)
        
        # Create a manager for sharing results between processes
        manager = mp.Manager()
        shared_results = manager.dict()

        # Create shared model storage if using parameter server
        shared_models = None
        if self.use_parameter_server and self.num_gpus > 1:
            print("Setting up shared model parameters...")
            shared_models = manager.dict()
            
            # Initialize shared model parameters by role instead of pod_key
            shared_models['runner_network'] = manager.dict()
            shared_models['blocker_network'] = manager.dict()
            print("Initialized shared models for runner and blocker roles")

        # Create shared experience buffer if enabled
        shared_experience = None
        if self.shared_experience_buffer:
            print("Setting up shared experience buffer...")
            shared_experience = SharedExperienceBuffer(manager, buffer_size=20000)
            print("Shared experience buffer initialized")

        # Create shared metrics for coordination
        shared_metrics = manager.dict()
        shared_metrics['best_global_reward'] = float('-inf')
        shared_metrics['stagnation_count'] = 0

        # Create a shared queue for visualization if needed
        vis_queue = None
        vis_connector = None
        if visualization_panel is not None:
            print("Setting up visualization connection...")
            vis_queue = manager.Queue()
            from sebulba_pod_trainer.training.visualization_connector import VisualizationConnector
            vis_connector = VisualizationConnector(visualization_panel)
            vis_connector.start(vis_queue)
            print("Visualization connection established")

        # Disable mixed precision by default and let workers decide if it's safe
        self.use_mixed_precision = False
        print("Mixed precision initially disabled for stability. Workers will enable if safe.")

        # Start processes for each GPU
        for i, gpu_id in enumerate(self.devices):
            env_indices = all_env_indices[i].tolist()
            print(f"Starting worker process for GPU {gpu_id} with env indices {env_indices}")
            
            # Pass all necessary data to the worker process
            p = mp.Process(
                target=self._worker,
                args=(
                    gpu_id,                      # GPU ID
                    env_indices,                 # Environment indices
                    deepcopy(self.env_config),   # Environment config (deep copy to avoid reference issues)
                    deepcopy(self.trainer_args), # Trainer args (deep copy to avoid reference issues)
                    self.total_envs,             # Total number of environments
                    self.envs_per_gpu,           # Environments per GPU
                    self.use_mixed_precision,    # Whether to use mixed precision
                    shared_results,              # Shared results dictionary
                    vis_queue,                   # Shared queue for visualization
                    shared_models,               # Shared model parameters
                    shared_experience,           # Shared experience buffer
                    shared_metrics,              # Shared metrics for coordination
                    self.sync_interval,          # Model synchronization interval
                    self.gradient_aggregation,   # Whether to aggregate gradients
                    self.adaptive_lr,            # Whether to use adaptive learning rate
                    self.num_gpus                # Number of GPUs for coordination
                )
            )
            self.processes.append(p)
            p.start()

        # Monitor training progress
        print("Monitoring parallel training progress with timeout prevention...")
        start_time = time.time()
        last_sync_report = time.time()
        
        # Wait for all processes to finish
        try:
            # Monitor processes and periodically check if they're alive
            while any(p.is_alive() for p in self.processes):
                time.sleep(2.0)
                
                # Report synchronization status periodically
                current_time = time.time()
                if current_time - last_sync_report > 30:  # Report every 30 seconds
                    alive_workers = sum(1 for p in self.processes if p.is_alive())
                    elapsed_minutes = (current_time - start_time) / 60
                    
                    print(f"Training status after {elapsed_minutes:.1f} minutes:")
                    print(f"  - Active workers: {alive_workers}/{len(self.processes)}")
                    print(f"  - Timeout prevention system active")
                    
                    # Report shared metrics if available
                    if shared_metrics:
                        best_reward = shared_metrics.get('best_global_reward', 'N/A')
                        stagnation = shared_metrics.get('stagnation_count', 'N/A')
                        print(f"  - Best global reward: {best_reward}")
                        print(f"  - Stagnation count: {stagnation}")
                        
                        # Report individual worker rewards
                        worker_rewards = []
                        for i in range(self.num_gpus):
                            reward_key = f'gpu_{i}_reward'
                            if reward_key in shared_metrics:
                                worker_rewards.append(f"GPU{i}: {shared_metrics[reward_key]:.4f}")
                        
                        if worker_rewards:
                            print(f"  - Worker rewards: {', '.join(worker_rewards)}")
                    
                    last_sync_report = current_time
                
                # Check for early termination due to errors
                for key, result in shared_results.items():
                    if 'completed' in result and result['completed'] is False:
                        print(f"Error detected in {key}: {result.get('error', 'Unknown error')}")
                        # Don't terminate other processes, let them complete naturally
                
        except KeyboardInterrupt:
            print("Training interrupted by user. Terminating processes...")
            for p in self.processes:
                if p.is_alive():
                    p.terminate()
        finally:
            # Wait for processes to finish
            print("Waiting for all processes to complete...")
            for i, p in enumerate(self.processes):
                p.join(timeout=10.0)
                if p.is_alive():
                    print(f"Process {i} did not terminate gracefully, forcing termination...")
                    p.terminate()
                    p.join(timeout=5.0)
            
            # Stop visualization connector if it was started
            if vis_connector:
                print("Stopping visualization connector...")
                vis_connector.stop()
        
        print("All worker processes completed")
        
        # Report final results with timeout prevention metrics
        self._report_training_results(shared_results)
        
        # Check for errors in any of the processes
        errors = []
        for key, result in shared_results.items():
            if not result.get('completed', False):
                errors.append(f"{key}: {result.get('error', 'Unknown error')}")
        
        if errors:
            error_msg = "Errors occurred in worker processes:\n" + "\n".join(errors)
            print(error_msg)
            raise RuntimeError(error_msg)
        
        print("Parallel training with timeout prevention completed successfully")
        return True

    def _report_training_results(self, shared_results):
        """Report final training results from all workers with timeout prevention metrics"""
        print("\n" + "="*60)
        print("PARALLEL TRAINING RESULTS WITH TIMEOUT PREVENTION")
        print("="*60)
        
        total_rewards = []
        runner_rewards = []
        blocker_rewards = []
        policy_losses = []
        value_losses = []
        timeout_metrics = []
        
        for key, result in shared_results.items():
            if result.get('completed', False):
                gpu_id = key.replace('gpu_', '')
                final_rewards = result.get('final_rewards', {})
                final_losses = result.get('final_losses', {})
                timeout_prevention = result.get('timeout_prevention_metrics', {})
                
                print(f"\nGPU {gpu_id} Results:")
                print(f"  Total Reward: {final_rewards.get('total', 'N/A'):.4f}")
                print(f"  Runner Reward: {final_rewards.get('runner', 'N/A'):.4f}")
                print(f"  Blocker Reward: {final_rewards.get('blocker', 'N/A'):.4f}")
                print(f"  Policy Loss: {final_losses.get('policy', 'N/A'):.4f}")
                print(f"  Value Loss: {final_losses.get('value', 'N/A'):.4f}")
                
                # Report timeout prevention metrics
                if timeout_prevention:
                    print(f"  Timeout Prevention Metrics:")
                    print(f"    - Total penalties: {timeout_prevention.get('total_penalties', 0)}")
                    print(f"    - Avg penalties/iteration: {timeout_prevention.get('avg_penalties_per_iteration', 0):.2f}")
                    print(f"    - Max penalty: {timeout_prevention.get('max_penalty', 0):.4f}")
                    print(f"    - Episodes with penalties: {timeout_prevention.get('episodes_with_penalties', 0)}")
                    
                    # Success indicator
                    avg_penalties = timeout_prevention.get('avg_penalties_per_iteration', 0)
                    if avg_penalties < 1.0:
                        print(f"    - STATUS: SUCCESS - Agents actively racing!")
                    elif avg_penalties < 5.0:
                        print(f"    - STATUS: GOOD - Low timeout exploitation")
                    else:
                        print(f"    - STATUS: WARNING - High timeout exploitation")
                
                # Collect for averaging
                if 'total' in final_rewards:
                    total_rewards.append(final_rewards['total'])
                if 'runner' in final_rewards:
                    runner_rewards.append(final_rewards['runner'])
                if 'blocker' in final_rewards:
                    blocker_rewards.append(final_rewards['blocker'])
                if 'policy' in final_losses:
                    policy_losses.append(final_losses['policy'])
                if 'value' in final_losses:
                    value_losses.append(final_losses['value'])
                if timeout_prevention:
                    timeout_metrics.append(timeout_prevention)
        
        # Calculate and report averages
        if total_rewards:
            print(f"\nAVERAGE ACROSS ALL WORKERS:")
            print(f"  Average Total Reward: {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")
            print(f"  Average Runner Reward: {np.mean(runner_rewards):.4f} ± {np.std(runner_rewards):.4f}")
            print(f"  Average Blocker Reward: {np.mean(blocker_rewards):.4f} ± {np.std(blocker_rewards):.4f}")
            print(f"  Average Policy Loss: {np.mean(policy_losses):.4f} ± {np.std(policy_losses):.4f}")
            print(f"  Average Value Loss: {np.mean(value_losses):.4f} ± {np.std(value_losses):.4f}")
        
        # Report timeout prevention summary
        if timeout_metrics:
            total_penalties = sum(tm.get('total_penalties', 0) for tm in timeout_metrics)
            avg_penalties_per_iter = np.mean([tm.get('avg_penalties_per_iteration', 0) for tm in timeout_metrics])
            max_penalty_overall = max(tm.get('max_penalty', 0) for tm in timeout_metrics)
            total_episodes_with_penalties = sum(tm.get('episodes_with_penalties', 0) for tm in timeout_metrics)
            
            print(f"\nTIMEOUT PREVENTION SUMMARY:")
            print(f"  Total penalties across all workers: {total_penalties}")
            print(f"  Average penalties per iteration: {avg_penalties_per_iter:.2f}")
            print(f"  Maximum penalty seen: {max_penalty_overall:.4f}")
            print(f"  Episodes with penalties: {total_episodes_with_penalties}")
            
            # Overall success assessment
            if avg_penalties_per_iter < 1.0:
                print(f"  OVERALL STATUS: ✅ SUCCESS - Timeout exploitation eliminated!")
            elif avg_penalties_per_iter < 5.0:
                print(f"  OVERALL STATUS: ⚠️  GOOD - Low timeout exploitation")
            else:
                print(f"  OVERALL STATUS: ❌ WARNING - Timeout exploitation still present")
        
        print("="*60)

    def _process_visualization_data(self, vis_queue, visualization_panel):
        """Process visualization data from the queue and send to the visualization panel"""
        try:
            print("Starting visualization data processing thread")
            while True:
                try:
                    # Get data from queue with timeout to allow checking if processes are alive
                    data = vis_queue.get(timeout=1.0)
                    
                    # Debug print to verify data is being received
                    if 'iteration' in data:
                        timeout_info = ""
                        if 'timeout_penalties' in data:
                            timeout_info = f", penalties={data['timeout_penalties']}"
                        print(f"Received metric data: iteration={data['iteration']}, reward={data.get('total_reward', 0):.4f}{timeout_info}")
                    elif 'race_state' in data:
                        print(f"Received race state data with {len(data['race_state'].get('pods', []))} pods")
                    
                    # Send data to visualization panel
                    if visualization_panel is not None:
                        visualization_panel.add_metric(data)
                except queue.Empty:
                    # Check if all processes are still alive
                    if not any(p.is_alive() for p in self.processes):
                        # All processes have finished, exit the thread
                        print("All training processes finished, stopping visualization thread")
                        break
                    # Continue waiting for data
                    continue
        except Exception as e:
            import traceback
            print(f"Error in visualization thread: {e}")
            print(traceback.format_exc())

    @staticmethod
    def extract_position_from_observation(observation):
        """Extract position data from an observation tensor"""
        # Most racing environments include position in the first 2 values of observation
        if observation is not None and len(observation.shape) >= 2 and observation.shape[1] >= 2:
            return observation[0, 0:2].tolist()
        return [0, 0]  # Default position if extraction fails

    @staticmethod
    def _get_checkpoints_from_env(env):
        """Extract checkpoint positions from environment"""
        if hasattr(env, 'get_checkpoints'):
            return env.get_checkpoints()
        elif hasattr(env, 'checkpoints'):
            # For optimized environment
            # Return first batch's checkpoints
            if isinstance(env.checkpoints, torch.Tensor):
                return env.checkpoints[0].cpu().numpy().tolist()
        return []
