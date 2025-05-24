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


from ..environment.race_env import RaceEnvironment
from ..environment.optimized_race_env import OptimizedRaceEnvironment
from .trainer import PPOTrainer
from .optimized_trainer import OptimizedPPOTrainer

class ParallelPPOTrainer:
    """
    Parallelized PPO Trainer that runs multiple environments across available GPUs
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

        # Print trainer args for debugging
        print(f"ParallelPPOTrainer initialized with args: {self.trainer_args}")
        print(f"Training iterations: {self.trainer_args.get('num_iterations', 'NOT FOUND')}")
        print(f"Steps per iteration: {self.trainer_args.get('steps_per_iteration', 'NOT FOUND')}")
        print(f"Using devices: {self.devices}")
        print(f"Number of GPUs: {self.num_gpus}")
        print(f"Environments per GPU: {self.envs_per_gpu}")
        
        # Calculate total number of environments
        self.total_envs = self.num_gpus * self.envs_per_gpu
        print(f"Total environments: {self.total_envs}")
    
    @staticmethod
    def _worker(gpu_id, env_indices, env_config, trainer_args, total_envs, envs_per_gpu, use_mixed_precision, shared_results, vis_queue=None):
        """Worker process that runs multiple environments on a single GPU with optional visualization"""
        try:
            print(f"Worker process for GPU {gpu_id} starting...")
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
            
            # Reduce learning rate to improve stability
            if 'learning_rate' in trainer_init_args:
                original_lr = trainer_init_args['learning_rate']
                trainer_init_args['learning_rate'] = original_lr * 0.5  # Reduce learning rate by half
                print(f"Reducing learning rate from {original_lr} to {trainer_init_args['learning_rate']} for stability")
            
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
                'steps_per_iteration': trainer_args.get('steps_per_iteration', 128),
                'save_interval': trainer_args.get('save_interval', 50),
                'save_dir': trainer_args.get('save_dir', 'models')
            }

            print(f"Worker on GPU {gpu_id} training with: iterations={training_args['num_iterations']}, steps={training_args['steps_per_iteration']}")
            
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
            def parallel_collect_trajectories(num_steps=128):
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
                        if vis_queue is not None and step % 5 == 0:
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
                        if dones.all():
                            all_observations[env_idx] = env.reset()
                        
                        # Ensure synchronization between operations
                        if device.type == 'cuda':
                            torch.cuda.synchronize(device)       

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

            # Override update_policy to return loss values and handle NaN
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

            for iteration in range(1, training_args['num_iterations'] + 1):
                start_time = time.time()
                
                # Adjust learning rate based on iteration (simple linear decay)
                progress = iteration / training_args['num_iterations']
                current_lr = max(min_lr, initial_lr * (1 - 0.9 * progress))  # Linear decay to 10% of initial
                
                # Update learning rate in optimizer
                for pod_key, optimizer in trainer.optimizers.items():
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                
                try:
                    # Collect trajectories
                    trainer.collect_trajectories(training_args['steps_per_iteration'])
                    
                    # Compute returns and advantages
                    trainer.compute_returns_and_advantages()
                    
                    # Update policy
                    policy_loss, value_loss = trainer.update_policy()
                    
                    # Check for NaN in network parameters after update
                    check_and_fix_network_parameters()
                    
                    # Calculate average reward
                    avg_reward = 0
                    for pod_key in trainer.pod_networks.keys():
                        if 'rewards' in trainer.storage[pod_key]:
                            rewards = torch.cat([r for r in trainer.storage[pod_key]['rewards']])
                            avg_reward += rewards.mean().item()
                    
                    avg_reward /= len(trainer.pod_networks)
                    
                    # Reset storage for next iteration
                    trainer.reset_storage()
                    
                    # Log progress
                    elapsed_time = time.time() - start_time
                    print(f"GPU {gpu_id} - Iteration {iteration}/{training_args['num_iterations']} completed in {elapsed_time:.2f}s, "
                        f"avg_reward={avg_reward:.4f}, policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}, lr={current_lr:.6f}")
                    
                    # Log metrics to file
                    with open(log_file, 'a') as f:
                        f.write(f"iteration={iteration},reward={avg_reward:.4f},policy_loss={policy_loss:.4f},"
                                f"value_loss={value_loss:.4f},time={elapsed_time:.2f},lr={current_lr:.6f}\n")
                    
                    # Send metrics to visualization if enabled
                    if vis_queue is not None:
                        metrics = {
                            'iteration': iteration,
                            'reward': avg_reward,
                            'policy_loss': policy_loss,
                            'value_loss': value_loss,
                            'learning_rate': current_lr,
                            'worker_id': f"gpu_{gpu_id}"  # Include worker ID to differentiate metrics
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
                                # Save with GPU ID in filename to avoid conflicts
                                torch.save(network.state_dict(), save_dir / f"{pod_key}_gpu{gpu_id}_iter{iteration}.pt")
                                # Also save as the latest model
                                torch.save(network.state_dict(), save_dir / f"{pod_key}_gpu{gpu_id}.pt")
                                print(f"GPU {gpu_id} - Models saved at iteration {iteration}")
                            else:
                                print(f"GPU {gpu_id} - Not saving model for {pod_key} due to NaN parameters")
                    
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

            # Store results in shared dictionary
            shared_results[f'gpu_{gpu_id}'] = {
                'completed': True,
                'env_indices': env_indices
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
        """Start parallel training across GPUs with improved error handling"""
        print("Starting parallel training...")
        print(f"Training parameters: num_iterations={self.trainer_args.get('num_iterations')}, steps_per_iteration={self.trainer_args.get('steps_per_iteration')}")
        
        # Create environment indices for each GPU
        all_env_indices = np.arange(self.total_envs).reshape(self.num_gpus, self.envs_per_gpu)
        
        # Create a manager for sharing results between processes
        manager = mp.Manager()
        shared_results = manager.dict()

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
            
            # Pass only serializable data to the worker process
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
                    vis_queue                    # Shared queue for visualization
                )
            )
            self.processes.append(p)
            p.start()

        # Wait for all processes to finish
        try:
            # Monitor processes and periodically check if they're alive
            while any(p.is_alive() for p in self.processes):
                time.sleep(1.0)
                
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
            for p in self.processes:
                p.join(timeout=5.0)
            
            # Stop visualization connector if it was started
            if vis_connector:
                print("Stopping visualization connector...")
                vis_connector.stop()
        
        print("All worker processes completed")
        
        # Check for errors in any of the processes
        errors = []
        for key, result in shared_results.items():
            if not result.get('completed', False):
                errors.append(f"{key}: {result.get('error', 'Unknown error')}")
        
        if errors:
            error_msg = "Errors occurred in worker processes:\n" + "\n".join(errors)
            print(error_msg)
            raise RuntimeError(error_msg)
        
        print("Parallel training completed successfully")
        return True

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
                        print(f"Received metric data: iteration={data['iteration']}, reward={data.get('reward', 0):.4f}")
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