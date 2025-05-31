import torch
import numpy as np
from pathlib import Path
import random
from typing import Dict, List, Tuple, Any
import copy
import json
import os
import glob
from datetime import datetime
import threading
import time

from ..models.neural_pod import PodNetwork
from ..environment.optimized_race_env import OptimizedRaceEnvironment
from .trainer import PPOTrainer

class PodLeague:
    """
    League system for self-play training of pod racing agents using OptimizedRaceEnvironment.
    Maintains a pool of agents at different skill levels for more effective training.
    """
    def __init__(self,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 league_size: int = 10,
                 base_save_dir: str = 'league_models',
                 multi_gpu: bool = False,
                 devices: List[int] = None,
                 batch_size: int = 64,
                 use_mixed_precision: bool = True):
        self.device = device
        self.league_size = league_size
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU optimization settings
        self.multi_gpu = multi_gpu
        self.devices = devices if devices else [0]  # Default to first GPU
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision
        
        # Print GPU configuration
        self.print_gpu_config()
        
        # League members (pod teams)
        self.league_members = []
        
        # League metadata
        self.metadata_path = self.base_save_dir / 'league_metadata.json'
        
        # Cache for environments and trainers to avoid recreating them
        self.env_cache = {}
        self.trainer_cache = {}
        
        # Training control
        self.training_lock = threading.Lock()
        self.stop_requested = False
        self.training_threads = []
        
        self.load_or_initialize_league()
    
    def detect_model_format(self, model_dir: Path) -> str:
        """
        Detect the format of models in the given directory.
        
        Returns:
            'standard': pod0.pt, pod1.pt
            'trainer': player0_pod0.pt, player0_pod1.pt, player1_pod0.pt, player1_pod1.pt
            'unknown': No recognizable format found
        """
        # Check for standard league format
        if (model_dir / "pod0.pt").exists() and (model_dir / "pod1.pt").exists():
            return 'standard'
        
        # Check for trainer format
        trainer_files = [
            "player0_pod0.pt", "player0_pod1.pt", 
            "player1_pod0.pt", "player1_pod1.pt"
        ]
        if all((model_dir / f).exists() for f in trainer_files):
            return 'trainer'
        
        # Check for partial trainer format (only player0)
        player0_files = ["player0_pod0.pt", "player0_pod1.pt"]
        if all((model_dir / f).exists() for f in player0_files):
            return 'trainer_partial'
        
        return 'unknown'
    
    def load_model_from_format(self, model_dir: Path, pod_idx: int, format_type: str) -> torch.Tensor:
        """
        Load a model state dict from the specified directory and format.
        
        Args:
            model_dir: Directory containing the models
            pod_idx: Pod index (0 or 1)
            format_type: Format type returned by detect_model_format
            
        Returns:
            Model state dict
        """
        if format_type == 'standard':
            model_path = model_dir / f"pod{pod_idx}.pt"
            
        elif format_type in ['trainer', 'trainer_partial']:
            model_path = model_dir / f"player0_pod{pod_idx}.pt"
            
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return torch.load(model_path, map_location=self.device)
    
    def import_trained_model(self, source_dir: str, member_name: str = None) -> int:
        """
        Import a trained model from PPO trainer or other sources into the league.
        
        Args:
            source_dir: Directory containing the trained models
            member_name: Optional name for the new league member
            
        Returns:
            Index of the newly created league member
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
        
        # Detect the format of the source models
        format_type = self.detect_model_format(source_path)
        if format_type == 'unknown':
            raise ValueError(f"No recognizable model format found in {source_dir}")
        
        print(f"Detected model format: {format_type}")
        
        # Generate new member ID and name
        new_id = len(self.league_members)
        if member_name is None:
            member_name = f"imported_agent_{new_id}"
        
        # Create new member entry
        new_member = {
            'id': new_id,
            'name': member_name,
            'wins': 0,
            'matches': 0,
            'elo': 1200,  # Start with slightly higher ELO since it's a trained model
            'generation': 0,
            'model_path': str(self.base_save_dir / f"member_{new_id}"),
            'imported_from': str(source_path),
            'import_format': format_type
        }
        
        # Create directory for this member
        member_dir = Path(new_member['model_path'])
        member_dir.mkdir(exist_ok=True)
        
        # Load and convert models to league format
        try:
            for pod_idx in range(2):
                # Load the model state dict from the source
                state_dict = self.load_model_from_format(source_path, pod_idx, format_type)
                
                # Save in league format
                torch.save(state_dict, member_dir / f"pod{pod_idx}.pt")
                
            print(f"Successfully imported models for {member_name}")
            
        except Exception as e:
            # Clean up on failure
            if member_dir.exists():
                import shutil
                shutil.rmtree(member_dir)
            raise RuntimeError(f"Failed to import models: {str(e)}")
        
        # Add to league
        self.league_members.append(new_member)
        
        # Save league metadata
        self.save_metadata()
        
        print(f"Added imported member: {new_member['name']} (ELO: {new_member['elo']})")
        return new_id
    
    def export_member_to_trainer_format(self, member_idx: int, export_dir: str):
        """
        Export a league member's models to trainer format for further training.
        
        Args:
            member_idx: Index of the league member to export
            export_dir: Directory to export the models to
        """
        if member_idx >= len(self.league_members):
            raise ValueError(f"Invalid member index: {member_idx}")
        
        member = self.league_members[member_idx]
        source_dir = Path(member['model_path'])
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Copy models to trainer format
        for pod_idx in range(2):
            source_file = source_dir / f"pod{pod_idx}.pt"
            if not source_file.exists():
                raise FileNotFoundError(f"Source model not found: {source_file}")
            
            # Export for both players (for self-play training)
            for player_idx in range(2):
                target_file = export_path / f"player{player_idx}_pod{pod_idx}.pt"
                import shutil
                shutil.copy2(source_file, target_file)
        
        # Export metadata
        metadata = {
            'exported_from_league': True,
            'original_member': member,
            'export_format': 'trainer',
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(export_path / "export_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Exported {member['name']} to {export_path} in trainer format")
    
    def print_gpu_config(self):
        """Print GPU configuration information"""
        print(f"League GPU Configuration:")
        print(f"  Multi-GPU: {self.multi_gpu}")
        print(f"  Devices: {self.devices}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Mixed Precision: {self.use_mixed_precision}")
        
        # Print CUDA device information if available
        if torch.cuda.is_available():
            print(f"  CUDA Devices Available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    Device {i}: {torch.cuda.get_device_name(i)}")
    
    def load_or_initialize_league(self):
        """Load existing league or initialize a new one"""
        if self.metadata_path.exists():
            # Load existing league
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.league_members = metadata['members']
            print(f"Loaded league with {len(self.league_members)} members")
        else:
            # Initialize new league with random members
            print("Initializing new league")
            self.initialize_league()
    
    def initialize_league(self):
        """Initialize a new league with random members"""
        # Create initial league members with random policies
        for i in range(self.league_size):
            member = {
                'id': i,
                'name': f"random_agent_{i}",
                'wins': 0,
                'matches': 0,
                'elo': 1000,  # Initial ELO rating
                'generation': 0,
                'model_path': str(self.base_save_dir / f"member_{i}")
            }
            
            # Create directory for this member
            member_dir = Path(member['model_path'])
            member_dir.mkdir(exist_ok=True)
            
            # Initialize random networks for this member
            for pod_idx in range(2):
                network = PodNetwork(
                    observation_dim=56,
                    hidden_layers=[
                        {'type': 'Linear+ReLU', 'size': 24},
                        {'type': 'Linear+ReLU', 'size': 16}
                    ],
                    policy_hidden_size=12,
                    value_hidden_size=12,
                    action_hidden_size=12
                ).to(self.device)
                torch.save(network.state_dict(), member_dir / f"pod{pod_idx}.pt")
            
            self.league_members.append(member)
        
        # Save league metadata
        self.save_metadata()
    
    def save_metadata(self):
        """Save league metadata to disk"""
        with self.training_lock:
            with open(self.metadata_path, 'w') as f:
                json.dump({'members': self.league_members}, f, indent=2)
    
    def update_elo(self, winner_idx: int, loser_idx: int, k: float = 32):
        """Update ELO ratings after a match"""
        with self.training_lock:
            winner = self.league_members[winner_idx]
            loser = self.league_members[loser_idx]
            
            # Calculate expected scores
            winner_expected = 1 / (1 + 10 ** ((loser['elo'] - winner['elo']) / 400))
            loser_expected = 1 / (1 + 10 ** ((winner['elo'] - loser['elo']) / 400))
            
            # Update ELO ratings
            winner['elo'] += k * (1 - winner_expected)
            loser['elo'] += k * (0 - loser_expected)
            
            # Update win/match counts
            winner['wins'] += 1
            winner['matches'] += 1
            loser['matches'] += 1
            
            # Save updated metadata
            self.save_metadata()
    
    def select_opponents(self, main_agent_idx: int) -> List[int]:
        """Select opponents for training from the league"""
        # Sort league by ELO
        sorted_indices = sorted(
            range(len(self.league_members)), 
            key=lambda i: self.league_members[i]['elo']
        )
        
        # Find position of main agent in sorted list
        main_pos = sorted_indices.index(main_agent_idx)
        
        # Select similar skill opponents
        similar_range = max(1, int(self.league_size * 0.3))
        similar_start = max(0, main_pos - similar_range // 2)
        similar_end = min(len(sorted_indices), similar_start + similar_range)
        similar_opponents = sorted_indices[similar_start:similar_end]
        
        # Select weaker opponents
        weaker_range = max(1, int(self.league_size * 0.1))
        weaker_start = max(0, main_pos - weaker_range * 2)
        weaker_end = max(0, main_pos - 1)
        weaker_opponents = sorted_indices[weaker_start:weaker_end]
        
        # Select stronger opponents
        stronger_range = max(1, int(self.league_size * 0.1))
        stronger_start = min(len(sorted_indices), main_pos + 1)
        stronger_end = min(len(sorted_indices), main_pos + stronger_range * 2)
        stronger_opponents = sorted_indices[stronger_start:stronger_end]
        
        # Combine and remove main agent if present
        all_opponents = similar_opponents + weaker_opponents + stronger_opponents
        all_opponents = [idx for idx in all_opponents if idx != main_agent_idx]
        
        # Ensure we have at least one opponent
        if not all_opponents:
            all_opponents = [i for i in range(len(self.league_members)) if i != main_agent_idx]
        
        return all_opponents
    
    def get_environment(self, device_id=None):
        """Get or create a race environment for the specified device"""
        # Determine which device to use
        if device_id is None:
            if self.multi_gpu and len(self.devices) > 0:
                device_id = self.devices[0]
            else:
                device_id = 0 if torch.cuda.is_available() else -1
        
        # Create device string
        device_str = f"cuda:{device_id}" if device_id >= 0 else "cpu"
        
        # Check if we already have an environment for this device
        if device_str in self.env_cache:
            return self.env_cache[device_str]
        
        # Create a new environment
        device = torch.device(device_str)
        
        # Always use OptimizedRaceEnvironment
        env = OptimizedRaceEnvironment(
            batch_size=self.batch_size,
            device=device
        )
        
        # Cache the environment
        self.env_cache[device_str] = env
        
        return env
    
    def get_trainer(self, env, device_id=None):
        """Get or create a trainer for the specified device and environment"""
        # Determine which device to use
        if device_id is None:
            if self.multi_gpu and len(self.devices) > 0:
                device_id = self.devices[0]
            else:
                device_id = 0 if torch.cuda.is_available() else -1
        
        # Create device string
        device_str = f"cuda:{device_id}" if device_id >= 0 else "cpu"
        
        # Check if we already have a trainer for this device
        if device_str in self.trainer_cache:
            # Update the environment if needed
            trainer = self.trainer_cache[device_str]
            trainer.env = env
            return trainer
        
        # Create a new trainer
        device = torch.device(device_str)
        
        # Determine which devices to use for multi-GPU
        devices_to_use = None
        if self.multi_gpu:
            devices_to_use = self.devices
        
        # Network configuration for the updated PodNetwork
        network_config = {
            'observation_dim': 56,
            'hidden_layers': [
                {'type': 'Linear+ReLU', 'size': 24},
                {'type': 'Linear+ReLU', 'size': 16}
            ],
            'policy_hidden_size': 12,
            'value_hidden_size': 12,
            'action_hidden_size': 12
        }
        
        # Create the trainer
        trainer = PPOTrainer(
            env=env,
            batch_size=self.batch_size,
            device=device,
            multi_gpu=self.multi_gpu,
            devices=devices_to_use,
            use_mixed_precision=self.use_mixed_precision,
            network_config=network_config
        )
        
        # Cache the trainer
        self.trainer_cache[device_str] = trainer
        
        return trainer
    
    def train_member(self, 
                        member_idx: int, 
                        num_iterations: int = 100,
                        steps_per_iteration: int = 200,
                        batch_size: int = None,
                        device_id: int = None):
        """Train a specific league member using the updated trainer"""
        try:
            if batch_size is None:
                batch_size = self.batch_size
                
            member = self.league_members[member_idx]
            print(f"Training member {member['name']} (ELO: {member['elo']:.1f})")
            
            # Select opponents
            opponent_indices = self.select_opponents(member_idx)
            print(f"Selected {len(opponent_indices)} opponents for training")
            
            # Determine device
            if device_id is None and self.multi_gpu:
                device_id = self.devices[member_idx % len(self.devices)]
            
            device = torch.device(f"cuda:{device_id}" if device_id is not None else "cpu")
            print(f"Training on device: {device}")
            
            # Create environment
            env = OptimizedRaceEnvironment(
                batch_size=batch_size,
                device=device
            )
            
            # Create trainer with proper network_config
            network_config = {
                'observation_dim': 56,
                'hidden_layers': [
                    {'type': 'Linear+ReLU', 'size': 24},
                    {'type': 'Linear+ReLU', 'size': 16}
                ],
                'policy_hidden_size': 12,
                'value_hidden_size': 12,
                'action_hidden_size': 12
            }
            
            trainer = PPOTrainer(
                env=env,
                batch_size=batch_size,
                device=device,
                multi_gpu=False,
                devices=None,
                use_mixed_precision=self.use_mixed_precision,
                network_config=network_config
            )
            
            # Load member's models
            member_dir = Path(member['model_path'])
            for pod_idx in range(2):
                pod_key = f"player0_pod{pod_idx}"
                model_path = member_dir / f"pod{pod_idx}.pt"
                if model_path.exists():
                    trainer.pod_networks[pod_key].load_state_dict(
                        torch.load(model_path, map_location=device)
                    )
                else:
                    print(f"Warning: Model file not found: {model_path}")
            
            # Train against each opponent
            for opponent_idx in opponent_indices:
                if self.stop_requested:
                    break
                    
                opponent = self.league_members[opponent_idx]
                print(f"Training against {opponent['name']} (ELO: {opponent['elo']:.1f})")
                
                # Load opponent's models
                opponent_dir = Path(opponent['model_path'])
                for pod_idx in range(2):
                    pod_key = f"player1_pod{pod_idx}"
                    model_path = opponent_dir / f"pod{pod_idx}.pt"
                    if model_path.exists():
                        trainer.pod_networks[pod_key].load_state_dict(
                            torch.load(model_path, map_location=device)
                        )
                    else:
                        print(f"Warning: Model file not found: {model_path}")
                
                # Train for some iterations
                iterations_per_opponent = max(1, num_iterations // len(opponent_indices))
                trainer.train(
                    num_iterations=iterations_per_opponent,
                    steps_per_iteration=steps_per_iteration,
                    save_interval=iterations_per_opponent,
                    save_dir=str(member_dir / f"training_vs_{opponent['id']}")
                )
            
            # Save final models
            for pod_idx in range(2):
                pod_key = f"player0_pod{pod_idx}"
                torch.save(
                    trainer.pod_networks[pod_key].state_dict(),
                    member_dir / f"pod{pod_idx}.pt"
                )
            
            # Update member metadata
            with self.training_lock:
                member['generation'] += 1
                self.save_metadata()
            
            print(f"Completed training for {member['name']}")
            
        except Exception as e:
            print(f"Error in train_member for member {member_idx}: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def train_all_members_parallel(self, 
                                 num_iterations: int = 100,
                                 steps_per_iteration: int = 200,
                                 max_concurrent_trainings: int = None):
        """
        Train all league members in parallel using multiple threads/GPUs
        
        Args:
            num_iterations: Number of training iterations per member
            steps_per_iteration: Steps per training iteration
            max_concurrent_trainings: Maximum number of concurrent training threads
        """
        print(f"Starting parallel training for {len(self.league_members)} members")
        
        # Determine maximum concurrent trainings
        if max_concurrent_trainings is None:
            if self.multi_gpu:
                max_concurrent_trainings = len(self.devices)
            else:
                max_concurrent_trainings = 1
        
        print(f"Using {max_concurrent_trainings} concurrent training threads")
        
        # Reset stop flag
        self.stop_requested = False
        
        # Create training queue with all member indices
        training_queue = list(range(len(self.league_members)))
        random.shuffle(training_queue)  # Randomize training order
        
        # Thread management
        self.training_threads = []
        active_threads = []
        completed_members = []
        failed_members = []
        
        def training_worker(member_idx: int, device_id: int = None):
            """Worker function for training a single member"""
            try:
                self.train_member(
                    member_idx=member_idx,
                    num_iterations=num_iterations,
                    steps_per_iteration=steps_per_iteration,
                    device_id=device_id
                )
                with self.training_lock:
                    completed_members.append(member_idx)
                    print(f"Completed training for member {member_idx}")
            except Exception as e:
                with self.training_lock:
                    failed_members.append((member_idx, str(e)))
                    print(f"Failed training for member {member_idx}: {str(e)}")
        
        # Start training threads
        queue_index = 0
        while queue_index < len(training_queue) and not self.stop_requested:
            # Clean up completed threads
            active_threads = [t for t in active_threads if t.is_alive()]
            
            # Start new threads if we have capacity
            while len(active_threads) < max_concurrent_trainings and queue_index < len(training_queue):
                if self.stop_requested:
                    break
                
                member_idx = training_queue[queue_index]
                
                # Assign device for multi-GPU training
                device_id = None
                if self.multi_gpu:
                    device_id = self.devices[queue_index % len(self.devices)]
                
                # Create and start training thread
                thread = threading.Thread(
                    target=training_worker,
                    args=(member_idx, device_id),
                    name=f"TrainingThread-Member{member_idx}"
                )
                thread.daemon = True
                thread.start()
                
                active_threads.append(thread)
                self.training_threads.append(thread)
                
                print(f"Started training thread for member {member_idx} on device {device_id}")
                queue_index += 1
            
            # Wait a bit before checking for completed threads
            time.sleep(1.0)
        
        # Wait for all remaining threads to complete
        print("Waiting for all training threads to complete...")
        for thread in active_threads:
            thread.join()
        
        # Print summary
        print(f"\nParallel training completed:")
        print(f"  Successfully trained: {len(completed_members)} members")
        print(f"  Failed: {len(failed_members)} members")
        
        if failed_members:
            print("Failed members:")
            for member_idx, error in failed_members:
                member_name = self.league_members[member_idx]['name']
                print(f"  {member_name} (ID: {member_idx}): {error}")
        
        # Clear training threads list
        self.training_threads = []
        
        return len(completed_members), len(failed_members)
    
    def stop_all_training(self):
        """Stop all ongoing training threads"""
        print("Stopping all training threads...")
        self.stop_requested = True
        
        # Wait for all threads to finish
        for thread in self.training_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)  # Wait up to 5 seconds per thread
        
        print("All training threads stopped")
    
    def evaluate_match(self, 
                    member1_idx: int, 
                    member2_idx: int,
                    num_races: int = 10,
                    device_id: int = None) -> Tuple[int, int]:
        """
        Evaluate a match between two league members
        
        Returns:
            Tuple of (member1_wins, member2_wins)
        """
        try:
            # Determine which device to use
            if device_id is None and self.multi_gpu:
                # Use a round-robin approach to distribute matches
                match_id = (member1_idx * self.league_size + member2_idx) % len(self.devices)
                device_id = self.devices[match_id]
                
            # Get device
            if device_id is not None:
                device = torch.device(f"cuda:{device_id}")
            else:
                device = self.device
                
            member1 = self.league_members[member1_idx]
            member2 = self.league_members[member2_idx]
            
            print(f"Evaluating: {member1['name']} vs {member2['name']} on {device}")
            
            # Create a new environment specifically for evaluation
            env = OptimizedRaceEnvironment(
                batch_size=num_races,
                device=device
            )
            
            # Load models for both members
            pod_networks = {}
            
            # Network configuration
            network_config = {
                'observation_dim': 56,
                'hidden_layers': [
                    {'type': 'Linear+ReLU', 'size': 24},
                    {'type': 'Linear+ReLU', 'size': 16}
                ],
                'policy_hidden_size': 12,
                'value_hidden_size': 12,
                'action_hidden_size': 12
            }
            
            # Load member1's models (player 0)
            member1_dir = Path(member1['model_path'])
            for pod_idx in range(2):
                pod_key = f"player0_pod{pod_idx}"
                model_path = member1_dir / f"pod{pod_idx}.pt"
                pod_networks[pod_key] = PodNetwork(**network_config).to(device)
                if model_path.exists():
                    pod_networks[pod_key].load_state_dict(
                        torch.load(model_path, map_location=device)
                    )
                else:
                    print(f"Warning: Model file not found: {model_path}")
            
            # Load member2's models (player 1)
            member2_dir = Path(member2['model_path'])
            for pod_idx in range(2):
                pod_key = f"player1_pod{pod_idx}"
                model_path = member2_dir / f"pod{pod_idx}.pt"
                pod_networks[pod_key] = PodNetwork(**network_config).to(device)
                if model_path.exists():
                    pod_networks[pod_key].load_state_dict(
                        torch.load(model_path, map_location=device)
                    )
                else:
                    print(f"Warning: Model file not found: {model_path}")
            
            # Run evaluation races
            print("Resetting environment...")
            observations = env.reset()
            
            if observations is None:
                raise RuntimeError("Environment reset returned None")
            
            print(f"Environment reset successful, got observations: {type(observations)}")
            if isinstance(observations, dict):
                print(f"Observation keys: {list(observations.keys())}")
            
            dones = torch.zeros(num_races, dtype=torch.bool, device=device)
            
            # Track which player won each race
            winners = torch.zeros(num_races, dtype=torch.long, device=device)
            
            # Run races until all are done
            max_steps = 600  # Safety limit
            for step in range(max_steps):
                if dones.all():
                    break
                    
                # Get actions from all pod networks
                actions = {}
                
                with torch.no_grad():
                    for pod_key, network in pod_networks.items():
                        # Check if observation exists for this pod
                        if observations is None:
                            raise RuntimeError(f"Observations is None at step {step}")
                        
                        if not isinstance(observations, dict):
                            raise RuntimeError(f"Observations is not a dict: {type(observations)}")
                        
                        if pod_key not in observations:
                            raise RuntimeError(f"Pod key '{pod_key}' not found in observations. Available keys: {list(observations.keys())}")
                        
                        obs = observations[pod_key]
                        if obs is None:
                            raise RuntimeError(f"Observation for {pod_key} is None")
                        
                        action = network.get_actions(obs, deterministic=True)
                        actions[pod_key] = action
                
                # Step the environment
                step_result = env.step(actions)
                
                if step_result is None:
                    raise RuntimeError(f"Environment step returned None at step {step}")
                
                if len(step_result) != 4:
                    raise RuntimeError(f"Environment step returned {len(step_result)} items, expected 4")
                
                observations, rewards, new_dones, info = step_result
                
                if observations is None:
                    raise RuntimeError(f"Observations is None after step {step}")
                
                if info is None:
                    raise RuntimeError(f"Info is None after step {step}")
                
                # Check for newly completed races
                newly_done = new_dones & ~dones
                if newly_done.any():
                    # Determine winners for newly completed races
                    for b in range(num_races):
                        if newly_done[b]:
                            # Check if checkpoint_progress exists in info
                            if "checkpoint_progress" not in info:
                                raise RuntimeError(f"'checkpoint_progress' not found in info. Available keys: {list(info.keys()) if isinstance(info, dict) else 'info is not a dict'}")
                            
                            checkpoint_progress = info["checkpoint_progress"]
                            if checkpoint_progress is None:
                                raise RuntimeError("checkpoint_progress is None")
                            
                            # Check which player's pod completed the race
                            player0_progress = max(
                                checkpoint_progress[0][b].item(),
                                checkpoint_progress[1][b].item()
                            )
                            player1_progress = max(
                                checkpoint_progress[2][b].item(),
                                checkpoint_progress[3][b].item()
                            )
                            
                            # Set winner (0 for player0, 1 for player1)
                            winners[b] = 0 if player0_progress > player1_progress else 1
                
                dones = new_dones
            
            # Count wins
            member1_wins = (winners == 0).sum().item()
            member2_wins = (winners == 1).sum().item()
            
            print(f"Results: {member1['name']} won {member1_wins}, {member2['name']} won {member2_wins}")
            
            # Update ELO if there's a clear winner
            if member1_wins > member2_wins:
                self.update_elo(member1_idx, member2_idx)
            elif member2_wins > member1_wins:
                self.update_elo(member2_idx, member1_idx)
            
            return member1_wins, member2_wins
            
        except Exception as e:
            print(f"Error in evaluate_match: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def run_tournament(self, matches_per_pair: int = 1):
        """Run a tournament between all league members with GPU optimization"""
        print("Starting league tournament")
        
        # Create all possible pairings
        pairings = [(i, j) for i in range(len(self.league_members)) 
                for j in range(len(self.league_members)) if i != j]
        
        # Shuffle pairings
        random.shuffle(pairings)
        
        # Distribute matches across GPUs if multi-GPU is enabled
        if self.multi_gpu and len(self.devices) > 1:
            # Group pairings by GPU
            gpu_pairings = [[] for _ in range(len(self.devices))]
            for i, pairing in enumerate(pairings):
                gpu_idx = i % len(self.devices)
                gpu_pairings[gpu_idx].append(pairing)
            
            # Run matches on each GPU sequentially to avoid memory issues
            for gpu_idx, device_id in enumerate(self.devices):
                for member1_idx, member2_idx in gpu_pairings[gpu_idx]:
                    for _ in range(matches_per_pair):
                        try:
                            self.evaluate_match(member1_idx, member2_idx, device_id=device_id)
                        except Exception as e:
                            print(f"Error in match {member1_idx} vs {member2_idx} on device {device_id}: {e}")
        else:
            # Run matches sequentially
            for member1_idx, member2_idx in pairings:
                for _ in range(matches_per_pair):
                    try:
                        self.evaluate_match(member1_idx, member2_idx)
                    except Exception as e:
                        print(f"Error in match {member1_idx} vs {member2_idx}: {e}")
        
        # Sort and display results
        sorted_members = sorted(
            self.league_members,
            key=lambda m: m['elo'],
            reverse=True
        )
        
        print("\nTournament Results:")
        for i, member in enumerate(sorted_members):
            win_rate = member['wins'] / max(1, member['matches'])
            print(f"{i+1}. {member['name']} - ELO: {member['elo']:.1f}, "
                f"Win Rate: {win_rate:.2f} ({member['wins']}/{member['matches']})")
    
    def add_new_member(self, base_member_idx: int = None):
        """Add a new member to the league, optionally based on an existing member"""
        # Generate new member ID
        new_id = len(self.league_members)
        
        # Create new member entry
        new_member = {
            'id': new_id,
            'name': f"agent_{new_id}",
            'wins': 0,
            'matches': 0,
            'elo': 1000,  # Initial ELO rating
            'generation': 0,
            'model_path': str(self.base_save_dir / f"member_{new_id}")
        }
        
        # Create directory for this member
        member_dir = Path(new_member['model_path'])
        member_dir.mkdir(exist_ok=True)
        
        # Network configuration
        network_config = {
            'observation_dim': 56,
            'hidden_layers': [
                {'type': 'Linear+ReLU', 'size': 24},
                {'type': 'Linear+ReLU', 'size': 16}
            ],
            'policy_hidden_size': 12,
            'value_hidden_size': 12,
            'action_hidden_size': 12
        }
        
        # Initialize networks for this member
        if base_member_idx is not None:
            # Copy and mutate from base member
            base_member = self.league_members[base_member_idx]
            base_dir = Path(base_member['model_path'])
            
            for pod_idx in range(2):
                # Load base network
                network = PodNetwork(**network_config).to(self.device)
                base_path = base_dir / f"pod{pod_idx}.pt"
                if base_path.exists():
                    network.load_state_dict(torch.load(base_path, map_location=self.device))
                
                # Apply mutation (small random changes to weights)
                with torch.no_grad():
                    for param in network.parameters():
                        noise = torch.randn_like(param) * 0.02  # Small noise
                        param.add_(noise)
                
                # Save mutated network
                torch.save(network.state_dict(), member_dir / f"pod{pod_idx}.pt")
        else:
            # Initialize random networks
            for pod_idx in range(2):
                network = PodNetwork(**network_config).to(self.device)
                torch.save(network.state_dict(), member_dir / f"pod{pod_idx}.pt")
        
        # Add to league
        self.league_members.append(new_member)
        
        # Save league metadata
        self.save_metadata()
        
        print(f"Added new member: {new_member['name']}")
        return new_id
    
    def evolve_league(self, 
                    iterations: int = 10, 
                    training_iterations: int = 100,
                    tournament_frequency: int = 2,
                    parallel_training: bool = True):
        """Evolve the league over multiple iterations with parallel training support"""
        # Flag to track if evolution should be stopped
        self.stop_requested = False
        
        for iteration in range(1, iterations + 1):
            if self.stop_requested:
                print("Evolution stopped by user request")
                break
                
            print(f"\n=== League Evolution Iteration {iteration}/{iterations} ===")
            
            # Train all members
            if parallel_training:
                print("Starting parallel training for all members...")
                completed, failed = self.train_all_members_parallel(
                    num_iterations=training_iterations
                )
                print(f"Parallel training completed: {completed} successful, {failed} failed")
                
                if self.stop_requested:
                    print("Evolution stopped by user request")
                    break
            else:
                # Train sequentially (original behavior)
                for member_idx in range(len(self.league_members)):
                    if self.stop_requested:
                        print("Evolution stopped by user request")
                        break
                    try:
                        self.train_member(
                            member_idx=member_idx,
                            num_iterations=training_iterations
                        )
                    except Exception as e:
                        print(f"Error training member {member_idx}: {e}")
            
            # Run tournament periodically
            if iteration % tournament_frequency == 0:
                try:
                    self.run_tournament()
                    
                    # Replace worst performer with a mutation of a top performer
                    if len(self.league_members) >= 4:  # Need enough members to make this meaningful
                        # Sort by ELO
                        sorted_indices = sorted(
                            range(len(self.league_members)), 
                            key=lambda i: self.league_members[i]['elo'],
                            reverse=True
                        )
                        
                        # Get the actual member objects before modifying the list
                        top_member = self.league_members[sorted_indices[0]]
                        worst_member = self.league_members[sorted_indices[-1]]
                        worst_idx = sorted_indices[-1]
                        
                        print(f"Replacing {worst_member['name']} "
                            f"with a mutation of {top_member['name']}")
                        
                        # Save the top member's model path before removing anything
                        top_member_path = top_member['model_path']
                        
                        # Remove worst member's files
                        import shutil
                        worst_member_dir = Path(worst_member['model_path'])
                        if worst_member_dir.exists():
                            shutil.rmtree(worst_member_dir)
                        
                        # Remove worst member from list
                        self.league_members.pop(worst_idx)
                        
                        # Create new member based on top performer
                        # We need to find the top member's new index after removal
                        new_top_idx = None
                        for i, member in enumerate(self.league_members):
                            if member['model_path'] == top_member_path:
                                new_top_idx = i
                                break
                        
                        if new_top_idx is not None:
                            self.add_new_member(base_member_idx=new_top_idx)
                        else:
                            # Fallback: add a random member if we can't find the top performer
                            print("Warning: Could not find top performer after removal, adding random member")
                            self.add_new_member()
                            
                except Exception as e:
                    print(f"Error during tournament in iteration {iteration}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Add a completely new random member occasionally
            if iteration % 3 == 0:
                try:
                    self.add_new_member()
                except Exception as e:
                    print(f"Error adding new member in iteration {iteration}: {e}")
        
        # Final tournament
        if not self.stop_requested:
            try:
                print("\n=== Final League Tournament ===")
                self.run_tournament(matches_per_pair=2)
                
                # Print final league standings
                sorted_members = sorted(
                    self.league_members,
                    key=lambda m: m['elo'],
                    reverse=True
                )
                
                print("\nFinal League Standings:")
                for i, member in enumerate(sorted_members):
                    win_rate = member['wins'] / max(1, member['matches'])
                    print(f"{i+1}. {member['name']} - ELO: {member['elo']:.1f}, "
                        f"Win Rate: {win_rate:.2f} ({member['wins']}/{member['matches']}), "
                        f"Generation: {member['generation']}")
                
                # Save best model to a special location
                if sorted_members:
                    best_member = sorted_members[0]
                    best_dir = self.base_save_dir / "best_model"
                    best_dir.mkdir(exist_ok=True)
                    
                    # Copy best member's models
                    source_dir = Path(best_member['model_path'])
                    for pod_idx in range(2):
                        source_path = source_dir / f"pod{pod_idx}.pt"
                        if source_path.exists():
                            torch.save(
                                torch.load(source_path, map_location=self.device),
                                best_dir / f"pod{pod_idx}.pt"
                            )
                    
                    # Save metadata about best model
                    with open(best_dir / "metadata.json", "w") as f:
                        json.dump({
                            "name": best_member['name'],
                            "elo": best_member['elo'],
                            "wins": best_member['wins'],
                            "matches": best_member['matches'],
                            "generation": best_member['generation'],
                            "source_path": best_member['model_path']
                        }, f, indent=2)
                    
                    print(f"\nBest model saved to {best_dir}")
                else:
                    print("Warning: No members found for final standings")
                    
            except Exception as e:
                print(f"Error during final tournament: {e}")
                import traceback
                traceback.print_exc()
