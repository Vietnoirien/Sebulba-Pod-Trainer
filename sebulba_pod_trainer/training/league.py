import torch
import numpy as np
from pathlib import Path
import random
from typing import Dict, List, Tuple, Any
import copy
import json
import os

from ..models.neural_pod import PodNetwork
from ..environment.race_env import RaceEnvironment
from ..environment.optimized_race_env import OptimizedRaceEnvironment
from .trainer import PPOTrainer
from .optimized_trainer import OptimizedPPOTrainer

class PodLeague:
    """
    League system for self-play training of pod racing agents.
    Maintains a pool of agents at different skill levels for more effective training.
    """
    def __init__(self,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 league_size: int = 10,
                 base_save_dir: str = 'league_models',
                 multi_gpu: bool = False,
                 devices: List[int] = None,
                 use_optimized_env: bool = True,
                 use_optimized_trainer: bool = True,
                 batch_size: int = 64,
                 use_mixed_precision: bool = True):
        self.device = device
        self.league_size = league_size
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU optimization settings
        self.multi_gpu = multi_gpu
        self.devices = devices if devices else [0]  # Default to first GPU
        self.use_optimized_env = use_optimized_env
        self.use_optimized_trainer = use_optimized_trainer
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision
        
        # Print GPU configuration
        self.print_gpu_config()
        
        # League members (pod teams)
        self.league_members = []
        
        # League metadata
        self.metadata_path = self.base_save_dir / 'league_metadata.json'
        self.load_or_initialize_league()
        
        # Cache for environments and trainers to avoid recreating them
        self.env_cache = {}
        self.trainer_cache = {}
    
    def print_gpu_config(self):
        """Print GPU configuration information"""
        print(f"League GPU Configuration:")
        print(f"  Multi-GPU: {self.multi_gpu}")
        print(f"  Devices: {self.devices}")
        print(f"  Using Optimized Environment: {self.use_optimized_env}")
        print(f"  Using Optimized Trainer: {self.use_optimized_trainer}")
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
                network = PodNetwork().to(self.device)
                torch.save(network.state_dict(), member_dir / f"pod{pod_idx}.pt")
            
            self.league_members.append(member)
        
        # Save league metadata
        self.save_metadata()
    
    def save_metadata(self):
        """Save league metadata to disk"""
        with open(self.metadata_path, 'w') as f:
            json.dump({'members': self.league_members}, f, indent=2)
    
    def update_elo(self, winner_idx: int, loser_idx: int, k: float = 32):
        """Update ELO ratings after a match"""
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
        
        if self.use_optimized_env:
            env = OptimizedRaceEnvironment(
                batch_size=self.batch_size,
                device=device
            )
        else:
            env = RaceEnvironment(
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
        
        # Create the appropriate trainer
        if self.use_optimized_trainer:
            trainer = OptimizedPPOTrainer(
                env=env,
                batch_size=self.batch_size,
                device=device,
                multi_gpu=self.multi_gpu,
                devices=devices_to_use,
                use_mixed_precision=self.use_mixed_precision
            )
        else:
            trainer = PPOTrainer(
                env=env,
                batch_size=self.batch_size,
                device=device,
                multi_gpu=self.multi_gpu,
                devices=devices_to_use,
                use_mixed_precision=self.use_mixed_precision
            )
        
        # Cache the trainer
        self.trainer_cache[device_str] = trainer
        
        return trainer
    
    def train_member(self, 
                    member_idx: int, 
                    num_iterations: int = 100,
                    steps_per_iteration: int = 128,
                    batch_size: int = None,
                    device_id: int = None):
        """Train a specific league member against selected opponents"""
        if batch_size is None:
            batch_size = self.batch_size
            
        member = self.league_members[member_idx]
        print(f"Training member {member['name']} (ELO: {member['elo']:.1f})")
        
        # Select opponents
        opponent_indices = self.select_opponents(member_idx)
        print(f"Selected {len(opponent_indices)} opponents for training")
        
        # Determine which device to use
        if device_id is None and self.multi_gpu:
            # Distribute members across available GPUs in a round-robin fashion
            device_id = self.devices[member_idx % len(self.devices)]
        
        # Create environment and trainer with the specific device
        if device_id is not None:
            device = torch.device(f"cuda:{device_id}")
        else:
            device = self.device
            
        # Create a new environment for this training session
        if self.use_optimized_env:
            env = OptimizedRaceEnvironment(
                batch_size=batch_size,
                device=device
            )
        else:
            env = RaceEnvironment(
                batch_size=batch_size,
                device=device
            )
        
        # Create a new trainer for this training session
        if self.use_optimized_trainer:
            trainer = OptimizedPPOTrainer(
                env=env,
                batch_size=batch_size,
                device=device,
                multi_gpu=False,  # Set to False to avoid device conflicts
                devices=None,
                use_mixed_precision=self.use_mixed_precision
            )
        else:
            trainer = PPOTrainer(
                env=env,
                batch_size=batch_size,
                device=device,
                multi_gpu=False,  # Set to False to avoid device conflicts
                devices=None,
                use_mixed_precision=self.use_mixed_precision
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
        
        # Train against each opponent
        for opponent_idx in opponent_indices:
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
            
            # Train for some iterations
            iterations_per_opponent = max(1, num_iterations // len(opponent_indices))
            trainer.train(
                num_iterations=iterations_per_opponent,
                steps_per_iteration=steps_per_iteration,
                save_interval=iterations_per_opponent,  # Only save at the end
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
        member['generation'] += 1
        self.save_metadata()
        
        print(f"Completed training for {member['name']}")
    
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
        
        # Create a new environment specifically for evaluation with the correct batch size
        # Don't use the cached environment to avoid batch size issues
        if self.use_optimized_env:
            env = OptimizedRaceEnvironment(
                batch_size=num_races,  # Use num_races as batch size for evaluation
                device=device
            )
        else:
            env = RaceEnvironment(
                batch_size=num_races,  # Use num_races as batch size for evaluation
                device=device
            )
        
        # Load models for both members
        pod_networks = {}
        
        # Load member1's models (player 0)
        member1_dir = Path(member1['model_path'])
        for pod_idx in range(2):
            pod_key = f"player0_pod{pod_idx}"
            model_path = member1_dir / f"pod{pod_idx}.pt"
            pod_networks[pod_key] = PodNetwork().to(device)
            if model_path.exists():
                pod_networks[pod_key].load_state_dict(
                    torch.load(model_path, map_location=device)
                )
        
        # Load member2's models (player 1)
        member2_dir = Path(member2['model_path'])
        for pod_idx in range(2):
            pod_key = f"player1_pod{pod_idx}"
            model_path = member2_dir / f"pod{pod_idx}.pt"
            pod_networks[pod_key] = PodNetwork().to(device)
            if model_path.exists():
                pod_networks[pod_key].load_state_dict(
                    torch.load(model_path, map_location=device)
                )
        
        # Run evaluation races
        observations = env.reset()
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
                    # Get deterministic action from network
                    obs = observations[pod_key]
                    action = network.get_actions(obs, deterministic=True)
                    actions[pod_key] = action
            
            # Step the environment
            observations, _, new_dones, info = env.step(actions)
            
            # Check for newly completed races
            newly_done = new_dones & ~dones
            if newly_done.any():
                # Determine winners for newly completed races
                for b in range(num_races):
                    if newly_done[b]:
                        # Check which player's pod completed the race
                        player0_progress = max(
                            info["checkpoint_progress"][0][b].item(),
                            info["checkpoint_progress"][1][b].item()
                        )
                        player1_progress = max(
                            info["checkpoint_progress"][2][b].item(),
                            info["checkpoint_progress"][3][b].item()
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
        
        # Initialize networks for this member
        if base_member_idx is not None:
            # Copy and mutate from base member
            base_member = self.league_members[base_member_idx]
            base_dir = Path(base_member['model_path'])
            
            for pod_idx in range(2):
                # Load base network
                network = PodNetwork().to(self.device)
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
                network = PodNetwork().to(self.device)
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
                     tournament_frequency: int = 2):
        """Evolve the league over multiple iterations with GPU optimization"""
        # Flag to track if evolution should be stopped
        self.stop_requested = False
        
        for iteration in range(1, iterations + 1):
            if self.stop_requested:
                print("Evolution stopped by user request")
                break
                
            print(f"\n=== League Evolution Iteration {iteration}/{iterations} ===")
            
            # Train each member with GPU distribution
            if self.multi_gpu and len(self.devices) > 1:
                # Distribute members across GPUs using threads
                import threading
                threads = []
                
                # Group members by GPU
                members_per_gpu = [[] for _ in range(len(self.devices))]
                for member_idx in range(len(self.league_members)):
                    gpu_idx = member_idx % len(self.devices)
                    members_per_gpu[gpu_idx].append(member_idx)
                
                def train_members_on_gpu(device_id, member_indices):
                    for member_idx in member_indices:
                        if self.stop_requested:
                            return
                        self.train_member(
                            member_idx=member_idx,
                            num_iterations=training_iterations,
                            device_id=device_id
                        )
                
                # Start a thread for each GPU
                for gpu_idx, device_id in enumerate(self.devices):
                    if members_per_gpu[gpu_idx]:  # Only start thread if there are members to train
                        thread = threading.Thread(
                            target=train_members_on_gpu,
                            args=(device_id, members_per_gpu[gpu_idx])
                        )
                        threads.append(thread)
                        thread.start()
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join()
                    
                # Check if stop was requested during training
                if self.stop_requested:
                    print("Evolution stopped by user request")
                    break
            else:
                # Train sequentially on a single GPU
                for member_idx in range(len(self.league_members)):
                    if self.stop_requested:
                        print("Evolution stopped by user request")
                        break
                    self.train_member(
                        member_idx=member_idx,
                        num_iterations=training_iterations
                    )
            
            # Run tournament periodically
            if iteration % tournament_frequency == 0:
                self.run_tournament()
                
                # Replace worst performer with a mutation of a top performer
                if len(self.league_members) >= 4:  # Need enough members to make this meaningful
                    # Sort by ELO
                    sorted_indices = sorted(
                        range(len(self.league_members)), 
                        key=lambda i: self.league_members[i]['elo'],
                        reverse=True
                    )
                    
                    # Select a top performer to clone
                    top_idx = sorted_indices[0]
                    
                    # Replace worst performer
                    worst_idx = sorted_indices[-1]
                    
                    print(f"Replacing {self.league_members[worst_idx]['name']} "
                          f"with a mutation of {self.league_members[top_idx]['name']}")
                    
                    # Remove worst member
                    self.league_members.pop(worst_idx)
                    
                    # Add new member based on top performer
                    self.add_new_member(base_member_idx=top_idx)
            
            # Add a completely new random member occasionally
            if iteration % 3 == 0:
                self.add_new_member()
        
        # Final tournament
        if not self.stop_requested:
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
