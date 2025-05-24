import sys
import torch
import json
from pathlib import Path

# Add debug prints to identify the issue with 'num_iterations'
def debug_training_setup():
    print("=== DEBUGGING TRAINING SETUP ===")
    
    # Print Python and PyTorch versions
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Load the configuration
    config_path = Path("sebulba_config.json")
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                print("\n=== CONFIGURATION ===")
                print(json.dumps(config, indent=2))
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
    else:
        print("Configuration file not found")
    
    # Try to create a ParallelPPOTrainer with explicit debug
    try:
        from sebulba_pod_trainer.training.trainer_parallel import ParallelPPOTrainer
        
        print("\n=== CREATING PARALLEL TRAINER ===")
        
        # Create environment config
        env_config = {
            'batch_size': 64,
            'num_checkpoints': 3,
            'laps': 3
        }
        print(f"Environment config: {env_config}")
        
        # Create trainer args with explicit parameters
        trainer_args = {
            'learning_rate': 0.0003,
            'batch_size': 64,
            'devices': [0, 1],
            'use_mixed_precision': True,
            'num_iterations': 100,  # Explicitly add this
            'steps_per_iteration': 100,  # Explicitly add this
            'save_interval': 50,
            'save_dir': 'models',
            'mini_batch_size': 8,
            'ppo_epochs': 10,
            'clip_param': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'gamma': 0.99,
            'gae_lambda': 0.95
        }
        print(f"Trainer args: {trainer_args}")
        
        # Create parallel trainer
        print("Creating ParallelPPOTrainer instance...")
        trainer = ParallelPPOTrainer(
            env_config=env_config,
            trainer_args=trainer_args,
            envs_per_gpu=8
        )
        
        print("ParallelPPOTrainer created successfully")
        print(f"Trainer attributes: {dir(trainer)}")
        
        # Try to access the num_iterations attribute
        print(f"trainer.trainer_args.get('num_iterations'): {trainer.trainer_args.get('num_iterations', 'NOT FOUND')}")
        
    except Exception as e:
        import traceback
        print(f"\n=== ERROR CREATING TRAINER ===")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    debug_training_setup()