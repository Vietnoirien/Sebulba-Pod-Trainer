import torch
import json
from pathlib import Path
from sebulba_pod_trainer.training.trainer_parallel import ParallelPPOTrainer

def run_debug_training():
    print("=== DEBUG TRAINING START ===")
    
    # Load configuration
    config_path = Path("sebulba_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"Loaded configuration: {config}")
    else:
        print("Configuration file not found, using default values")
        config = {
            'device': 'cuda:0',
            'multi_gpu': True,
            'devices': [0, 1],
            'batch_size': 64,
            'learning_rate': 0.0003,
            'save_dir': 'models',
            'use_mixed_precision': True,
            'use_parallel': True,
            'envs_per_gpu': 8,
            'num_iterations': 100,
            'steps_per_iteration': 100,
            'save_interval': 50,
            'ppo_params': {
                'mini_batch_size': 8,
                'ppo_epochs': 10,
                'clip_param': 0.2,
                'value_coef': 0.5,
                'entropy_coef': 0.01,
                'max_grad_norm': 0.5,
                'gamma': 0.99,
                'gae_lambda': 0.95
            }
        }
    
    # Create environment config
    env_config = {
        'batch_size': config['batch_size'],
        'num_checkpoints': 3,
        'laps': 3
    }
    
    # Create trainer args with explicit parameters
    trainer_args = {
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'devices': config['devices'],
        'use_mixed_precision': config.get('use_mixed_precision', True),
        'num_iterations': config.get('num_iterations', 100),
        'steps_per_iteration': config.get('steps_per_iteration', 100),
        'save_interval': config.get('save_interval', 50),
        'save_dir': config.get('save_dir', 'models')
    }
    
    # Add PPO parameters
    if 'ppo_params' in config:
        for key, value in config['ppo_params'].items():
            trainer_args[key] = value
    
    print(f"Environment config: {env_config}")
    print(f"Trainer args: {trainer_args}")
    print(f"num_iterations: {trainer_args.get('num_iterations')}")
    
    try:
        # Create parallel trainer
        trainer = ParallelPPOTrainer(
            env_config=env_config,
            trainer_args=trainer_args,
            envs_per_gpu=config.get('envs_per_gpu', 8)
        )
        
        # Start parallel training
        trainer.train()
        print("Training completed successfully!")
        
    except Exception as e:
        import traceback
        print(f"Error during training: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    run_debug_training()