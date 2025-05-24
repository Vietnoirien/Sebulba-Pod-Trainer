"""
Script to check the configuration before training.
Run this before starting training to verify the configuration is correct.
"""

import json
from pathlib import Path

def check_config():
    config_path = Path("sebulba_config.json")
    if not config_path.exists():
        print("Error: Configuration file not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("=== CONFIGURATION CHECK ===")
        print(f"Configuration file: {config_path}")
        
        # Check for required parameters
        required_params = ['num_iterations', 'steps_per_iteration', 'save_interval', 'save_dir']
        missing_params = [param for param in required_params if param not in config]
        
        if missing_params:
            print(f"Warning: Missing required parameters: {missing_params}")
            print("These should be added to the configuration file.")
        else:
            print("All required parameters are present.")
        
        # Print key parameters
        print("\n=== KEY PARAMETERS ===")
        print(f"num_iterations: {config.get('num_iterations', 'NOT FOUND')}")
        print(f"steps_per_iteration: {config.get('steps_per_iteration', 'NOT FOUND')}")
        print(f"save_interval: {config.get('save_interval', 'NOT FOUND')}")
        print(f"save_dir: {config.get('save_dir', 'NOT FOUND')}")
        print(f"batch_size: {config.get('batch_size', 'NOT FOUND')}")
        print(f"learning_rate: {config.get('learning_rate', 'NOT FOUND')}")
        print(f"device: {config.get('device', 'NOT FOUND')}")
        print(f"multi_gpu: {config.get('multi_gpu', 'NOT FOUND')}")
        print(f"devices: {config.get('devices', 'NOT FOUND')}")
        print(f"use_parallel: {config.get('use_parallel', 'NOT FOUND')}")
        print(f"use_mixed_precision: {config.get('use_mixed_precision', 'NOT FOUND')}")
        
        # Check PPO parameters
        ppo_params = config.get('ppo_params', {})
        print("\n=== PPO PARAMETERS ===")
        if not ppo_params:
            print("Warning: No PPO parameters found")
        else:
            for key, value in ppo_params.items():
                print(f"{key}: {value}")
        
        return True
    
    except Exception as e:
        print(f"Error checking configuration: {str(e)}")
        return False

if __name__ == "__main__":
    check_config()