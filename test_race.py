import torch
import numpy as np
import matplotlib.pyplot as plt
from sebulba_pod_trainer.environment.race_env import RaceEnvironment
from sebulba_pod_trainer.models.pod import Pod

# Import our fix
from fix_race_env import fixed_check_checkpoints

# Apply the monkey patch
RaceEnvironment._check_checkpoints = fixed_check_checkpoints

def visualize_race(env, batch_idx=0):
    """Visualize the current state of a specific race in the batch"""
    plt.figure(figsize=(10, 6))
    
    # Plot the track (checkpoints)
    checkpoints = env.checkpoints[batch_idx].cpu().numpy()
    plt.scatter(checkpoints[:, 0], checkpoints[:, 1], s=200, c='green', alpha=0.7, label='Checkpoints')
    
    # Add checkpoint numbers
    for i, (x, y) in enumerate(checkpoints):
        plt.text(x, y, str(i), fontsize=12, ha='center', va='center')
    
    # Plot the pods
    colors = ['blue', 'blue', 'red', 'red']
    labels = ['Player 1 Pod 1', 'Player 1 Pod 2', 'Player 2 Pod 1', 'Player 2 Pod 2']
    
    for i, pod in enumerate(env.pods):
        pos = pod.position[batch_idx].cpu().numpy()
        vel = pod.velocity[batch_idx].cpu().numpy()
        
        # Plot pod position
        plt.scatter(pos[0], pos[1], s=150, c=colors[i], alpha=0.8, label=labels[i])
        
        # Plot velocity vector
        plt.arrow(pos[0], pos[1], vel[0], vel[1], width=20, head_width=100, 
                  head_length=100, fc=colors[i], ec=colors[i], alpha=0.6)
        
        # Show current checkpoint number
        current_cp = pod.current_checkpoint[batch_idx].item()
        plt.text(pos[0], pos[1] + 300, f"CP: {current_cp}", fontsize=10, ha='center')
    
    # Set plot limits
    plt.xlim(0, 16000)
    plt.ylim(0, 9000)
    plt.grid(True)
    plt.legend()
    plt.title(f"Race State - Turn {env.turn_count[batch_idx].item()}")
    
    return plt

def run_test_race():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Use CPU for easier testing
    device = torch.device('cpu')
    
    # Create a race environment with a small batch size
    batch_size = 1
    num_checkpoints = 3
    laps = 1  # Short race with just one lap
    
    env = RaceEnvironment(
        num_checkpoints=num_checkpoints,
        laps=laps,
        batch_size=batch_size,
        device=device
    )
    
    # Reset the environment
    observations = env.reset()
    
    # Create a simple policy that just aims at the next checkpoint with full thrust
    def simple_policy(obs):
        actions = {}
        for key in obs.keys():
            # Extract the relative position to next checkpoint (features 2 and 3)
            rel_pos = obs[key][:, 2:4]
            
            # Normalize to get direction
            direction = rel_pos / (torch.norm(rel_pos, dim=1, keepdim=True) + 1e-8)
            
            # Set angle adjustment to 0 (go straight to checkpoint)
            angle = torch.zeros(batch_size, 1, device=device)
            
            # Set thrust to maximum (1.0)
            thrust = torch.ones(batch_size, 1, device=device)
            
            # Combine into action
            actions[key] = torch.cat([angle, thrust], dim=1)
        return actions
    
    # Run the race for a fixed number of steps
    max_steps = 50
    visualize_steps = [0, 10, 25, max_steps-1]  # Steps to visualize
    
    for step in range(max_steps):
        # Get actions from our simple policy
        actions = simple_policy(observations)
        
        # Take a step in the environment
        observations, rewards, dones, info = env.step(actions)
        
        # Print some information
        print(f"Step {step+1}:")
        for pod_idx, pod in enumerate(env.pods):
            player_idx = pod_idx // 2
            team_pod_idx = pod_idx % 2
            pod_key = f"player{player_idx}_pod{team_pod_idx}"
            
            print(f"  {pod_key}:")
            print(f"    Position: {pod.position[0].tolist()}")
            print(f"    Velocity: {pod.velocity[0].tolist()}")
            print(f"    Checkpoint: {pod.current_checkpoint[0].item()}")
            print(f"    Reward: {rewards[pod_key][0].item():.4f}")
        
        # Visualize at specific steps
        if step in visualize_steps:
            plt = visualize_race(env)
            plt.savefig(f"race_step_{step+1}.png")
            plt.close()
        
        # Check if race is done
        if dones[0]:
            print(f"Race completed after {step+1} steps!")
            break
    
    # Final visualization
    plt = visualize_race(env)
    plt.savefig("race_final.png")
    plt.close()
    
    # Print final results
    print("\nFinal Results:")
    for pod_idx, pod in enumerate(env.pods):
        player_idx = pod_idx // 2
        team_pod_idx = pod_idx % 2
        pod_key = f"player{player_idx}_pod{team_pod_idx}"
        
        print(f"  {pod_key}: Checkpoints reached: {pod.current_checkpoint[0].item()}/{env.total_checkpoints}")

if __name__ == "__main__":
    run_test_race()
