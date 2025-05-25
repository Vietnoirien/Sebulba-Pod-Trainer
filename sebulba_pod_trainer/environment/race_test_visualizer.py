import pygame
import torch
import numpy as np
import time
import argparse
from typing import Dict, List, Tuple

from .optimized_race_env import OptimizedRaceEnvironment

# Visualization constants
SCALE_FACTOR = 10  # Scale down the game coordinates for display
WIDTH_DISPLAY = int(16000 / SCALE_FACTOR)
HEIGHT_DISPLAY = int(9000 / SCALE_FACTOR)
FPS = 30
BACKGROUND_COLOR = (0, 0, 30)  # Dark blue
CHECKPOINT_COLOR = (255, 0, 0)  # Red
POD_COLORS = [
    (0, 255, 0),    # Player 1, Pod 1: Green
    (0, 200, 0),    # Player 1, Pod 2: Dark Green
    (0, 0, 255),    # Player 2, Pod 1: Blue
    (0, 0, 200),    # Player 2, Pod 2: Dark Blue
]
FONT_COLOR = (255, 255, 255)  # White
CHECKPOINT_RADIUS = 600  # Game units
POD_RADIUS = 400  # Game units


class RaceTestVisualizer:
    """
    Visualizer for testing the OptimizedRaceEnvironment with pygame.
    """
    def __init__(self, 
                 num_checkpoints: int = 3,
                 laps: int = 3,
                 batch_index: int = 0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 24)
        self.screen = pygame.display.set_mode((WIDTH_DISPLAY, HEIGHT_DISPLAY))
        pygame.display.set_caption("Mad Pod Racing - Environment Test")
        self.clock = pygame.time.Clock()
        
        # Create environment (batch size 1 for visualization)
        self.env = OptimizedRaceEnvironment(
            num_checkpoints=num_checkpoints,
            laps=laps,
            batch_size=1,
            device=torch.device(device)
        )
        
        # Index to visualize from the batch
        self.batch_index = batch_index
        
        # Load pod images
        self.pod_images = self._load_pod_images()
        
        # Initialize AI state tracking
        self._init_ai_state()
        
    def _init_ai_state(self):
        """Initialize AI state tracking to prevent repetitive behavior"""
        self.action_counter = 0
        self.pod_stuck_counters = [0] * 4  # Track how long each pod has been stuck
        self.pod_last_positions = [None] * 4  # Track last positions
        self.pod_action_history = [[] for _ in range(4)]  # Track recent actions
        self.pod_random_seeds = [np.random.randint(0, 1000) for _ in range(4)]  # Individual randomness
        
    def _load_pod_images(self) -> List[pygame.Surface]:
        """Load or create pod images for visualization"""
        images = []
        for color in POD_COLORS:
            # Create a surface for the pod
            img = pygame.Surface((POD_RADIUS*2//SCALE_FACTOR, POD_RADIUS*2//SCALE_FACTOR), pygame.SRCALPHA)
            pygame.draw.circle(
                img, 
                color, 
                (POD_RADIUS//SCALE_FACTOR, POD_RADIUS//SCALE_FACTOR), 
                POD_RADIUS//SCALE_FACTOR
            )
            # Add a direction indicator
            pygame.draw.line(
                img,
                (255, 255, 255),
                (POD_RADIUS//SCALE_FACTOR, POD_RADIUS//SCALE_FACTOR),
                (POD_RADIUS*2//SCALE_FACTOR, POD_RADIUS//SCALE_FACTOR),
                3
            )
            images.append(img)
        return images
    
    def _draw_checkpoints(self) -> None:
        """Draw the checkpoints on the screen"""
        batch_idx = self.batch_index
        num_checkpoints = self.env.batch_checkpoint_counts[batch_idx].item()
        checkpoints = self.env.checkpoints[batch_idx, :num_checkpoints].cpu().numpy()
        
        for i, (x, y) in enumerate(checkpoints):
            # Draw checkpoint circle
            pygame.draw.circle(
                self.screen,
                CHECKPOINT_COLOR,
                (int(x/SCALE_FACTOR), int(y/SCALE_FACTOR)),
                int(CHECKPOINT_RADIUS/SCALE_FACTOR),
                width=2
            )
            # Draw checkpoint number
            text = self.font.render(str(i), True, FONT_COLOR)
            self.screen.blit(
                text,
                (int(x/SCALE_FACTOR) - text.get_width()//2, 
                 int(y/SCALE_FACTOR) - text.get_height()//2)
            )
    
    def _draw_pods(self) -> None:
        """Draw the pods on the screen"""
        for i, pod in enumerate(self.env.pods):
            # Get pod position and angle
            x, y = pod.position[self.batch_index].cpu().numpy()
            angle = pod.angle[self.batch_index].item()
            
            # Get the appropriate image and rotate it
            img = self.pod_images[i]
            rotated_img = pygame.transform.rotate(img, -angle)
            
            # Calculate position accounting for rotation
            pos_x = int(x/SCALE_FACTOR) - rotated_img.get_width()//2
            pos_y = int(y/SCALE_FACTOR) - rotated_img.get_height()//2
            
            # Draw the pod
            self.screen.blit(rotated_img, (pos_x, pos_y))
            
            # Draw pod info
            current_cp = pod.current_checkpoint[self.batch_index].item()
            batch_idx = self.batch_index
            num_cp = self.env.batch_checkpoint_counts[batch_idx].item()
            next_cp = current_cp % num_cp
            
            # Display pod info
            info_text = f"Pod {i}: CP {current_cp} (Next: {next_cp})"
            text = self.font.render(info_text, True, FONT_COLOR)
            self.screen.blit(
                text,
                (10, 10 + i * 30)
            )
    
    def _draw_info(self) -> None:
        """Draw additional information on the screen"""
        turn_count = self.env.turn_count[self.batch_index].item()
        info_text = f"Turn: {turn_count}"
        text = self.font.render(info_text, True, FONT_COLOR)
        self.screen.blit(text, (WIDTH_DISPLAY - text.get_width() - 10, 10))
        
        # Draw race status
        if self.env.done[self.batch_index].item():
            status_text = "RACE FINISHED"
            text = self.font.render(status_text, True, (255, 255, 0))
            self.screen.blit(
                text,
                (WIDTH_DISPLAY//2 - text.get_width()//2, 10)
            )
    
    def _get_random_actions(self) -> Dict[str, torch.Tensor]:
        """Generate random actions for all pods"""
        actions = {}
        for i in range(4):
            player_idx = i // 2
            team_pod_idx = i % 2
            pod_key = f"player{player_idx}_pod{team_pod_idx}"
            
            # Random angle adjustment [-1, 1] and thrust [0, 1]
            angle = torch.rand(1, 1, device=self.env.device) * 2 - 1  # [-1, 1]
            thrust = torch.rand(1, 1, device=self.env.device)  # [0, 1]
            
            # Occasionally use boost or shield
            if torch.rand(1).item() < 0.01:  # 1% chance for boost
                thrust = torch.ones(1, 1, device=self.env.device)  # Boost (>0.9)
            elif torch.rand(1).item() < 0.01:  # 1% chance for shield
                thrust = -torch.ones(1, 1, device=self.env.device)  # Shield (<-0.9)
                
            actions[pod_key] = torch.cat([angle, thrust], dim=1)
        
        return actions
    
    def _is_pod_stuck(self, pod_idx: int) -> bool:
        """Check if a pod is stuck (not moving much)"""
        pod = self.env.pods[pod_idx]
        current_pos = pod.position[self.batch_index].cpu().numpy()
        
        if self.pod_last_positions[pod_idx] is not None:
            distance_moved = np.linalg.norm(current_pos - self.pod_last_positions[pod_idx])
            if distance_moved < 50:  # Very small movement
                self.pod_stuck_counters[pod_idx] += 1
            else:
                self.pod_stuck_counters[pod_idx] = 0
        
        self.pod_last_positions[pod_idx] = current_pos.copy()
        return self.pod_stuck_counters[pod_idx] > 10  # Stuck for more than 10 steps
    
    def _get_simple_ai_actions(self) -> Dict[str, torch.Tensor]:
        """Generate simple AI actions for all pods with anti-stuck mechanisms"""
        actions = {}
        self.action_counter += 1
        
        for i, pod in enumerate(self.env.pods):
            player_idx = i // 2
            team_pod_idx = i % 2
            pod_key = f"player{player_idx}_pod{team_pod_idx}"
            
            # Check if pod is stuck
            is_stuck = self._is_pod_stuck(i)
            
            # Get next checkpoint position using batch-specific checkpoint count
            batch_idx = self.batch_index
            num_checkpoints = self.env.batch_checkpoint_counts[batch_idx].item()
            next_cp_idx = pod.current_checkpoint[batch_idx].item() % num_checkpoints
            next_cp_pos = self.env.checkpoints[batch_idx, next_cp_idx].cpu().numpy()
            
            # Calculate angle to next checkpoint
            pod_pos = pod.position[batch_idx].cpu().numpy()
            pod_angle = pod.angle[batch_idx].item()
            
            # Calculate target angle
            dx = next_cp_pos[0] - pod_pos[0]
            dy = next_cp_pos[1] - pod_pos[1]
            target_angle = np.degrees(np.arctan2(dy, dx))
            
            # Normalize angle difference to [-180, 180]
            angle_diff = (target_angle - pod_angle + 180) % 360 - 180
            
            # Add randomness based on pod-specific seed and time
            np.random.seed((self.pod_random_seeds[i] + self.action_counter) % 10000)
            base_randomness = (np.random.random() - 0.5) * 0.3  # Base randomness
            
            # If stuck, add more aggressive randomness
            if is_stuck:
                stuck_randomness = (np.random.random() - 0.5) * 60  # Large random angle
                angle_diff += stuck_randomness
                print(f"Pod {i} is stuck, adding random angle: {stuck_randomness:.1f}")
            
            # Add periodic variation to prevent cycles
            cycle_randomness = np.sin(self.action_counter * 0.1 + i) * 10  # Sinusoidal variation
            angle_diff += cycle_randomness + base_randomness * 18
            
            # Convert to normalized [-1, 1] format
            normalized_angle = angle_diff / 18.0
            normalized_angle = np.clip(normalized_angle, -1.0, 1.0)
            
            # Calculate distance to next checkpoint
            distance = np.sqrt(dx**2 + dy**2)
            
            # Determine thrust with more variation
            if is_stuck:
                # If stuck, try different thrust patterns
                if self.pod_stuck_counters[i] % 4 == 0:
                    thrust_value = 1.0  # Full thrust
                elif self.pod_stuck_counters[i] % 4 == 1:
                    thrust_value = 0.0  # No thrust
                elif self.pod_stuck_counters[i] % 4 == 2:
                    thrust_value = -1.0  # Shield
                else:
                    thrust_value = 0.5  # Half thrust
            else:
                # Normal thrust calculation with more variation
                if abs(angle_diff) > 90:
                    thrust_value = 0.1 + np.random.random() * 0.3  # 0.1-0.4
                elif abs(angle_diff) > 45:
                    thrust_value = 0.3 + np.random.random() * 0.4  # 0.3-0.7
                elif abs(angle_diff) > 20:
                    thrust_value = 0.5 + np.random.random() * 0.4  # 0.5-0.9
                else:
                    thrust_value = 0.7 + np.random.random() * 0.3  # 0.7-1.0
                
                # Distance-based adjustments
                if distance < 1000:
                    thrust_value *= 0.5
                elif distance > 5000:
                    thrust_value = min(thrust_value * 1.2, 1.0)
                
                # Occasional special actions
                if np.random.random() < 0.05:  # 5% chance
                    if np.random.random() < 0.5:
                        thrust_value = 1.1  # Boost
                    else:
                        thrust_value = -1.0  # Shield
                        print("Shield activated!")
            
            # Create tensors
            angle_tensor = torch.tensor([[normalized_angle]], device=self.env.device, dtype=torch.float32)
            thrust_tensor = torch.tensor([[thrust_value]], device=self.env.device, dtype=torch.float32)
            
            # Store action in history (keep last 5 actions)
            action_tuple = (normalized_angle, thrust_value)
            self.pod_action_history[i].append(action_tuple)
            if len(self.pod_action_history[i]) > 5:
                self.pod_action_history[i].pop(0)
            
            actions[pod_key] = torch.cat([angle_tensor, thrust_tensor], dim=1)
        
        return actions
    
    def run(self, max_steps: int = 1000, delay: float = 0.05, ai_mode: str = 'simple') -> None:
        """Run the visualization"""
        # Reset environment
        observations = self.env.reset()
        
        step = 0
        running = True
        
        while running and step < max_steps and not self.env.done[self.batch_index].item():
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        # Reset environment
                        observations = self.env.reset()
                        self._init_ai_state()
                        step = 0
                        print("Environment reset!")
                    elif event.key == pygame.K_SPACE:
                        # Pause/unpause
                        paused = True
                        while paused:
                            for pause_event in pygame.event.get():
                                if pause_event.type == pygame.QUIT:
                                    running = False
                                    paused = False
                                elif pause_event.type == pygame.KEYDOWN:
                                    if pause_event.key == pygame.K_SPACE:
                                        paused = False
                            time.sleep(0.1)
            
            # Clear screen
            self.screen.fill(BACKGROUND_COLOR)
            
            # Draw environment
            self._draw_checkpoints()
            self._draw_pods()
            self._draw_info()
            
            # Draw controls info
            controls_text = [
                "Controls: ESC=Quit, R=Reset, SPACE=Pause",
                f"AI Mode: {ai_mode.upper()}"
            ]
            for i, text in enumerate(controls_text):
                rendered = self.font.render(text, True, FONT_COLOR)
                self.screen.blit(rendered, (10, HEIGHT_DISPLAY - 60 + i * 25))
            
            # Update display
            pygame.display.flip()
            
            # Generate actions
            if ai_mode == 'random':
                actions = self._get_random_actions()
            else:  # simple AI
                actions = self._get_simple_ai_actions()
            
            # Step environment
            observations, rewards, done, info = self.env.step(actions)
            
            # Print rewards and actions (less verbose)
            if step % 10 == 0:  # Print every 10 steps to reduce spam
                print(f"Step {step}")
                print(f"  Rewards: {', '.join([f'{k}: {v[self.batch_index].item():.3f}' for k, v in rewards.items()])}")
                
                # Show stuck status
                stuck_pods = [i for i in range(4) if self.pod_stuck_counters[i] > 5]
                if stuck_pods:
                    print(f"  Stuck pods: {stuck_pods}")
            
            # Delay for visualization
            time.sleep(delay)
            
            # Increment step counter
            step += 1
            
            # Limit FPS
            self.clock.tick(FPS)
        
        # Show final state for a moment
        if self.env.done[self.batch_index].item():
            print("Race finished!")
            
            # Show final results
            final_text = "RACE FINISHED - Press any key to exit"
            text_surface = self.font.render(final_text, True, (255, 255, 0))
            self.screen.blit(
                text_surface,
                (WIDTH_DISPLAY//2 - text_surface.get_width()//2, HEIGHT_DISPLAY//2)
            )
            pygame.display.flip()
            
            # Wait for key press
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                        waiting = False
                time.sleep(0.1)
        
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Race Test Visualizer')
    parser.add_argument('--checkpoints', type=int, default=3, help='Number of checkpoints')
    parser.add_argument('--laps', type=int, default=3, help='Number of laps')
    parser.add_argument('--delay', type=float, default=0.05, help='Delay between steps (seconds)')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum number of steps')
    parser.add_argument('--ai', type=str, default='simple', choices=['simple', 'random'], 
                        help='AI mode: simple or random')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cuda/cpu)')
    
    args = parser.parse_args()
    
    print(f"Starting Race Test Visualizer...")
    print(f"Checkpoints: {args.checkpoints}, Laps: {args.laps}")
    print(f"AI Mode: {args.ai}, Device: {args.device}")
    print(f"Controls: ESC=Quit, R=Reset, SPACE=Pause")
    
    visualizer = RaceTestVisualizer(
        num_checkpoints=args.checkpoints,
        laps=args.laps,
        device=args.device
    )
    
    visualizer.run(
        max_steps=args.max_steps,
        delay=args.delay,
        ai_mode=args.ai
    )


if __name__ == "__main__":
    main()
