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
        checkpoints = self.env.checkpoints[self.batch_index].cpu().numpy()
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
            next_cp = current_cp % self.env.num_checkpoints
            
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
    
    def _get_simple_ai_actions(self) -> Dict[str, torch.Tensor]:
        """Generate simple AI actions for all pods"""
        actions = {}
        
        for i, pod in enumerate(self.env.pods):
            player_idx = i // 2
            team_pod_idx = i % 2
            pod_key = f"player{player_idx}_pod{team_pod_idx}"
            
            # Get next checkpoint position
            next_cp_idx = pod.current_checkpoint[self.batch_index].item() % self.env.num_checkpoints
            next_cp_pos = self.env.checkpoints[self.batch_index, next_cp_idx].cpu().numpy()
            
            # Calculate angle to next checkpoint
            pod_pos = pod.position[self.batch_index].cpu().numpy()
            pod_angle = pod.angle[self.batch_index].item()
            
            # Calculate target angle
            dx = next_cp_pos[0] - pod_pos[0]
            dy = next_cp_pos[1] - pod_pos[1]
            target_angle = np.degrees(np.arctan2(dy, dx))
            
            # Normalize angle difference to [-180, 180]
            angle_diff = (target_angle - pod_angle + 180) % 360 - 180
            
            # Convert to normalized [-1, 1] format - use a more aggressive steering factor
            # The key fix: increase the steering factor to make turns more responsive
            normalized_angle = torch.tensor([[angle_diff / 18.0]], device=self.env.device)
            # Clamp to ensure we stay within the valid range
            normalized_angle = torch.clamp(normalized_angle, -1.0, 1.0)
            
            # Calculate distance to next checkpoint
            distance = np.sqrt(dx**2 + dy**2)
            
            # Determine thrust based on angle difference and distance
            # Improve the turning logic by reducing thrust more aggressively when turning
            if abs(angle_diff) > 90:
                # If we're facing away from the checkpoint, stop completely
                thrust_value = 0.0
            elif abs(angle_diff) > 45:
                # If we need to make a significant turn, slow down more
                thrust_value = 0.3
            elif distance < 1500:
                # If we're close to the checkpoint, slow down more to prepare for the turn
                thrust_value = 0.5
            else:
                # Otherwise, go at full speed
                thrust_value = 1.0
                
            # Occasionally use boost on long straights when well-aligned
            if abs(angle_diff) < 5 and distance > 5000 and torch.rand(1).item() < 0.1:
                if pod.boost_available[self.batch_index].item() > 0:
                    thrust_value = 1.1  # Boost
            
            thrust = torch.tensor([[thrust_value]], device=self.env.device)
            
            # Combine angle and thrust
            actions[pod_key] = torch.cat([normalized_angle, thrust], dim=1)
        
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
            
            # Clear screen
            self.screen.fill(BACKGROUND_COLOR)
            
            # Draw environment
            self._draw_checkpoints()
            self._draw_pods()
            self._draw_info()
            
            # Update display
            pygame.display.flip()
            
            # Generate actions
            if ai_mode == 'random':
                actions = self._get_random_actions()
            else:  # simple AI
                actions = self._get_simple_ai_actions()
            
            # Step environment
            observations, rewards, done, info = self.env.step(actions)
            
            # Print rewards
            print(f"Step {step}, Rewards: {', '.join([f'{k}: {v[self.batch_index].item():.3f}' for k, v in rewards.items()])}")
            
            # Delay for visualization
            time.sleep(delay)
            
            # Increment step counter
            step += 1
            
            # Limit FPS
            self.clock.tick(FPS)
        
        # Show final state for a moment
        if self.env.done[self.batch_index].item():
            print("Race finished!")
            time.sleep(2)
        
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