import pygame
import torch
import numpy as np
import time
import argparse
import os
from typing import Dict, List, Tuple

from .optimized_race_env import OptimizedRaceEnvironment
from .heuristic_adapter import HeuristicPodAdapter

# Visualization constants (same as race_test_visualizer.py)
SCALE_FACTOR = 10
WIDTH_DISPLAY = int(16000 / SCALE_FACTOR)
HEIGHT_DISPLAY = int(9000 / SCALE_FACTOR)
FPS = 30
BACKGROUND_COLOR = (0, 0, 30)
CHECKPOINT_COLOR = (255, 0, 0)
POD_COLORS = [
    (0, 255, 0),    # Player 1, Pod 1: Green
    (0, 200, 0),    # Player 1, Pod 2: Dark Green
    (0, 0, 255),    # Player 2, Pod 1: Blue
    (0, 0, 200),    # Player 2, Pod 2: Dark Blue
]
FONT_COLOR = (255, 255, 255)
CHECKPOINT_RADIUS = 600
POD_RADIUS = 400


class HeuristicTestVisualizer:
    """
    Visualizer for testing the heuristic pod with OptimizedRaceEnvironment.
    """
    def __init__(self, 
                 num_checkpoints: int = 3,
                 laps: int = 3,
                 batch_index: int = 0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 heuristic_path: str = None):
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 16)
        self.screen = pygame.display.set_mode((WIDTH_DISPLAY, HEIGHT_DISPLAY))
        pygame.display.set_caption("Mad Pod Racing - Heuristic Test")
        self.clock = pygame.time.Clock()
        
        # Create environment
        self.env = OptimizedRaceEnvironment(
            num_checkpoints=num_checkpoints,
            laps=laps,
            batch_size=1,
            device=torch.device(device)
        )
        
        # Create heuristic adapter
        self.heuristic_adapter = HeuristicPodAdapter(heuristic_path)
        
        # Index to visualize from the batch
        self.batch_index = batch_index
        
        # Load pod images
        self.pod_images = self._load_pod_images()
        
        # Statistics tracking
        self.step_count = 0
        self.checkpoint_times = {}
        
        # Store last actions for display
        self.last_actions = {}
        
    def _load_pod_images(self) -> List[pygame.Surface]:
        """Load or create pod images for visualization"""
        images = []
        for color in POD_COLORS:
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
            pygame.draw.circle(
                self.screen,
                CHECKPOINT_COLOR,
                (int(x/SCALE_FACTOR), int(y/SCALE_FACTOR)),
                int(CHECKPOINT_RADIUS/SCALE_FACTOR),
                width=2
            )
            text = self.font.render(str(i), True, FONT_COLOR)
            self.screen.blit(
                text,
                (int(x/SCALE_FACTOR) - text.get_width()//2, 
                 int(y/SCALE_FACTOR) - text.get_height()//2)
            )
    
    def _draw_pods(self) -> None:
        """Draw the pods on the screen with better text layout and thrust display"""
        for i, pod in enumerate(self.env.pods):
            x, y = pod.position[self.batch_index].cpu().numpy()
            vx, vy = pod.velocity[self.batch_index].cpu().numpy()
            angle = pod.angle[self.batch_index].item()
            
            img = self.pod_images[i]
            rotated_img = pygame.transform.rotate(img, -angle)
            
            pos_x = int(x/SCALE_FACTOR) - rotated_img.get_width()//2
            pos_y = int(y/SCALE_FACTOR) - rotated_img.get_height()//2
            
            self.screen.blit(rotated_img, (pos_x, pos_y))
            
            # Draw velocity vector for debugging
            if abs(vx) > 10 or abs(vy) > 10:  # Only draw if significant velocity
                end_x = int((x + vx * 3)/SCALE_FACTOR)  # Scale velocity for visibility
                end_y = int((y + vy * 3)/SCALE_FACTOR)
                pygame.draw.line(
                    self.screen,
                    (255, 255, 0),  # Yellow for velocity
                    (int(x/SCALE_FACTOR), int(y/SCALE_FACTOR)),
                    (end_x, end_y),
                    2
                )
            
            # Pod info with better spacing
            current_cp = pod.current_checkpoint[self.batch_index].item()
            batch_idx = self.batch_index
            num_cp = self.env.batch_checkpoint_counts[batch_idx].item()
            
            # Determine pod role
            player_idx = i // 2
            team_pod_idx = i % 2
            role = "Runner" if team_pod_idx == 0 else "Blocker"
            
            speed = np.sqrt(vx*vx + vy*vy)
            
            # Calculate text positions with proper spacing
            base_y = 10 + i * 55  # Increased spacing to accommodate thrust info
            
            # Main pod info
            info_text = f"P{player_idx+1} {role}: CP {current_cp}/{num_cp * self.env.laps}"
            text = self.small_font.render(info_text, True, POD_COLORS[i])
            self.screen.blit(text, (10, base_y))
            
            # Speed info on the same line
            speed_text = f"Speed: {speed:.0f}"
            speed_surface = self.small_font.render(speed_text, True, POD_COLORS[i])
            self.screen.blit(speed_surface, (200, base_y))
            
            # Shield/boost status on second line
            shield_cd = pod.shield_cooldown[self.batch_index].item()
            boost_avail = pod.boost_available[self.batch_index].item()
            status_text = f"Shield: {shield_cd}, Boost: {'✓' if boost_avail else '✗'}"
            status_surface = self.small_font.render(status_text, True, (200, 200, 200))
            self.screen.blit(status_surface, (20, base_y + 18))  # Second line
            
            # Action info on third line (NEW)
            pod_key = f"player{player_idx}_pod{team_pod_idx}"
            if pod_key in self.last_actions:
                action = self.last_actions[pod_key]
                if action is not None and len(action) >= 4:
                    angle_adj = action[0] * 18.0  # Convert to degrees
                    thrust_val = action[1] * 100.0  # Convert to thrust value
                    shield_prob = action[2]
                    boost_prob = action[3]
                    
                    action_text = f"Thrust: {thrust_val:.0f}, Angle: {angle_adj:.1f}°"
                    action_surface = self.small_font.render(action_text, True, (150, 150, 255))
                    self.screen.blit(action_surface, (20, base_y + 36))  # Third line
                    
                    # Show probabilities if significant
                    if shield_prob > 0.1 or boost_prob > 0.1:
                        prob_text = f"Shield: {shield_prob:.2f}, Boost: {boost_prob:.2f}"
                        prob_surface = self.small_font.render(prob_text, True, (255, 150, 150))
                        self.screen.blit(prob_surface, (250, base_y + 36))

    def _draw_info(self) -> None:
        """Draw additional information on the screen with better positioning"""
        turn_count = self.env.turn_count[self.batch_index].item()
        
        # Position info in top-right corner
        info_y_start = 10
        
        # Turn count
        info_text = f"Turn: {turn_count}"
        text = self.font.render(info_text, True, FONT_COLOR)
        self.screen.blit(text, (WIDTH_DISPLAY - text.get_width() - 10, info_y_start))
        
        # Heuristic status
        heuristic_text = "Heuristic Controller Active"
        text = self.font.render(heuristic_text, True, (0, 255, 0))
        self.screen.blit(text, (WIDTH_DISPLAY - text.get_width() - 10, info_y_start + 30))
        
        # Race status
        if self.env.done[self.batch_index].item():
            status_text = "RACE FINISHED"
            text = self.font.render(status_text, True, (255, 255, 0))
            self.screen.blit(
                text,
                (WIDTH_DISPLAY//2 - text.get_width()//2, 10)
            )
        
        # Performance stats in bottom-left
        if self.step_count > 0:
            stats_text = f"Avg FPS: {self.step_count / (time.time() - self.start_time):.1f}"
            text = self.small_font.render(stats_text, True, FONT_COLOR)
            self.screen.blit(text, (10, HEIGHT_DISPLAY - 50))
        
        # Additional debug info
        if hasattr(self.env, '_last_reward_breakdown') and self.env._last_reward_breakdown:
            debug_y = HEIGHT_DISPLAY - 30
            debug_text = "Reward Debug (last step):"
            text = self.small_font.render(debug_text, True, (255, 255, 0))
            self.screen.blit(text, (10, debug_y))
            
            # Show reward breakdown for first pod as example
            if 'player0_pod0' in self.env._last_reward_breakdown:
                reward_info = self.env._last_reward_breakdown['player0_pod0']
                reward_text = f"P1 Runner Reward: {reward_info.get('total_reward', 0):.3f}"
                text = self.small_font.render(reward_text, True, (255, 255, 0))
                self.screen.blit(text, (200, debug_y))

    def run(self, max_steps: int = 2000, delay: float = 0.02) -> None:
        """Run the visualization with better error handling"""
        print("Starting heuristic pod test...")
        print("Controls: ESC=Quit, R=Reset, SPACE=Pause, D=Toggle Debug")
        
        # Reset environment and adapter
        observations = self.env.reset()
        self.heuristic_adapter.reset()
        
        step = 0
        running = True
        show_debug = False
        self.start_time = time.time()
        
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
                        self.heuristic_adapter.reset()
                        step = 0
                        self.start_time = time.time()
                        print("Environment reset!")
                    elif event.key == pygame.K_d:
                        # Toggle debug info
                        show_debug = not show_debug
                        print(f"Debug info: {'ON' if show_debug else 'OFF'}")
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
            
            # Draw debug info if enabled
            if show_debug:
                self._draw_debug_info()
            
            # Update display
            pygame.display.flip()
            
            # Get actions from heuristic adapter
            try:
                actions = self.heuristic_adapter.get_actions(
                    observations, 
                    env_state=self.env, 
                    batch_idx=self.batch_index
                )
                
                # Store actions for display (NEW)
                self.last_actions = {}
                for pod_key, action_tensor in actions.items():
                    if action_tensor is not None:
                        # Convert tensor to list for easier display
                        action_values = action_tensor[self.batch_index].cpu().numpy().tolist()
                        self.last_actions[pod_key] = action_values
                    else:
                        self.last_actions[pod_key] = None
                        
            except Exception as e:
                print(f"Error getting heuristic actions: {e}")
                # Fallback to simple actions
                actions = {}
                self.last_actions = {}
                for i in range(4):
                    player_idx = i // 2
                    team_pod_idx = i % 2
                    pod_key = f"player{player_idx}_pod{team_pod_idx}"
                    actions[pod_key] = torch.tensor([[0.0, 0.5, 0.0, 0.0]], dtype=torch.float32)
                    self.last_actions[pod_key] = [0.0, 0.5, 0.0, 0.0]
            
            # Step environment
            observations, rewards, done, info = self.env.step(actions)
            
            # Print progress periodically
            if step % 50 == 0:
                print(f"Step {step}: Turn {self.env.turn_count[self.batch_index].item()}")
                for i, pod in enumerate(self.env.pods):
                    cp = pod.current_checkpoint[self.batch_index].item()
                    total_cp = self.env.batch_checkpoint_counts[self.batch_index].item() * self.env.laps
                    progress = cp / total_cp * 100
                    speed = torch.norm(pod.velocity[self.batch_index]).item()
                    
                    # Show thrust info in console too
                    player_idx = i // 2
                    team_pod_idx = i % 2
                    pod_key = f"player{player_idx}_pod{team_pod_idx}"
                    thrust_info = ""
                    if pod_key in self.last_actions and self.last_actions[pod_key]:
                        thrust_val = self.last_actions[pod_key][1] * 100.0
                        thrust_info = f", Thrust: {thrust_val:.0f}"
                    
                    print(f"  Pod {i}: {cp}/{total_cp} checkpoints ({progress:.1f}%), Speed: {speed:.0f}{thrust_info}")
            
            # Delay for visualization
            time.sleep(delay)
            step += 1
            self.step_count = step
            
            # Limit FPS
            self.clock.tick(FPS)
        
        # Show final state (rest of the method remains the same)
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
            
            # Print final statistics
            total_time = time.time() - self.start_time
            print(f"Race completed in {step} steps ({total_time:.2f} seconds)")
            for i, pod in enumerate(self.env.pods):
                cp = pod.current_checkpoint[self.batch_index].item()
                total_cp = self.env.batch_checkpoint_counts[self.batch_index].item() * self.env.laps
                speed = torch.norm(pod.velocity[self.batch_index]).item()
                print(f"Pod {i} final progress: {cp}/{total_cp} checkpoints, Final speed: {speed:.0f}")
            
            # Wait for key press
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                        waiting = False
                time.sleep(0.1)
        
        pygame.quit()

    def _draw_debug_info(self) -> None:
        """Draw additional debug information"""
        debug_y_start = 250  # Start below pod info (increased due to more pod info lines)
        
        # Show action values if available
        debug_text = "Debug Info (Press D to toggle):"
        text = self.small_font.render(debug_text, True, (255, 255, 0))
        self.screen.blit(text, (10, debug_y_start))
        
        # Show environment stats
        for i, pod in enumerate(self.env.pods):
            y_pos = debug_y_start + 20 + i * 15
            
            # Pod detailed state
            pos = pod.position[self.batch_index]
            vel = pod.velocity[self.batch_index]
            debug_info = f"Pod {i}: Pos({pos[0]:.0f},{pos[1]:.0f}) Vel({vel[0]:.1f},{vel[1]:.1f})"
            
            text = self.small_font.render(debug_info, True, (200, 200, 200))
            self.screen.blit(text, (10, y_pos))


def main():
    parser = argparse.ArgumentParser(description='Heuristic Pod Test Visualizer')
    parser.add_argument('--checkpoints', type=int, default=3, help='Number of checkpoints')
    parser.add_argument('--laps', type=int, default=3, help='Number of laps')
    parser.add_argument('--delay', type=float, default=0.02, help='Delay between steps (seconds)')
    parser.add_argument('--max_steps', type=int, default=800, help='Maximum number of steps')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--heuristic', type=str, default='pods/heuristic_pod.py',
                        help='Path to heuristic pod controller')
    
    args = parser.parse_args()
    
    # Check if heuristic file exists
    if not os.path.exists(args.heuristic):
        print(f"Warning: Heuristic file not found at {args.heuristic}")
        print("Will use fallback controller")
    
    print(f"Starting Heuristic Pod Test Visualizer...")
    print(f"Checkpoints: {args.checkpoints}, Laps: {args.laps}")
    print(f"Heuristic: {args.heuristic}, Device: {args.device}")
    print(f"Controls: ESC=Quit, R=Reset, SPACE=Pause")
    
    visualizer = HeuristicTestVisualizer(
        num_checkpoints=args.checkpoints,
        laps=args.laps,
        device=args.device,
        heuristic_path=args.heuristic
    )
    
    visualizer.run(
        max_steps=args.max_steps,
        delay=args.delay
    )


if __name__ == "__main__":
    main()
