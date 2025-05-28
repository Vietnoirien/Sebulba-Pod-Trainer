import threading
import time
import queue
import wx
from collections import defaultdict, deque

class VisualizationConnector:
    """Handles recording race states during training and smooth playback for visualization"""
    
    def __init__(self, visualization_panel):
        self.visualization_panel = visualization_panel
        self.recording_queue = None
        self.playback_thread = None
        self.recording_thread = None
        self.is_running = False
        
        # Recording storage - organized by worker and timestamped
        self.recorded_states = defaultdict(deque)  # worker_id -> deque of (timestamp, race_state)
        self.recorded_metrics = deque()  # (timestamp, metrics)
        
        # Playback control
        self.playback_speed = 1.0  # 1.0 = real-time, 2.0 = 2x speed, etc.
        self.recording_start_time = None
        self.playback_start_time = None
        
        # Buffering for smooth playback
        self.playback_buffer_size = 100  # Keep last 100 states per worker
        self.metrics_buffer_size = 1000   # Keep last 1000 metrics
        
    def start(self, recording_queue):
        """Start the recording and playback system"""
        self.recording_queue = recording_queue
        self.is_running = True
        self.recording_start_time = time.time()
        self.playback_start_time = time.time()
        
        # Start recording thread (receives data from training)
        self.recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
        self.recording_thread.start()
        
        # Start playback thread (sends data to UI)
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        
        print("VisualizationConnector started - recording and playback active")
    
    def stop(self):
        """Stop the recording and playback system"""
        self.is_running = False
        
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)
        
        print("VisualizationConnector stopped")
    
    def _recording_worker(self):
        """Background thread that records incoming data from training processes"""
        print("Recording worker started")
        
        while self.is_running:
            try:
                # Get data from training processes with timeout
                data = self.recording_queue.get(timeout=0.5)
                current_time = time.time()
                relative_time = current_time - self.recording_start_time
                
                if 'race_state' in data:
                    # Record race state with timestamp
                    race_state = data['race_state']
                    worker_id = race_state.get('worker_id', 'unknown')
                    
                    # Add to recording buffer
                    self.recorded_states[worker_id].append((relative_time, race_state))
                    
                    # Limit buffer size to prevent memory issues
                    if len(self.recorded_states[worker_id]) > self.playback_buffer_size:
                        self.recorded_states[worker_id].popleft()
                
                elif 'iteration' in data:
                    # Record training metrics with timestamp
                    self.recorded_metrics.append((relative_time, data))
                    
                    # Limit buffer size
                    if len(self.recorded_metrics) > self.metrics_buffer_size:
                        self.recorded_metrics.popleft()
                
            except queue.Empty:
                # No data available, continue
                continue
            except Exception as e:
                print(f"Error in recording worker: {e}")
                continue
        
        print("Recording worker stopped")
    
    def _playback_worker(self):
        """Background thread that plays back recorded data to the UI smoothly"""
        print("Playback worker started")
        
        # Track what we've already played back
        last_played_metrics_time = 0
        last_played_race_times = defaultdict(float)  # worker_id -> last_played_time
        
        while self.is_running:
            try:
                current_time = time.time()
                playback_time = (current_time - self.playback_start_time) * self.playback_speed
                
                # Playback metrics
                self._playback_metrics(playback_time, last_played_metrics_time)
                last_played_metrics_time = playback_time
                
                # Playback race states for all workers
                for worker_id in list(self.recorded_states.keys()):
                    last_played_time = last_played_race_times[worker_id]
                    new_last_time = self._playback_race_states(worker_id, playback_time, last_played_time)
                    last_played_race_times[worker_id] = new_last_time
                
                # Sleep to maintain smooth playback (60 FPS equivalent)
                time.sleep(1.0 / 60.0)
                
            except Exception as e:
                print(f"Error in playback worker: {e}")
                time.sleep(0.1)
                continue
        
        print("Playback worker stopped")
    
    def _playback_metrics(self, current_playback_time, last_played_time):
        """Playback metrics that should be shown at current time"""
        try:
            for recorded_time, metrics in self.recorded_metrics:
                if last_played_time < recorded_time <= current_playback_time:
                    # This metric should be played back now
                    wx.CallAfter(self._send_metrics_to_ui, metrics)
        except Exception as e:
            print(f"Error in metrics playback: {e}")
    
    def _playback_race_states(self, worker_id, current_playback_time, last_played_time):
        """Playback race states for a specific worker"""
        try:
            worker_states = self.recorded_states[worker_id]
            new_last_time = last_played_time
            
            for recorded_time, race_state in worker_states:
                if last_played_time < recorded_time <= current_playback_time:
                    # This race state should be played back now
                    wx.CallAfter(self._send_race_state_to_ui, worker_id, race_state)
                    new_last_time = recorded_time
            
            return new_last_time
            
        except Exception as e:
            print(f"Error in race state playback for {worker_id}: {e}")
            return last_played_time
    
    def _send_metrics_to_ui(self, metrics):
        """Send metrics to UI thread safely"""
        try:
            if self.visualization_panel and hasattr(self.visualization_panel, '_add_metric_data'):
                # Extract the metrics data
                iteration = metrics.get('iteration', 0)
                reward = metrics.get('total_reward', metrics.get('reward', 0))
                policy_loss = metrics.get('policy_loss', 0)
                value_loss = metrics.get('value_loss', 0)
                
                # Send to visualization panel
                self.visualization_panel._add_metric_data(iteration, reward, policy_loss, value_loss)
                
        except Exception as e:
            print(f"Error sending metrics to UI: {e}")
    
    def _send_race_state_to_ui(self, worker_id, race_state):
        """Send race state to UI thread safely"""
        try:
            if self.visualization_panel and hasattr(self.visualization_panel, 'update_race_visualization_throttled'):
                # Send to visualization panel
                self.visualization_panel.update_race_visualization_throttled(race_state, worker_id)
                
        except Exception as e:
            print(f"Error sending race state to UI: {e}")
    
    def get_recording_stats(self):
        """Get statistics about the current recording"""
        total_race_states = sum(len(states) for states in self.recorded_states.values())
        total_metrics = len(self.recorded_metrics)
        active_workers = len(self.recorded_states)
        
        return {
            'total_race_states': total_race_states,
            'total_metrics': total_metrics,
            'active_workers': active_workers,
            'recording_duration': time.time() - self.recording_start_time if self.recording_start_time else 0
        }
