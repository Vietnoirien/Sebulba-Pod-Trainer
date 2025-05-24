import queue
import threading
import time
import traceback
from collections import defaultdict, deque

class VisualizationConnector:
    """Connects training processes with visualization panel"""
    
    def __init__(self, visualization_panel=None):
        self.visualization_panel = visualization_panel
        self.active = False
        self.worker_data = {}
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))  # Buffer metrics per worker
        self.race_state_buffer = {}
        self.last_update_time = defaultdict(float)
        self.update_frequency = 1.0  # Update visualization every 1 second
        print("VisualizationConnector initialized")
    
    def start(self, shared_queue):
        """Start processing visualization data from queue"""
        if not self.visualization_panel:
            print("No visualization panel provided, visualization disabled")
            return
            
        self.active = True
        print("Starting visualization connector thread")
        self.process_thread = threading.Thread(
            target=self._process_queue_data,
            args=(shared_queue,)
        )
        self.process_thread.daemon = True
        self.process_thread.start()
        print("Visualization connector thread started")
        
    def stop(self):
        """Stop processing visualization data"""
        print("Stopping visualization connector")
        self.active = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2.0)
        print("Visualization connector stopped")
    
    def _process_queue_data(self, shared_queue):
        """Process data from the queue and send to visualization panel"""
        print("Visualization data processing thread started")
        
        while self.active:
            try:
                # Get data with timeout to allow checking active flag
                data = shared_queue.get(timeout=0.5)
                
                # Process the data
                self._handle_data(data)
                    
            except queue.Empty:
                # Check if we should send buffered data
                self._check_and_send_buffered_data()
                time.sleep(0.1)
                continue
            except Exception as e:
                print(f"Error processing visualization data: {e}")
                print(traceback.format_exc())
    
    def _handle_data(self, data):
        """Handle incoming data and buffer it appropriately"""
        try:
            # Ensure worker_id is present in all data
            worker_id = self._extract_worker_id(data)
            
            if 'race_state' in data:
                self._handle_race_state(data, worker_id)
            elif 'iteration' in data:
                self._handle_metrics(data, worker_id)
            else:
                print(f"Unknown data type received: {list(data.keys())}")
                
        except Exception as e:
            print(f"Error handling data: {e}")
            print(traceback.format_exc())
    
    def _extract_worker_id(self, data):
        """Extract worker ID from data"""
        if 'race_state' in data:
            return data['race_state'].get('worker_id', 'main')
        elif 'worker_id' in data:
            return data['worker_id']
        else:
            return 'main'
    
    def _handle_race_state(self, data, worker_id):
        """Handle race state data"""
        try:
            # Ensure worker_id is in race_state
            if 'worker_id' not in data['race_state']:
                data['race_state']['worker_id'] = worker_id
            
            # Store the latest race state for this worker
            self.race_state_buffer[worker_id] = data
            
            # Send immediately for race visualization (more responsive)
            if self.visualization_panel is not None:
                try:
                    # Call the visualization panel's method directly
                    self.visualization_panel.store_race_visualization(worker_id, data['race_state'])
                    
                    # Debug info
                    pods_count = len(data['race_state'].get('pods', []))
                    checkpoints_count = len(data['race_state'].get('checkpoints', []))
                    print(f"Sent race state from {worker_id}: {pods_count} pods, {checkpoints_count} checkpoints")
                        
                except Exception as e:
                    print(f"Error sending race state to visualization panel: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"Error in _handle_race_state: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_metrics(self, data, worker_id):
        """Handle training metrics data"""
        # Ensure worker_id is in data
        if 'worker_id' not in data:
            data['worker_id'] = worker_id
        
        # Buffer the metrics
        self.metrics_buffer[worker_id].append(data)
        
        # Debug print for metrics
        iteration = data['iteration']
        reward = data.get('reward', 0)
        print(f"Buffered metric from {worker_id}: iteration={iteration}, reward={reward:.4f}")
    
    def _check_and_send_buffered_data(self):
        """Check if we should send buffered metrics data"""
        current_time = time.time()
        
        for worker_id in list(self.metrics_buffer.keys()):
            last_update = self.last_update_time[worker_id]
            
            # Send buffered metrics if enough time has passed
            if (current_time - last_update) >= self.update_frequency:
                self._send_buffered_metrics(worker_id)
                self.last_update_time[worker_id] = current_time
    
    def _send_buffered_metrics(self, worker_id):
        """Send buffered metrics for a specific worker"""
        if worker_id not in self.metrics_buffer or len(self.metrics_buffer[worker_id]) == 0:
            return
        
        # Get the latest metrics from buffer
        metrics_to_send = list(self.metrics_buffer[worker_id])
        
        # Clear the buffer
        self.metrics_buffer[worker_id].clear()
        
        # Send all buffered metrics
        if self.visualization_panel is not None:
            try:
                for metric in metrics_to_send:
                    self.visualization_panel.add_metric(metric)
                
                print(f"Sent {len(metrics_to_send)} buffered metrics from {worker_id}")
                
            except Exception as e:
                print(f"Error sending buffered metrics to visualization panel: {e}")
                print(traceback.format_exc())
    
    def set_update_frequency(self, frequency):
        """Set the update frequency for metrics (in seconds)"""
        self.update_frequency = max(0.1, frequency)  # Minimum 0.1 seconds
        print(f"Visualization update frequency set to {self.update_frequency} seconds")
    
    def get_worker_stats(self):
        """Get statistics about workers and their data"""
        stats = {}
        
        for worker_id in self.metrics_buffer:
            stats[worker_id] = {
                'buffered_metrics': len(self.metrics_buffer[worker_id]),
                'has_race_state': worker_id in self.race_state_buffer,
                'last_update': self.last_update_time.get(worker_id, 0)
            }
        
        return stats
    
    def flush_all_buffers(self):
        """Flush all buffered data immediately"""
        print("Flushing all visualization buffers")
        
        for worker_id in list(self.metrics_buffer.keys()):
            self._send_buffered_metrics(worker_id)
            self.last_update_time[worker_id] = time.time()
        
        print("All buffers flushed")
    
    def clear_worker_data(self, worker_id):
        """Clear data for a specific worker"""
        if worker_id in self.metrics_buffer:
            self.metrics_buffer[worker_id].clear()
        
        if worker_id in self.race_state_buffer:
            del self.race_state_buffer[worker_id]
        
        if worker_id in self.last_update_time:
            del self.last_update_time[worker_id]
        
        print(f"Cleared data for worker: {worker_id}")
    
    def get_latest_race_state(self, worker_id):
        """Get the latest race state for a specific worker"""
        return self.race_state_buffer.get(worker_id, None)
    
    def get_buffered_metrics_count(self, worker_id):
        """Get the number of buffered metrics for a specific worker"""
        return len(self.metrics_buffer.get(worker_id, []))
