import queue
import threading
import time
import traceback

class VisualizationConnector:
    """Connects training processes with visualization panel"""
    
    def __init__(self, visualization_panel=None):
        self.visualization_panel = visualization_panel
        self.active = False
        self.worker_data = {}
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
                
                # Ensure worker_id is present in all data
                if 'race_state' in data and 'worker_id' not in data['race_state']:
                    data['race_state']['worker_id'] = 'main'
                elif 'iteration' in data and 'worker_id' not in data:
                    data['worker_id'] = 'main'
                
                # Debug print
                if 'iteration' in data:
                    worker_id = data.get('worker_id', 'main')
                    print(f"Visualization received metric from {worker_id}: iteration={data['iteration']}, reward={data.get('reward', 0):.4f}")
                elif 'race_state' in data:
                    worker_id = data['race_state'].get('worker_id', 'main')
                    pods_count = len(data['race_state'].get('pods', []))
                    checkpoints_count = len(data['race_state'].get('checkpoints', []))
                    # print(f"Visualization received race state from {worker_id} with {pods_count} pods and {checkpoints_count} checkpoints")
                
                # Send data to visualization panel
                if self.visualization_panel is not None:
                    try:
                        self.visualization_panel.add_metric(data)
                    except Exception as e:
                        print(f"Error sending data to visualization panel: {e}")
                        print(traceback.format_exc())
                    
            except queue.Empty:
                # Just continue if queue is empty
                time.sleep(0.1)
                continue
            except Exception as e:
                print(f"Error processing visualization data: {e}")
                print(traceback.format_exc())
