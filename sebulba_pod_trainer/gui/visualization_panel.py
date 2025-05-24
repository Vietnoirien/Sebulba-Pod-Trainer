import wx
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import threading
import time
import os
from pathlib import Path
import queue
import numpy as np
import math

class VisualizationPanel(wx.Panel):
    def __init__(self, parent, main_frame):
        super(VisualizationPanel, self).__init__(parent)
        
        self.main_frame = main_frame
        self.config = main_frame.config
        
        # Training metrics
        self.rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.iterations = []
        
        # Queue for receiving metrics from training thread
        self.metrics_queue = queue.Queue()
        
        # Dictionary to store multiple race visualizations
        self.race_visualizations = {}
        self.active_worker_id = None
        
        # Create UI components
        self.create_ui()
        
        # Initialize plots
        self.init_plots()
        
        # Start update timer for real-time plotting
        self.update_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.update_timer)
        self.update_timer.Start(1000)  # Update every second
    
    def create_ui(self):
        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Controls section
        controls_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Model selection
        model_label = wx.StaticText(self, label="Model Directory:")
        self.model_dir_ctrl = wx.TextCtrl(self)
        self.browse_btn = wx.Button(self, label="Browse...")
        
        controls_sizer.Add(model_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        controls_sizer.Add(self.model_dir_ctrl, 1, wx.EXPAND | wx.RIGHT, 5)
        controls_sizer.Add(self.browse_btn, 0)
        
        # Refresh button
        self.refresh_btn = wx.Button(self, label="Refresh")
        controls_sizer.Add(self.refresh_btn, 0, wx.LEFT, 10)
        
        # Live monitoring toggle
        self.live_check = wx.CheckBox(self, label="Live Monitoring")
        self.live_check.SetValue(True)
        controls_sizer.Add(self.live_check, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        
        main_sizer.Add(controls_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Plots section
        plots_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create notebook for different plots
        self.plot_notebook = wx.Notebook(self)
        
        # Rewards plot panel
        self.rewards_panel = wx.Panel(self.plot_notebook)
        rewards_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create matplotlib figure for rewards
        self.rewards_figure = Figure(figsize=(8, 4), dpi=100)
        self.rewards_canvas = FigureCanvas(self.rewards_panel, -1, self.rewards_figure)
        rewards_sizer.Add(self.rewards_canvas, 1, wx.EXPAND)
        self.rewards_panel.SetSizer(rewards_sizer)
        
        # Losses plot panel
        self.losses_panel = wx.Panel(self.plot_notebook)
        losses_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create matplotlib figure for losses
        self.losses_figure = Figure(figsize=(8, 4), dpi=100)
        self.losses_canvas = FigureCanvas(self.losses_panel, -1, self.losses_figure)
        losses_sizer.Add(self.losses_canvas, 1, wx.EXPAND)
        self.losses_panel.SetSizer(losses_sizer)
        
        # Race visualization panel
        self.race_panel = wx.Panel(self.plot_notebook)
        race_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Add worker selection dropdown for multiple visualizations
        worker_sizer = wx.BoxSizer(wx.HORIZONTAL)
        worker_label = wx.StaticText(self.race_panel, label="Display Mode:")
        self.worker_choice = wx.Choice(self.race_panel, choices=["All Environments", "Single Environment"])
        self.worker_choice.SetSelection(0)  # Default to "All Environments"
        self.active_worker_id = "All Environments"
        self.worker_choice.Bind(wx.EVT_CHOICE, self.on_worker_selected)
        worker_sizer.Add(worker_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        worker_sizer.Add(self.worker_choice, 0)
        
        # Add specific worker selection (only visible in single mode)
        self.single_worker_label = wx.StaticText(self.race_panel, label="Worker:")
        self.single_worker_choice = wx.Choice(self.race_panel, choices=[])
        self.single_worker_choice.Bind(wx.EVT_CHOICE, self.on_single_worker_selected)
        worker_sizer.Add(self.single_worker_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 5)
        worker_sizer.Add(self.single_worker_choice, 0)
        
        # Initially hide single worker controls
        self.single_worker_label.Hide()
        self.single_worker_choice.Hide()
        
        race_sizer.Add(worker_sizer, 0, wx.ALL, 5)
        
        # Create matplotlib figure for race visualization
        self.race_figure = Figure(figsize=(12, 8), dpi=100)
        self.race_canvas = FigureCanvas(self.race_panel, -1, self.race_figure)
        race_sizer.Add(self.race_canvas, 1, wx.EXPAND)
        
        # Add visualization controls
        race_controls = wx.BoxSizer(wx.HORIZONTAL)
        self.visualize_race_btn = wx.Button(self.race_panel, label="Visualize Race")
        race_controls.Add(self.visualize_race_btn, 0, wx.RIGHT, 10)
        
        # Visualization frequency
        vis_freq_label = wx.StaticText(self.race_panel, label="Visualization Frequency:")
        self.vis_freq_ctrl = wx.SpinCtrl(self.race_panel, min=1, max=100, initial=10)
        race_controls.Add(vis_freq_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        race_controls.Add(self.vis_freq_ctrl, 0)
        
        race_sizer.Add(race_controls, 0, wx.ALL, 5)
        self.race_panel.SetSizer(race_sizer)
        
        # Add panels to notebook
        self.plot_notebook.AddPage(self.rewards_panel, "Rewards")
        self.plot_notebook.AddPage(self.losses_panel, "Losses")
        self.plot_notebook.AddPage(self.race_panel, "Race Visualization")
        
        plots_sizer.Add(self.plot_notebook, 1, wx.EXPAND)
        
        main_sizer.Add(plots_sizer, 1, wx.EXPAND | wx.ALL, 10)
        
        # Simulation section
        sim_box = wx.StaticBox(self, label="Simulation")
        sim_sizer = wx.StaticBoxSizer(sim_box, wx.VERTICAL)
        
        # Model selection strategy
        model_selection_sizer = wx.BoxSizer(wx.HORIZONTAL)
        model_selection_label = wx.StaticText(self, label="Model Selection:")
        self.model_selection_choice = wx.Choice(self, choices=["Best Performance", "Latest Iteration", "Manual Select"])
        self.model_selection_choice.SetSelection(0)  # Default to best performance
        self.model_selection_choice.Bind(wx.EVT_CHOICE, self.on_model_selection_changed)
        
        model_selection_sizer.Add(model_selection_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        model_selection_sizer.Add(self.model_selection_choice, 0)
        
        # Manual model selection (initially hidden)
        self.manual_model_label = wx.StaticText(self, label="Select Model:")
        self.manual_model_choice = wx.Choice(self, choices=[])
        model_selection_sizer.Add(self.manual_model_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 5)
        model_selection_sizer.Add(self.manual_model_choice, 0)
        
        # Initially hide manual controls
        self.manual_model_label.Hide()
        self.manual_model_choice.Hide()
        
        sim_sizer.Add(model_selection_sizer, 0, wx.ALL, 5)
        
        # Simulation controls
        sim_controls = wx.BoxSizer(wx.HORIZONTAL)
        
        self.sim_btn = wx.Button(self, label="Run Simulation")
        self.sim_iterations_ctrl = wx.SpinCtrl(self, min=1, max=100, initial=10)
        sim_iter_label = wx.StaticText(self, label="Iterations:")
        
        sim_controls.Add(self.sim_btn, 0, wx.RIGHT, 10)
        sim_controls.Add(sim_iter_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        sim_controls.Add(self.sim_iterations_ctrl, 0)
        
        sim_sizer.Add(sim_controls, 0, wx.ALL, 5)
        
        # Simulation results
        self.sim_results = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 150))
        sim_sizer.Add(self.sim_results, 1, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(sim_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Set sizer
        self.SetSizer(main_sizer)
        
        # Bind events
        self.browse_btn.Bind(wx.EVT_BUTTON, self.on_browse)
        self.refresh_btn.Bind(wx.EVT_BUTTON, self.on_refresh)
        self.sim_btn.Bind(wx.EVT_BUTTON, self.on_run_simulation)
        self.visualize_race_btn.Bind(wx.EVT_BUTTON, self.on_visualize_race)
    
    def on_model_selection_changed(self, event):
        """Handle model selection strategy change"""
        selection = self.model_selection_choice.GetSelection()
        if selection != wx.NOT_FOUND:
            strategy = self.model_selection_choice.GetString(selection)
            
            if strategy == "Manual Select":
                # Show manual controls and populate model list
                self.manual_model_label.Show()
                self.manual_model_choice.Show()
                self.populate_manual_model_list()
            else:
                # Hide manual controls
                self.manual_model_label.Hide()
                self.manual_model_choice.Hide()
            
            # Refresh layout
            self.Layout()
    
    def populate_manual_model_list(self):
        """Populate the manual model selection dropdown"""
        model_dir = self.model_dir_ctrl.GetValue()
        if not model_dir:
            return
            
        model_path = Path(model_dir)
        if not model_path.exists():
            return
        
        # Clear existing choices
        self.manual_model_choice.Clear()
        
        # Find all model files
        model_files = []
        
        # Standard files
        for pod_name in ["player0_pod0", "player0_pod1"]:
            standard_file = model_path / f"{pod_name}.pt"
            if standard_file.exists():
                model_files.append(f"Standard: {pod_name}.pt")
        
        # GPU-specific files
        gpu_files = list(model_path.glob("player0_pod*_gpu*.pt"))
        for gpu_file in sorted(gpu_files):
            model_files.append(f"GPU: {gpu_file.name}")
        
        # Iteration files
        iter_files = list(model_path.glob("player0_pod*_gpu*_iter*.pt"))
        for iter_file in sorted(iter_files, key=lambda x: self.extract_iteration_number(x.name)):
            iteration = self.extract_iteration_number(iter_file.name)
            model_files.append(f"Iter {iteration}: {iter_file.name}")
        
        # Add to dropdown
        for model_file in model_files:
            self.manual_model_choice.Append(model_file)
        
        if model_files:
            self.manual_model_choice.SetSelection(0)
    
    def extract_iteration_number(self, filename):
        """Extract iteration number from filename"""
        try:
            if '_iter' in filename:
                return int(filename.split('_iter')[-1].split('.')[0])
        except (ValueError, IndexError):
            pass
        return 0
    
    def init_plots(self):
        """Initialize empty plots"""
        # Rewards plot
        self.rewards_ax = self.rewards_figure.add_subplot(111)
        self.rewards_ax.set_title('Average Rewards per Iteration')
        self.rewards_ax.set_xlabel('Iteration')
        self.rewards_ax.set_ylabel('Reward')
        self.rewards_line, = self.rewards_ax.plot([], [], 'b-')
        self.rewards_canvas.draw()
        
        # Losses plot
        self.losses_ax = self.losses_figure.add_subplot(111)
        self.losses_ax.set_title('Training Losses')
        self.losses_ax.set_xlabel('Iteration')
        self.losses_ax.set_ylabel('Loss')
        self.policy_line, = self.losses_ax.plot([], [], 'r-', label='Policy Loss')
        self.value_line, = self.losses_ax.plot([], [], 'g-', label='Value Loss')
        self.losses_ax.legend()
        self.losses_canvas.draw()
        
        # Initialize the main race visualization plot
        self.init_race_plot()
    
    def init_race_plot(self):
        """Initialize the race visualization plot"""
        # Clear any existing plots
        self.race_figure.clear()
        
        # Race visualization plot - start with single subplot
        self.race_axes = {}
        self.init_single_race_plot()
        
        self.race_canvas.draw()
    
    def init_single_race_plot(self):
        """Initialize a single race plot"""
        ax = self.race_figure.add_subplot(111)
        ax.set_title('Race Visualization')
        ax.set_xlim(0, 16000)
        ax.set_ylim(0, 9000)
        ax.set_aspect('equal')
        ax.grid(True)
        
        # Store the default axis
        self.race_axes['default'] = ax
    
    def calculate_subplot_layout(self, num_workers):
        """Calculate optimal subplot layout for given number of workers"""
        if num_workers <= 1:
            return 1, 1
        elif num_workers <= 2:
            return 1, 2
        elif num_workers <= 4:
            return 2, 2
        elif num_workers <= 6:
            return 2, 3
        elif num_workers <= 9:
            return 3, 3
        else:
            # For more than 9, use a grid that accommodates all
            cols = math.ceil(math.sqrt(num_workers))
            rows = math.ceil(num_workers / cols)
            return rows, cols
    
    def init_multi_race_plots(self, worker_ids):
        """Initialize multiple race plots for different workers"""
        # Clear existing plots
        self.race_figure.clear()
        self.race_axes = {}
        
        # Filter out non-worker entries
        actual_workers = [w for w in worker_ids if w not in ["All Environments", "Single Environment"]]
        
        if len(actual_workers) <= 1:
            self.init_single_race_plot()
            return
        
        # Calculate subplot layout
        rows, cols = self.calculate_subplot_layout(len(actual_workers))
        
        # Create subplots
        for i, worker_id in enumerate(actual_workers):
            ax = self.race_figure.add_subplot(rows, cols, i + 1)
            ax.set_title(f'Environment: {worker_id}')
            ax.set_xlim(0, 16000)
            ax.set_ylim(0, 9000)
            ax.set_aspect('equal')
            ax.grid(True)
            
            # Store the axis for this worker
            self.race_axes[worker_id] = ax
        
        # Adjust layout to prevent overlap
        self.race_figure.tight_layout()
    
    def update_plots(self):
        """Update plots with current data"""
        if not self.iterations:
            return
        
        # Update rewards plot
        self.rewards_line.set_data(self.iterations, self.rewards)
        self.rewards_ax.relim()
        self.rewards_ax.autoscale_view()
        self.rewards_canvas.draw()
        
        # Update losses plot
        self.policy_line.set_data(self.iterations, self.policy_losses)
        self.value_line.set_data(self.iterations, self.value_losses)
        self.losses_ax.relim()
        self.losses_ax.autoscale_view()
        self.losses_canvas.draw()
    
    def on_timer(self, event):
        """Process any new metrics data from the queue"""
        if not self.live_check.GetValue():
            return
            
        # Process all available metrics
        updated = False
        race_updated = False
        
        while not self.metrics_queue.empty():
            try:
                metric = self.metrics_queue.get_nowait()
                
                # Handle metrics with iteration data
                if 'iteration' in metric:
                    worker_id = metric.get('worker_id', 'main')
                    iteration = metric['iteration']
                    
                    # For parallel training, we might get metrics from different workers
                    # We'll use the iteration number directly
                    self.iterations.append(iteration)
                    self.rewards.append(metric.get('reward', 0))
                    self.policy_losses.append(metric.get('policy_loss', 0))
                    self.value_losses.append(metric.get('value_loss', 0))
                    updated = True
                    
                    # Debug print
                    print(f"Added metric from {worker_id}: iteration={iteration}, reward={metric.get('reward', 0):.4f}")
                    
                # Handle race state visualization
                elif 'race_state' in metric:
                    worker_id = metric['race_state'].get('worker_id', 'main')
                    self.update_race_visualization(metric['race_state'], worker_id)
                    race_updated = True
                    
                    # Add worker to dropdown if it's new
                    self.add_worker_to_dropdown(worker_id)
                    
            except queue.Empty:
                break
        
        if updated:
            self.update_plots()
    
    def add_worker_to_dropdown(self, worker_id):
        """Add a worker to the dropdown if it's not already there"""
        # Add to single worker dropdown
        single_worker_choices = [self.single_worker_choice.GetString(i) for i in range(self.single_worker_choice.GetCount())]
        if worker_id not in single_worker_choices:
            self.single_worker_choice.Append(worker_id)
        
        # Check if we need to update the race plot layout
        if self.active_worker_id == "All Environments":
            self.update_race_plot_layout()
    
    def update_race_plot_layout(self):
        """Update race plot layout based on current workers"""
        if self.active_worker_id == "All Environments":
            worker_ids = list(self.race_visualizations.keys())
            self.init_multi_race_plots(worker_ids)
            # Redraw all current visualizations
            for worker_id, race_state in self.race_visualizations.items():
                self.draw_single_race_environment(worker_id, race_state)
            self.race_canvas.draw()
    
    def on_worker_selected(self, event):
        """Handle display mode selection"""
        selection = self.worker_choice.GetSelection()
        if selection != wx.NOT_FOUND:
            mode = self.worker_choice.GetString(selection)
            self.active_worker_id = mode
            
            if mode == "All Environments":
                # Hide single worker controls
                self.single_worker_label.Hide()
                self.single_worker_choice.Hide()
                # Update to multi-plot view
                self.update_race_plot_layout()
            else:  # Single Environment
                # Show single worker controls
                self.single_worker_label.Show()
                self.single_worker_choice.Show()
                # Initialize single plot
                self.init_single_race_plot()
                # Update with selected worker if any
                if self.single_worker_choice.GetCount() > 0:
                    self.single_worker_choice.SetSelection(0)
                    self.on_single_worker_selected(None)
            
            # Refresh layout
            self.race_panel.Layout()
    
    def on_single_worker_selected(self, event):
        """Handle single worker selection"""
        if self.active_worker_id != "Single Environment":
            return
            
        selection = self.single_worker_choice.GetSelection()
        if selection != wx.NOT_FOUND:
            worker_id = self.single_worker_choice.GetString(selection)
            self.update_single_worker_visualization(worker_id)
    
    def get_worker_colors(self, worker_id):
        """Get consistent colors for a worker"""
        # Create a consistent color scheme for workers
        worker_color_map = {
            'main': ['blue', 'cyan', 'red', 'orange'],
            'visualization': ['blue', 'cyan', 'red', 'orange'],
            'gpu_0': ['darkblue', 'lightblue', 'darkred', 'coral'],
            'gpu_1': ['navy', 'skyblue', 'maroon', 'salmon'],
            'gpu_2': ['mediumblue', 'powderblue', 'firebrick', 'lightsalmon'],
            'gpu_3': ['royalblue', 'lightcyan', 'crimson', 'peachpuff'],
        }
        
        # Generate colors for unknown workers
        if worker_id not in worker_color_map:
            worker_hash = hash(worker_id) % 8
            base_colors = [
                ['purple', 'plum', 'green', 'lime'],
                ['brown', 'tan', 'teal', 'turquoise'],
                ['olive', 'yellow', 'pink', 'magenta'],
                ['gray', 'silver', 'indigo', 'violet'],
                ['darkgreen', 'lightgreen', 'darkgoldenrod', 'gold'],
                ['darkcyan', 'aqua', 'darkmagenta', 'fuchsia'],
                ['darkslategray', 'lightgray', 'darkkhaki', 'khaki'],
                ['midnightblue', 'lightsteelblue', 'darkslateblue', 'mediumpurple']
            ]
            worker_color_map[worker_id] = base_colors[worker_hash]
        
        return worker_color_map[worker_id]
    
    def draw_single_race_environment(self, worker_id, race_state):
        """Draw a single race environment on its designated axis"""
        # Get the appropriate axis
        if worker_id in self.race_axes:
            ax = self.race_axes[worker_id]
        elif 'default' in self.race_axes:
            ax = self.race_axes['default']
        else:
            return
        
        # Clear the axis
        ax.clear()
        ax.set_title(f'Environment: {worker_id}')
        ax.set_xlim(0, 16000)
        ax.set_ylim(0, 9000)
        ax.set_aspect('equal')
        ax.grid(True)
        
        worker_colors = self.get_worker_colors(worker_id)
        
        # Draw checkpoints
        if 'checkpoints' in race_state and race_state['checkpoints']:
            checkpoints = race_state['checkpoints']
            for i, (x, y) in enumerate(checkpoints):
                ax.add_patch(
                    matplotlib.patches.Circle((x, y), 600, fill=False, edgecolor='green', linewidth=2)
                )
                # Add checkpoint number
                ax.text(x, y, str(i), fontsize=10, ha='center', va='center')
        
        # Draw pods
        if 'pods' in race_state:
            pods = race_state['pods']
            
            for i, pod in enumerate(pods):
                if 'position' not in pod or len(pod['position']) < 2:
                    continue  # Skip pods with invalid position data
                    
                x, y = pod['position']
                angle = pod.get('angle', 0)
                color = worker_colors[i % len(worker_colors)]
                
                # Draw collision radius circle (400 units)
                collision_circle = matplotlib.patches.Circle(
                    (x, y), 400, 
                    fill=False, 
                    edgecolor=color, 
                    linestyle='--', 
                    alpha=0.5,
                    linewidth=1
                )
                ax.add_patch(collision_circle)
                
                # Draw pod as a triangle pointing in the direction of travel
                dx = 400 * np.cos(np.radians(angle))
                dy = 400 * np.sin(np.radians(angle))
                
                ax.arrow(
                    x, y, dx, dy, 
                    head_width=200, head_length=300, 
                    fc=color, ec=color, alpha=0.7
                )
                
                # Add pod label
                ax.text(x, y-600, f'P{i}', fontsize=8, ha='center', va='center', 
                       color=color, weight='bold')
                
                # Add pod trail if available
                if 'trail' in pod and len(pod['trail']) > 1:
                    trail = pod['trail'][-30:]  # Show last 30 points
                    if len(trail) > 1:
                        trail_x, trail_y = zip(*trail)
                        ax.plot(trail_x, trail_y, color=color, alpha=0.3, linewidth=1)
        
        # Add legend for single environment view
        if worker_id in self.race_axes or len(self.race_axes) == 1:
            legend_elements = [
                matplotlib.patches.Patch(color='green', label='Checkpoints (600 radius)'),
                matplotlib.patches.Patch(color=worker_colors[0], label='Pod 0 (Player 0)'),
                matplotlib.patches.Patch(color=worker_colors[1], label='Pod 1 (Player 0)'),
                matplotlib.patches.Patch(color=worker_colors[2], label='Pod 2 (Player 1)'),
                matplotlib.patches.Patch(color=worker_colors[3], label='Pod 3 (Player 1)'),
                matplotlib.lines.Line2D([0], [0], linestyle='--', color='gray', alpha=0.5, label='Collision Radius (400)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=6)
    
    def update_single_worker_visualization(self, worker_id):
        """Update visualization for a single worker in single environment mode"""
        if worker_id not in self.race_visualizations:
            return
            
        race_state = self.race_visualizations[worker_id]
        self.draw_single_race_environment(worker_id, race_state)
        self.race_canvas.draw()

    def add_metric(self, metric_dict):
        """Add a metric to the queue (called from training thread)"""
        try:
            # Handle race state visualization
            if 'race_state' in metric_dict:
                race_state = metric_dict['race_state']
                worker_id = race_state.get('worker_id', 'main')
                
                # Store the race state for this worker
                wx.CallAfter(self.store_race_visualization, worker_id, race_state)
                return
                
            # Handle metrics with iteration data
            if 'iteration' in metric_dict:
                worker_id = metric_dict.get('worker_id', 'main')
                iteration = metric_dict['iteration']
                
                # For parallel training, we might get metrics from different workers
                # We'll use the iteration number directly
                wx.CallAfter(self._add_metric_data, 
                            iteration, 
                            metric_dict.get('reward', 0), 
                            metric_dict.get('policy_loss', 0), 
                            metric_dict.get('value_loss', 0))
                
                # Debug print
                print(f"Added metric from {worker_id}: iteration={iteration}, reward={metric_dict.get('reward', 0):.4f}")
        except Exception as e:
            import traceback
            print(f"Error in add_metric: {e}")
            print(traceback.format_exc())

    def store_race_visualization(self, worker_id, race_state):
        """Store race state for a specific worker (thread-safe)"""
        try:
            # Ensure we're on the main thread
            if not wx.IsMainThread():
                wx.CallAfter(self.store_race_visualization, worker_id, race_state)
                return
            
            # Store the race state
            self.race_visualizations[worker_id] = race_state
            
            # Add worker to dropdown if it's new
            self.add_worker_to_dropdown(worker_id)
            
            # Update visualization based on current selection
            if self.active_worker_id == "All Environments":
                # Check if we need to update layout (new worker added)
                if worker_id not in self.race_axes:
                    self.update_race_plot_layout()
                else:
                    # Just update this worker's plot
                    self.draw_single_race_environment(worker_id, race_state)
                    self.race_canvas.draw()
            elif self.active_worker_id == "Single Environment":
                # Update if this is the selected worker
                selection = self.single_worker_choice.GetSelection()
                if selection != wx.NOT_FOUND:
                    selected_worker = self.single_worker_choice.GetString(selection)
                    if worker_id == selected_worker:
                        self.update_single_worker_visualization(worker_id)
            
            # Switch to the race visualization tab
            self.plot_notebook.SetSelection(2)  # Index 2 is the Race Visualization tab
            
            # Debug output
            pods_count = len(race_state.get('pods', []))
            checkpoints_count = len(race_state.get('checkpoints', []))
            print(f"Updated race visualization for {worker_id}: {pods_count} pods, {checkpoints_count} checkpoints")
            
        except Exception as e:
            print(f"Error in store_race_visualization: {e}")
            import traceback
            traceback.print_exc()

    def add_metric(self, metric_dict):
        """Add a metric to the queue (called from training thread)"""
        try:
            # Handle race state visualization
            if 'race_state' in metric_dict:
                race_state = metric_dict['race_state']
                worker_id = race_state.get('worker_id', 'main')
                
                # Store the race state for this worker (thread-safe)
                self.store_race_visualization(worker_id, race_state)
                return
                
            # Handle metrics with iteration data
            if 'iteration' in metric_dict:
                worker_id = metric_dict.get('worker_id', 'main')
                iteration = metric_dict['iteration']
                
                # Add metric data (thread-safe)
                wx.CallAfter(self._add_metric_data, 
                            iteration, 
                            metric_dict.get('reward', 0), 
                            metric_dict.get('policy_loss', 0), 
                            metric_dict.get('value_loss', 0))
                
                print(f"Added metric from {worker_id}: iteration={iteration}, reward={metric_dict.get('reward', 0):.4f}")
                
        except Exception as e:
            print(f"Error in add_metric: {e}")
            import traceback
            traceback.print_exc()

    def update_race_visualization(self, race_state, worker_id='main'):
        """Update the race visualization with new state data"""
        try:
            # Store the race state for this worker
            self.race_visualizations[worker_id] = race_state
            
            # Update the visualization based on current mode
            if self.active_worker_id == "All Environments":
                # Update or create subplot for this worker
                if worker_id not in self.race_axes:
                    self.update_race_plot_layout()
                else:
                    self.draw_single_race_environment(worker_id, race_state)
                    self.race_canvas.draw()
            elif self.active_worker_id == "Single Environment":
                # Update if this is the selected worker
                selection = self.single_worker_choice.GetSelection()
                if selection != wx.NOT_FOUND:
                    selected_worker = self.single_worker_choice.GetString(selection)
                    if worker_id == selected_worker:
                        self.update_single_worker_visualization(worker_id)
                
        except Exception as e:
            import traceback
            print(f"Error in update_race_visualization: {e}")
            print(traceback.format_exc())

    def load_training_data(self, model_dir):
        """Load training data from model directory"""
        try:
            # Reset data
            self.rewards = []
            self.policy_losses = []
            self.value_losses = []
            self.iterations = []
            
            # Check if directory exists
            model_path = Path(model_dir)
            if not model_path.exists():
                wx.MessageBox(f"Model directory does not exist: {model_dir}", "Directory Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Check for model files - support both standard and GPU-specific filenames
            standard_files_exist = (model_path / "player0_pod0.pt").exists() and (model_path / "player0_pod1.pt").exists()
            
            # Look for GPU-specific model files (from parallel training)
            gpu_files = list(model_path.glob("player0_pod0_gpu*.pt")) + list(model_path.glob("player0_pod0_gpu*_iter*.pt"))
            gpu_files_exist = len(gpu_files) > 0
            
            if not (standard_files_exist or gpu_files_exist):
                wx.MessageBox(f"Model files not found in directory: {model_dir}\n"
                            f"Expected either standard files (player0_pod0.pt) or "
                            f"GPU-specific files (player0_pod0_gpu0.pt)", 
                            "Model Files Error", wx.OK | wx.ICON_ERROR)
            
            # Look for training log files - support both standard and GPU-specific logs
            standard_log_path = model_path / "training_log.txt"
            gpu_log_paths = list(model_path.glob("training_log_gpu_*.txt"))
            
            log_files = [standard_log_path] if standard_log_path.exists() else []
            log_files.extend([p for p in gpu_log_paths if p.exists()])
            
            if not log_files:
                wx.MessageBox(f"No training log files found in: {model_path}", "Log File Error", wx.OK | wx.ICON_WARNING)
                return
            
            # Parse all log files
            for log_path in log_files:
                with open(log_path, 'r') as f:
                    for line in f:
                        try:
                            parts = line.strip().split(',')
                            if len(parts) >= 4:
                                iteration = int(parts[0].split('=')[1])
                                reward = float(parts[1].split('=')[1])
                                policy_loss = float(parts[2].split('=')[1])
                                value_loss = float(parts[3].split('=')[1])
                                
                                self.iterations.append(iteration)
                                self.rewards.append(reward)
                                self.policy_losses.append(policy_loss)
                                self.value_losses.append(value_loss)
                        except Exception as parse_error:
                            print(f"Error parsing line: {line} - {str(parse_error)}")
                            continue
            
            # Update plots
            if not self.iterations:
                wx.MessageBox("No valid training data found in log files", "Data Error", wx.OK | wx.ICON_WARNING)
                return
                
            self.update_plots()
            
            # Populate manual model list if in manual mode
            if self.model_selection_choice.GetSelection() == 2:  # Manual Select
                self.populate_manual_model_list()
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            wx.MessageBox(f"Error loading training data: {str(e)}\n\nDetails:\n{error_details}", 
                        "Data Loading Error", wx.OK | wx.ICON_ERROR)
    
    def on_browse(self, event):
        """Browse for model directory"""
        with wx.DirDialog(self, "Select Model Directory") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                self.model_dir_ctrl.SetValue(path)
                self.load_training_data(path)
    
    def on_refresh(self, event):
        """Refresh training data"""
        model_dir = self.model_dir_ctrl.GetValue()
        if model_dir:
            self.load_training_data(model_dir)
    
    def find_best_model_file(self, pod_name, model_path):
        """Find the best model file based on selection strategy"""
        selection = self.model_selection_choice.GetSelection()
        strategy = self.model_selection_choice.GetString(selection)
        
        if strategy == "Manual Select":
            return self.get_manual_selected_model(pod_name, model_path)
        elif strategy == "Latest Iteration":
            return self.find_latest_model_file(pod_name, model_path)
        else:  # Best Performance
            return self.find_best_performance_model_file(pod_name, model_path)
    
    def get_manual_selected_model(self, pod_name, model_path):
        """Get manually selected model file"""
        selection = self.manual_model_choice.GetSelection()
        if selection == wx.NOT_FOUND:
            return None
            
        selected_text = self.manual_model_choice.GetString(selection)
        
        # Extract filename from the display text
        if ": " in selected_text:
            filename = selected_text.split(": ", 1)[1]
            model_file = model_path / filename
            
            # Check if this file matches the requested pod
            if pod_name in filename and model_file.exists():
                return model_file
        
        # Fallback to best performance if manual selection doesn't match
        return self.find_best_performance_model_file(pod_name, model_path)
    
    def find_latest_model_file(self, pod_name, model_path):
        """Find the latest model file by iteration number"""
        # First try standard filename
        standard_file = model_path / f"{pod_name}.pt"
        if standard_file.exists():
            return standard_file
        
        # Collect all iteration files for this pod
        iter_files = list(model_path.glob(f"{pod_name}_gpu*_iter*.pt"))
        
        if not iter_files:
            # Try GPU files without iteration
            gpu_files = list(model_path.glob(f"{pod_name}_gpu*.pt"))
            return gpu_files[0] if gpu_files else None
        
        # Sort by iteration number and return the latest
        iter_files_with_numbers = []
        for file in iter_files:
            try:
                iter_num = self.extract_iteration_number(file.name)
                iter_files_with_numbers.append((iter_num, file))
            except (ValueError, IndexError):
                continue
        
        if iter_files_with_numbers:
            iter_files_with_numbers.sort(key=lambda x: x[0], reverse=True)
            return iter_files_with_numbers[0][1]
        
        return None
    
    def find_best_performance_model_file(self, pod_name, model_path):
        """Find the best model file based on training performance"""
        # First try standard filename (usually the final/best model)
        standard_file = model_path / f"{pod_name}.pt"
        if standard_file.exists():
            return standard_file
        
        # Collect all possible model files for this pod
        all_model_files = []
        
        # GPU-specific files without iteration (usually latest/best)
        gpu_files = list(model_path.glob(f"{pod_name}_gpu*.pt"))
        all_model_files.extend(gpu_files)
        
        # GPU-specific files with iteration
        iter_files = list(model_path.glob(f"{pod_name}_gpu*_iter*.pt"))
        all_model_files.extend(iter_files)
        
        if not all_model_files:
            return None
        
        # Try to find the best model based on training logs
        best_model = self.find_best_model_from_logs(all_model_files, model_path)
        if best_model:
            return best_model
        
        # Fallback: sort by iteration number and take the latest
        iter_files_with_numbers = []
        for file in all_model_files:
            try:
                if '_iter' in str(file):
                    iter_num = self.extract_iteration_number(file.name)
                    iter_files_with_numbers.append((iter_num, file))
            except (ValueError, IndexError):
                continue
        
        if iter_files_with_numbers:
            # Sort by iteration number and return the latest
            iter_files_with_numbers.sort(key=lambda x: x[0], reverse=True)
            return iter_files_with_numbers[0][1]
        
        # Final fallback: return the first file
        return all_model_files[0]

    def find_best_model_from_logs(self, model_files, model_path):
        """Find the best model based on training log performance"""
        # Look for training log files
        log_files = []
        standard_log = model_path / "training_log.txt"
        if standard_log.exists():
            log_files.append(standard_log)
        
        gpu_logs = list(model_path.glob("training_log_gpu_*.txt"))
        log_files.extend(gpu_logs)
        
        if not log_files:
            return None
        
        # Parse logs to find best performing iterations
        best_iteration = None
        best_reward = float('-inf')
        iteration_rewards = {}
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            parts = line.strip().split(',')
                            if len(parts) >= 2:
                                iteration = int(parts[0].split('=')[1])
                                reward = float(parts[1].split('=')[1])
                                
                                # Track the best reward and its iteration
                                if reward > best_reward:
                                    best_reward = reward
                                    best_iteration = iteration
                                
                                # Store all iteration rewards for averaging
                                if iteration not in iteration_rewards:
                                    iteration_rewards[iteration] = []
                                iteration_rewards[iteration].append(reward)
                                
                        except (ValueError, IndexError):
                            continue
            except Exception:
                continue
        
        # If we found a best iteration, look for corresponding model file
        if best_iteration is not None:
            for model_file in model_files:
                if f'_iter{best_iteration}.' in str(model_file):
                    wx.CallAfter(self.sim_results.AppendText, 
                               f"Selected best model: {model_file.name} (iteration {best_iteration}, reward {best_reward:.4f})\n")
                    return model_file
        
        # Alternative: find iteration with best average reward over multiple runs
        if iteration_rewards:
            avg_rewards = {iter_num: sum(rewards)/len(rewards) 
                          for iter_num, rewards in iteration_rewards.items()}
            best_avg_iteration = max(avg_rewards.keys(), key=lambda k: avg_rewards[k])
            best_avg_reward = avg_rewards[best_avg_iteration]
            
            for model_file in model_files:
                if f'_iter{best_avg_iteration}.' in str(model_file):
                    wx.CallAfter(self.sim_results.AppendText, 
                               f"Selected best average model: {model_file.name} (iteration {best_avg_iteration}, avg reward {best_avg_reward:.4f})\n")
                    return model_file
        
        return None
    
    def on_visualize_race(self, event):
        """Start race visualization"""
        model_dir = self.model_dir_ctrl.GetValue()
        if not model_dir:
            wx.MessageBox("Please select a model directory first", "Visualization Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Check if model directory exists
        model_path = Path(model_dir)
        if not model_path.exists():
            wx.MessageBox(f"Model directory does not exist: {model_dir}", "Visualization Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Check if model files exist - support both standard and GPU-specific filenames
        standard_files_exist = (model_path / "player0_pod0.pt").exists() and (model_path / "player0_pod1.pt").exists()
        
        # Look for GPU-specific model files (from parallel training)
        gpu_files = list(model_path.glob("player0_pod0_gpu*.pt")) + list(model_path.glob("player0_pod1_gpu*.pt"))
        gpu_files_exist = len(gpu_files) >= 2  # Need at least one file for each pod
        
        if not (standard_files_exist or gpu_files_exist):
            wx.MessageBox(f"Model files not found in directory: {model_dir}\n"
                        f"Expected either standard files (player0_pod0.pt) or "
                        f"GPU-specific files (player0_pod0_gpu0.pt)", 
                        "Visualization Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Start visualization in a separate thread
        self.visualize_race_btn.Disable()
        
        vis_thread = threading.Thread(target=self.run_race_visualization, args=(model_dir,))
        vis_thread.daemon = True
        vis_thread.start()
    
    def run_race_visualization(self, model_dir):
        """Run race visualization with the selected model"""
        try:
            import torch
            from sebulba_pod_trainer.models.neural_pod import PodNetwork
            from sebulba_pod_trainer.environment.optimized_race_env import OptimizedRaceEnvironment as RaceEnvironment
            
            # Create environment
            env = RaceEnvironment(batch_size=1, device=torch.device('cpu'))
            
            # Load models
            pod_networks = {}
            
            # Reset race visualization
            wx.CallAfter(self.race_figure.clear)
            wx.CallAfter(self.init_race_plot)
            wx.CallAfter(self.race_canvas.draw)
            
            # Reset environment to get initial observations
            observations = env.reset()
            
            # Get observation dimension from the environment
            obs_dim = next(iter(observations.values())).shape[1]

            # Load player 0 models
            model_path = Path(model_dir)
            
            # Function to create network with matching architecture
            def create_network_from_checkpoint(model_file):
                """Create a PodNetwork that matches the saved checkpoint architecture"""
                # Load the checkpoint to inspect its structure
                checkpoint = torch.load(model_file, map_location='cpu')
                
                # Try to infer architecture from checkpoint keys and shapes
                hidden_layers = []
                
                # Look for encoder layers to determine hidden layer sizes
                encoder_keys = [k for k in checkpoint.keys() if k.startswith('encoder.') and k.endswith('.weight')]
                encoder_keys.sort(key=lambda x: int(x.split('.')[1]))  # Sort by layer index
                
                for key in encoder_keys:
                    layer_idx = int(key.split('.')[1])
                    if layer_idx % 2 == 0:  # Only weight layers (not activation layers)
                        weight_shape = checkpoint[key].shape
                        hidden_size = weight_shape[0]  # Output size of this layer
                        hidden_layers.append({'type': 'Linear+ReLU', 'size': hidden_size})
                
                # Get head sizes from checkpoint
                policy_hidden_size = 16  # Default
                value_hidden_size = 16   # Default
                special_hidden_size = 16 # Default
                
                if 'policy_head.0.weight' in checkpoint:
                    policy_hidden_size = checkpoint['policy_head.0.weight'].shape[0]
                if 'value_head.0.weight' in checkpoint:
                    value_hidden_size = checkpoint['value_head.0.weight'].shape[0]
                if 'special_action_head.0.weight' in checkpoint:
                    special_hidden_size = checkpoint['special_action_head.0.weight'].shape[0]
                
                # Create network with matching architecture
                network = PodNetwork(
                    observation_dim=obs_dim,
                    hidden_layers=hidden_layers,
                    policy_hidden_size=policy_hidden_size,
                    value_hidden_size=value_hidden_size,
                    special_hidden_size=special_hidden_size
                )
                
                return network, checkpoint
            
            # Load player 0 models
            for i in range(2):
                pod_key = f"player0_pod{i}"
                model_file = self.find_best_model_file(pod_key, model_path)
                if model_file:
                    try:
                        network, checkpoint = create_network_from_checkpoint(model_file)
                        network.load_state_dict(checkpoint)
                        network.eval()
                        pod_networks[pod_key] = network
                        wx.CallAfter(self.sim_results.AppendText, f"Loaded model from {model_file.name}\n")
                    except Exception as e:
                        wx.CallAfter(self.sim_results.AppendText, f"Error loading {pod_key} from {model_file}: {str(e)}\n")
                        # Create a default network as fallback
                        pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
                else:
                    wx.CallAfter(self.sim_results.AppendText, f"No model found for {pod_key}\n")
                    # Create a default network
                    pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
            
            # Use the same model for opponent (player1)
            for i in range(2):
                pod_key = f"player1_pod{i}"
                source_pod_key = f"player0_pod{i}"
                
                if source_pod_key in pod_networks:
                    # Clone the network architecture and weights
                    source_network = pod_networks[source_pod_key]
                    
                    # Create new network with same architecture
                    model_file = self.find_best_model_file(source_pod_key, model_path)
                    if model_file:
                        try:
                            network, checkpoint = create_network_from_checkpoint(model_file)
                            network.load_state_dict(checkpoint)
                            network.eval()
                            pod_networks[pod_key] = network
                        except Exception as e:
                            wx.CallAfter(self.sim_results.AppendText, f"Error loading {pod_key}: {str(e)}\n")
                            # Use default network as fallback
                            pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
                    else:
                        # Use default network
                        pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
                else:
                    # Use default network
                    pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
            
            # Get checkpoint positions from environment
            checkpoints = env.get_checkpoints()
            
            # Create race state for visualization
            race_state = {
                'checkpoints': checkpoints,
                'pods': [],
                'worker_id': 'visualization'
            }
            
            # Add pod trails
            pod_trails = {pod_key: [] for pod_key in pod_networks.keys()}
            
            # Visualization frequency
            vis_freq = self.vis_freq_ctrl.GetValue()
            
            # Run race
            done = False
            max_steps = 200  # Limit steps to avoid infinite loops
            
            for step in range(max_steps):
                if done:
                    break
                
                # Get actions from all pod networks
                actions = {}
                with torch.no_grad():
                    for pod_key, network in pod_networks.items():
                        # Get deterministic action from network
                        if pod_key in observations:
                            obs = observations[pod_key]
                            action = network.get_actions(obs, deterministic=True)
                            actions[pod_key] = action
                
                # Step the environment
                observations, rewards, done, info = env.step(actions)
                
                # Update pod trails using actual pod states instead of observations
                pod_states = env.get_pod_states()
                for pod_idx, pod_state in enumerate(pod_states):
                    pod_key = f"player{pod_idx//2}_pod{pod_idx%2}"
                    if pod_key in pod_networks:
                        position = pod_state['position']
                        pod_trails[pod_key].append(position)

                # Visualize every vis_freq steps
                if step % vis_freq == 0:
                    # Update race state for visualization using actual pod states
                    race_state['pods'] = []
                    
                    for pod_idx, pod_state in enumerate(pod_states):
                        pod_key = f"player{pod_idx//2}_pod{pod_idx%2}"
                        if pod_key in pod_networks:
                            pod_info = {
                                'position': pod_state['position'],
                                'angle': pod_state['angle'],  # This is the actual absolute angle
                                'trail': pod_trails[pod_key]
                            }
                            race_state['pods'].append(pod_info)

                    # Update visualization
                    wx.CallAfter(self.update_race_visualization, race_state, 'visualization')
                    
                    # Small delay to make visualization visible
                    time.sleep(0.1) 
            
            # Final update
            wx.CallAfter(self.update_race_visualization, race_state, 'visualization')
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            wx.CallAfter(wx.MessageBox, f"Error during race visualization: {str(e)}\n\nDetails:\n{error_details}", 
                        "Visualization Error", wx.OK | wx.ICON_ERROR)
        
        finally:
            wx.CallAfter(self.visualize_race_btn.Enable)
    
    def on_run_simulation(self, event):
        """Run a simulation with the selected model"""
        model_dir = self.model_dir_ctrl.GetValue()
        if not model_dir:
            wx.MessageBox("Please select a model directory first", "Simulation Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Check if model directory exists
        model_path = Path(model_dir)
        if not model_path.exists():
            wx.MessageBox(f"Model directory does not exist: {model_dir}", "Simulation Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Check if model files exist - support both standard and GPU-specific filenames
        standard_files_exist = (model_path / "player0_pod0.pt").exists() and (model_path / "player0_pod1.pt").exists()
        
        # Look for GPU-specific model files (from parallel training)
        gpu_files = list(model_path.glob("player0_pod0_gpu*.pt")) + list(model_path.glob("player0_pod1_gpu*.pt"))
        gpu_files_exist = len(gpu_files) >= 2  # Need at least one file for each pod
        
        if not (standard_files_exist or gpu_files_exist):
            wx.MessageBox(f"Model files not found in directory: {model_dir}\n"
                        f"Expected either standard files (player0_pod0.pt) or "
                        f"GPU-specific files (player0_pod0_gpu0.pt)", 
                        "Simulation Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Start simulation in a separate thread
        self.sim_btn.Disable()
        self.sim_results.Clear()
        self.sim_results.AppendText("Starting simulation...\n")
        
        sim_thread = threading.Thread(target=self.run_simulation, args=(model_dir,))
        sim_thread.daemon = True
        sim_thread.start()
    
    def run_simulation(self, model_dir):
        """Run simulation with the selected model"""
        try:
            import torch
            from sebulba_pod_trainer.models.neural_pod import PodNetwork
            from sebulba_pod_trainer.environment.optimized_race_env import OptimizedRaceEnvironment as RaceEnvironment
            
            # Create environment
            env = RaceEnvironment(batch_size=1, device=torch.device('cpu'))
            
            # Load models
            pod_networks = {}
            
            # Reset environment to get initial observations
            observations = env.reset()
            
            # Get observation dimension from the environment
            obs_dim = next(iter(observations.values())).shape[1]
            
            model_path = Path(model_dir)
                        
            # Function to create network with matching architecture
            def create_network_from_checkpoint(model_file):
                """Create a PodNetwork that matches the saved checkpoint architecture"""
                # Load the checkpoint to inspect its structure
                checkpoint = torch.load(model_file, map_location='cpu')
                
                # Try to infer architecture from checkpoint keys and shapes
                hidden_layers = []
                
                # Look for encoder layers to determine hidden layer sizes
                encoder_keys = [k for k in checkpoint.keys() if k.startswith('encoder.') and k.endswith('.weight')]
                encoder_keys.sort(key=lambda x: int(x.split('.')[1]))  # Sort by layer index
                
                for key in encoder_keys:
                    layer_idx = int(key.split('.')[1])
                    if layer_idx % 2 == 0:  # Only weight layers (not activation layers)
                        weight_shape = checkpoint[key].shape
                        hidden_size = weight_shape[0]  # Output size of this layer
                        hidden_layers.append({'type': 'Linear+ReLU', 'size': hidden_size})
                
                # Get head sizes from checkpoint
                policy_hidden_size = 16  # Default
                value_hidden_size = 16   # Default
                special_hidden_size = 16 # Default
                
                if 'policy_head.0.weight' in checkpoint:
                    policy_hidden_size = checkpoint['policy_head.0.weight'].shape[0]
                if 'value_head.0.weight' in checkpoint:
                    value_hidden_size = checkpoint['value_head.0.weight'].shape[0]
                if 'special_action_head.0.weight' in checkpoint:
                    special_hidden_size = checkpoint['special_action_head.0.weight'].shape[0]
                
                # Create network with matching architecture
                network = PodNetwork(
                    observation_dim=obs_dim,
                    hidden_layers=hidden_layers,
                    policy_hidden_size=policy_hidden_size,
                    value_hidden_size=value_hidden_size,
                    special_hidden_size=special_hidden_size
                )
                
                return network, checkpoint
            
            # Load models for player 0
            for i in range(2):
                pod_key = f"player0_pod{i}"
                model_file = self.find_best_model_file(pod_key, model_path)
                
                if model_file:
                    try:
                        network, checkpoint = create_network_from_checkpoint(model_file)
                        network.load_state_dict(checkpoint)
                        network.eval()
                        pod_networks[pod_key] = network
                        wx.CallAfter(self.sim_results.AppendText, f"Loaded model from {model_file.name}\n")
                    except Exception as e:
                        wx.CallAfter(self.sim_results.AppendText, f"Error loading {pod_key} from {model_file}: {str(e)}\n")
                        # Create a default network as fallback
                        pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
                else:
                    wx.CallAfter(self.sim_results.AppendText, f"No model found for {pod_key}, using random initialization\n")
                    # Create a default network
                    pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
            
            # Use the same model for opponent (player1)
            for i in range(2):
                pod_key = f"player1_pod{i}"
                source_pod_key = f"player0_pod{i}"
                
                # Use player0's model for player1
                model_file = self.find_best_model_file(source_pod_key, model_path)
                
                if model_file:
                    try:
                        network, checkpoint = create_network_from_checkpoint(model_file)
                        network.load_state_dict(checkpoint)
                        network.eval()
                        pod_networks[pod_key] = network
                    except Exception as e:
                        wx.CallAfter(self.sim_results.AppendText, f"Error loading {pod_key}: {str(e)}\n")
                        # Use default network as fallback
                        pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
                else:
                    # Use default network
                    pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
            
            # Run simulation
            iterations = self.sim_iterations_ctrl.GetValue()
            
            wx.CallAfter(self.sim_results.AppendText, f"Running {iterations} race(s)...\n")
            
            player0_wins = 0
            player1_wins = 0
            
            for i in range(iterations):
                wx.CallAfter(self.sim_results.AppendText, f"Race {i+1}/{iterations}...\n")
                
                observations = env.reset()
                done = False
                max_steps = 1000  # Safety limit
                
                for step in range(max_steps):
                    if done:
                        break
                    
                    # Get actions from all pod networks
                    actions = {}
                    with torch.no_grad():
                        for pod_key, network in pod_networks.items():
                            # Get deterministic action from network
                            if pod_key in observations:
                                obs = observations[pod_key]
                                action = network.get_actions(obs, deterministic=True)
                                actions[pod_key] = action
                    
                    # Step the environment
                    observations, _, done, info = env.step(actions)
                
                # Determine winner
                player0_progress = max(
                    info["checkpoint_progress"][0][0].item(),
                    info["checkpoint_progress"][1][0].item()
                )
                player1_progress = max(
                    info["checkpoint_progress"][2][0].item(),
                    info["checkpoint_progress"][3][0].item()
                )
                
                if player0_progress > player1_progress:
                    player0_wins += 1
                    wx.CallAfter(self.sim_results.AppendText, f"  Result: Player 0 wins! Progress: {player0_progress} vs {player1_progress}\n")
                else:
                    player1_wins += 1
                    wx.CallAfter(self.sim_results.AppendText, f"  Result: Player 1 wins! Progress: {player1_progress} vs {player0_progress}\n")
            
            # Show final results
            wx.CallAfter(self.sim_results.AppendText, f"\nFinal Results:\n")
            wx.CallAfter(self.sim_results.AppendText, f"Player 0 wins: {player0_wins}/{iterations} ({player0_wins/iterations*100:.1f}%)\n")
            wx.CallAfter(self.sim_results.AppendText, f"Player 1 wins: {player1_wins}/{iterations} ({player1_wins/iterations*100:.1f}%)\n")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            wx.CallAfter(self.sim_results.AppendText, f"Error during simulation: {str(e)}\n")
            wx.CallAfter(self.sim_results.AppendText, f"Details:\n{error_details}\n")
        
        finally:
            wx.CallAfter(self.sim_btn.Enable)
