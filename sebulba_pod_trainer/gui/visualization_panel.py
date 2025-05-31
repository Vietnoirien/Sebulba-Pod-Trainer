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
        self.metrics_queue = queue.Queue(maxsize=1000)  # Limit queue size
        
        # Dictionary to store multiple race visualizations
        self.race_visualizations = {}
        self.active_worker_id = None

        # Add throttling for visualization updates
        self.last_race_update = {}
        self.race_update_interval = 0.5  # Update race visualization every 500ms
        self.last_plot_update = 0
        self.plot_update_interval = 1.0  # Update plots every 1 second

        # Add missing initialization for updating_visualization flag
        self.updating_visualization = False
        
        # Create UI components
        self.create_ui()
        
        # Initialize plots
        self.init_plots()
        
        # Start update timer for real-time plotting
        self.update_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.update_timer)
        self.update_timer.Start(100)  # Update every 100ms
    
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

    def on_browse(self, event):
        """Handle browse button click to select model directory"""
        with wx.DirDialog(self, "Choose model directory", 
                        defaultPath=self.model_dir_ctrl.GetValue(),
                        style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dialog:
            
            if dialog.ShowModal() == wx.ID_OK:
                selected_path = dialog.GetPath()
                self.model_dir_ctrl.SetValue(selected_path)
                # Automatically load training data when directory is selected
                self.load_training_data(selected_path)

    def on_refresh(self, event):
        """Handle refresh button click to reload training data"""
        model_dir = self.model_dir_ctrl.GetValue()
        if model_dir:
            self.load_training_data(model_dir)
        else:
            wx.MessageBox("Please select a model directory first", "Refresh Error", wx.OK | wx.ICON_WARNING)
    
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
        
        # New standard files (runner/blocker)
        for role in ["runner", "blocker"]:
            standard_file = model_path / f"player0_{role}.pt"
            if standard_file.exists():
                model_files.append(f"Standard: player0_{role}.pt")
        
        # Legacy standard files (pod0/pod1)
        for pod_name in ["player0_pod0", "player0_pod1"]:
            standard_file = model_path / f"{pod_name}.pt"
            if standard_file.exists():
                model_files.append(f"Legacy: {pod_name}.pt")
        
        # New GPU-specific files (runner/blocker)
        for role in ["runner", "blocker"]:
            gpu_files = list(model_path.glob(f"player0_{role}_gpu*.pt"))
            for gpu_file in sorted(gpu_files):
                model_files.append(f"GPU: {gpu_file.name}")
        
        # Legacy GPU-specific files (pod0/pod1)
        gpu_files = list(model_path.glob("player0_pod*_gpu*.pt"))
        for gpu_file in sorted(gpu_files):
            model_files.append(f"Legacy GPU: {gpu_file.name}")
        
        # New iteration files (runner/blocker)
        for role in ["runner", "blocker"]:
            iter_files = list(model_path.glob(f"player0_{role}_gpu*_iter*.pt"))
            for iter_file in sorted(iter_files, key=lambda x: self.extract_iteration_number(x.name)):
                iteration = self.extract_iteration_number(iter_file.name)
                model_files.append(f"Iter {iteration}: {iter_file.name}")
        
        # Legacy iteration files (pod0/pod1)
        iter_files = list(model_path.glob("player0_pod*_gpu*_iter*.pt"))
        for iter_file in sorted(iter_files, key=lambda x: self.extract_iteration_number(x.name)):
            iteration = self.extract_iteration_number(iter_file.name)
            model_files.append(f"Legacy Iter {iteration}: {iter_file.name}")
        
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
        """Update plots with current data - optimized for recording/playback"""
        if not self.iterations:
            return
        
        try:
            # Only update if enough time has passed (throttling)
            current_time = time.time()
            if hasattr(self, 'last_plot_redraw') and current_time - self.last_plot_redraw < 0.2:
                return
            
            # Limit data points for performance (show last 1000 points)
            max_points = 1000
            if len(self.iterations) > max_points:
                start_idx = len(self.iterations) - max_points
                iterations = self.iterations[start_idx:]
                rewards = self.rewards[start_idx:]
                policy_losses = self.policy_losses[start_idx:]
                value_losses = self.value_losses[start_idx:]
            else:
                iterations = self.iterations
                rewards = self.rewards
                policy_losses = self.policy_losses
                value_losses = self.value_losses
            
            # Update rewards plot
            self.rewards_line.set_data(iterations, rewards)
            self.rewards_ax.relim()
            self.rewards_ax.autoscale_view()
            
            # Update losses plot
            self.policy_line.set_data(iterations, policy_losses)
            self.value_line.set_data(iterations, value_losses)
            self.losses_ax.relim()
            self.losses_ax.autoscale_view()
            
            # Use draw_idle for better performance
            self.rewards_canvas.draw_idle()
            self.losses_canvas.draw_idle()
            
            # Add the throttling timestamp
            self.last_plot_redraw = current_time
            
        except Exception as e:
            print(f"Error updating plots: {e}")

    def _add_metric_data(self, iteration, reward, policy_loss, value_loss):
        """Add metric data to the internal lists (thread-safe)"""
        try:
            # Ensure we're on the main thread
            if not wx.IsMainThread():
                wx.CallAfter(self._add_metric_data, iteration, reward, policy_loss, value_loss)
                return
            
            self.iterations.append(iteration)
            self.rewards.append(reward)
            self.policy_losses.append(policy_loss)
            self.value_losses.append(value_loss)
            
            # Update plots
            self.update_plots()
            
        except Exception as e:
            print(f"Error in _add_metric_data: {e}")
            import traceback
            traceback.print_exc()
    
    def on_timer(self, event):
        """Process any new metrics data from the queue - now handled by VisualizationConnector"""
        # The VisualizationConnector now handles all the heavy lifting
        # This timer just needs to handle any remaining direct updates
        
        if not self.live_check.GetValue() or self.updating_visualization:
            return
        
        # The recording/playback system in VisualizationConnector handles most updates
        # We just need to handle any direct queue items that bypass the connector
        current_time = time.time()
        
        # Throttle any remaining direct updates
        if current_time - self.last_plot_update < self.plot_update_interval:
            return
            
        self.updating_visualization = True
        
        try:
            # Process any remaining direct metrics (fallback)
            metrics_processed = 0
            max_metrics_per_update = 5  # Reduced since connector handles most
            
            while not self.metrics_queue.empty() and metrics_processed < max_metrics_per_update:
                try:
                    metric = self.metrics_queue.get_nowait()
                    
                    # Handle direct metrics (fallback path)
                    if 'iteration' in metric:
                        self.iterations.append(metric['iteration'])
                        self.rewards.append(metric.get('total_reward', metric.get('reward', 0)))
                        self.policy_losses.append(metric.get('policy_loss', 0))
                        self.value_losses.append(metric.get('value_loss', 0))
                        metrics_processed += 1
                        
                except queue.Empty:
                    break
            
            # Update plots only if we processed metrics
            if metrics_processed > 0:
                self.update_plots()
                self.last_plot_update = current_time
                
        except Exception as e:
            print(f"Error in on_timer: {e}")
        finally:
            self.updating_visualization = False

    def update_race_visualization_throttled(self, race_state, worker_id='main'):
        """Update race visualization - now optimized for recording/playback system"""
        try:
            # Store the race state for this worker
            self.race_visualizations[worker_id] = race_state
            
            # Add worker to dropdown if needed
            self.add_worker_to_dropdown(worker_id)
            
            # Only update visualization if this worker is currently being displayed
            should_update = False
            
            if self.active_worker_id == "All Environments":
                should_update = True
            elif self.active_worker_id == "Single Environment":
                selection = self.single_worker_choice.GetSelection()
                if selection != wx.NOT_FOUND:
                    selected_worker = self.single_worker_choice.GetString(selection)
                    should_update = (worker_id == selected_worker)
            
            if should_update:
                # Update layout if needed
                if self.active_worker_id == "All Environments" and worker_id not in self.race_axes:
                    self.update_race_plot_layout()
                else:
                    # Draw the specific environment
                    self.draw_single_race_environment(worker_id, race_state)
                    # Use draw_idle for better performance
                    self.race_canvas.draw_idle()
                    
        except Exception as e:
            print(f"Error in update_race_visualization_throttled: {e}")

    def _update_race_plot_async(self, worker_id, race_state):
        """Update race plot in a separate thread and then update UI"""
        try:
            # This runs in a background thread
            if self.active_worker_id == "All Environments":
                if worker_id not in self.race_axes:
                    # Schedule layout update on main thread
                    wx.CallAfter(self.update_race_plot_layout)
                else:
                    # Schedule drawing update on main thread
                    wx.CallAfter(self.draw_single_race_environment, worker_id, race_state)
                    wx.CallAfter(self.race_canvas.draw_idle)  # Use draw_idle for better performance
            elif self.active_worker_id == "Single Environment":
                selection = self.single_worker_choice.GetSelection()
                if selection != wx.NOT_FOUND:
                    selected_worker = self.single_worker_choice.GetString(selection)
                    if worker_id == selected_worker:
                        wx.CallAfter(self.update_single_worker_visualization, worker_id)
                        
        except Exception as e:
            print(f"Error in _update_race_plot_async: {e}")
            import traceback
            traceback.print_exc()
    
    def add_worker_to_dropdown(self, worker_id):
        """Add a worker to the dropdown if it's not already there - optimized"""
        try:
            # Add to single worker dropdown efficiently
            single_worker_choices = [self.single_worker_choice.GetString(i) 
                                   for i in range(self.single_worker_choice.GetCount())]
            if worker_id not in single_worker_choices:
                self.single_worker_choice.Append(worker_id)
                
                # Auto-select first worker in single mode if none selected
                if (self.active_worker_id == "Single Environment" and 
                    self.single_worker_choice.GetSelection() == wx.NOT_FOUND):
                    self.single_worker_choice.SetSelection(0)
        except Exception as e:
            print(f"Error adding worker to dropdown: {e}")

    def update_race_plot_layout(self):
        """Update race plot layout based on current workers - optimized"""
        try:
            if self.active_worker_id == "All Environments":
                worker_ids = list(self.race_visualizations.keys())
                if len(worker_ids) != len(self.race_axes) or not all(w in self.race_axes for w in worker_ids):
                    # Only rebuild layout if actually needed
                    self.init_multi_race_plots(worker_ids)
                    
                    # Redraw all current visualizations
                    for worker_id, race_state in self.race_visualizations.items():
                        self.draw_single_race_environment(worker_id, race_state)
                    
                    self.race_canvas.draw_idle()
        except Exception as e:
            print(f"Error updating race plot layout: {e}")
    
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
        """Draw a single race environment - optimized for recording/playback"""
        try:
            # Get the appropriate axis
            if worker_id in self.race_axes:
                ax = self.race_axes[worker_id]
            elif 'default' in self.race_axes:
                ax = self.race_axes['default']
            else:
                return
            
            # Only clear if we have new data
            if not hasattr(ax, '_last_update_time') or time.time() - ax._last_update_time > 0.1:
                ax.clear()
                ax._last_update_time = time.time()
            else:
                # Just clear the artists instead of the whole axis
                for artist in ax.patches + ax.lines + ax.texts:
                    artist.remove()
            
            # Set up axis properties
            ax.set_title(f'Environment: {worker_id}')
            ax.set_xlim(0, 16000)
            ax.set_ylim(0, 9000)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)  # Lighter grid for better performance
            
            worker_colors = self.get_worker_colors(worker_id)
            
            # Draw checkpoints efficiently
            if 'checkpoints' in race_state and race_state['checkpoints']:
                checkpoints = race_state['checkpoints']
                for i, (x, y) in enumerate(checkpoints):
                    circle = matplotlib.patches.Circle(
                        (x, y), 600, fill=False, edgecolor='green', 
                        linewidth=1.5, alpha=0.7
                    )
                    ax.add_patch(circle)
                    # Add checkpoint number with smaller font
                    ax.text(x, y, str(i), fontsize=8, ha='center', va='center', 
                           color='darkgreen', weight='bold')
            
            # Draw pods efficiently
            if 'pods' in race_state:
                pods = race_state['pods']
                
                for i, pod in enumerate(pods):
                    if 'position' not in pod or len(pod['position']) < 2:
                        continue
                        
                    x, y = pod['position']
                    angle = pod.get('angle', 0)
                    color = worker_colors[i % len(worker_colors)]
                    
                    # Draw collision radius circle with reduced alpha
                    collision_circle = matplotlib.patches.Circle(
                        (x, y), 400, 
                        fill=False, 
                        edgecolor=color, 
                        linestyle='--', 
                        alpha=0.3,
                        linewidth=1
                    )
                    ax.add_patch(collision_circle)
                    
                    # Draw pod as a triangle pointing in the direction of travel
                    dx = 400 * np.cos(np.radians(angle))
                    dy = 400 * np.sin(np.radians(angle))
                    
                    ax.arrow(
                        x, y, dx, dy, 
                        head_width=200, head_length=300, 
                        fc=color, ec=color, alpha=0.8
                    )
                    
                    # Add pod label with smaller font
                    ax.text(x, y-600, f'P{i}', fontsize=7, ha='center', va='center', 
                           color=color, weight='bold')
                    
                    # Add pod trail if available (limit trail length for performance)
                    if 'trail' in pod and len(pod['trail']) > 1:
                        trail = pod['trail'][-20:]  # Show last 20 points only
                        if len(trail) > 1:
                            trail_x, trail_y = zip(*trail)
                            ax.plot(trail_x, trail_y, color=color, alpha=0.2, linewidth=1)
            
            # Add legend only for single environment view (performance optimization)
            if len(self.race_axes) == 1 or self.active_worker_id == "Single Environment":
                legend_elements = [
                    matplotlib.patches.Patch(color='green', label='Checkpoints'),
                    matplotlib.patches.Patch(color=worker_colors[0], label='Pod 0'),
                    matplotlib.patches.Patch(color=worker_colors[1], label='Pod 1'),
                    matplotlib.patches.Patch(color=worker_colors[2], label='Pod 2'),
                    matplotlib.patches.Patch(color=worker_colors[3], label='Pod 3'),
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=6)
                
        except Exception as e:
            print(f"Error drawing race environment for {worker_id}: {e}")

    def update_single_worker_visualization(self, worker_id):
        """Update visualization for a single worker in single environment mode"""
        if worker_id not in self.race_visualizations:
            return
            
        race_state = self.race_visualizations[worker_id]
        self.draw_single_race_environment(worker_id, race_state)
        self.race_canvas.draw()

    def store_race_visualization(self, worker_id, race_state):
        """Store race state for a specific worker (thread-safe with queue)"""
        try:
            # Instead of using wx.CallAfter, put data in queue for processing by timer
            if not self.metrics_queue.full():
                self.metrics_queue.put({'race_state': race_state})
            else:
                # If queue is full, skip this update to prevent blocking
                print(f"Visualization queue full, skipping race state update for {worker_id}")
                
        except Exception as e:
            print(f"Error in store_race_visualization: {e}")
            import traceback
            traceback.print_exc()

    def add_metric(self, metric_dict):
        """Add a metric - now primarily handled by VisualizationConnector"""
        try:
            # For direct metrics that bypass the connector, use the queue
            if not self.metrics_queue.full():
                self.metrics_queue.put_nowait(metric_dict)
        except queue.Full:
            # Queue is full, skip this update
            pass
        except Exception as e:
            print(f"Error in add_metric: {e}")

    def load_training_data(self, model_dir):
        """Load training data from model directory - simplified"""
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
            
            # Check for any .pt files
            model_files = list(model_path.glob("*.pt"))
            if not model_files:
                wx.MessageBox(f"No .pt model files found in directory: {model_dir}", "Model Files Error", wx.OK | wx.ICON_ERROR)
                return
            
            print(f"Found {len(model_files)} .pt files in {model_dir}")
            
            # Look for training log files
            log_files = []
            
            # Check for various log file patterns
            log_patterns = [
                "training_log.txt",
                "training_log_*.txt",
                "*.log"
            ]
            
            for pattern in log_patterns:
                log_files.extend(list(model_path.glob(pattern)))
            
            if not log_files:
                wx.MessageBox(f"No training log files found in: {model_path}", "Log File Error", wx.OK | wx.ICON_WARNING)
                return
            
            print(f"Found {len(log_files)} log files")
            
            # Parse all log files
            for log_path in log_files:
                print(f"Parsing log file: {log_path}")
                try:
                    with open(log_path, 'r') as f:
                        for line in f:
                            try:
                                # Handle various log formats
                                if 'iteration=' in line and ('reward=' in line or 'total_reward=' in line):
                                    parts = line.strip().split(',')
                                    data = {}
                                    
                                    # Parse key=value pairs
                                    for part in parts:
                                        if '=' in part:
                                            key, value = part.split('=', 1)
                                            try:
                                                data[key] = float(value)
                                            except ValueError:
                                                data[key] = value
                                    
                                    # Extract required fields
                                    iteration = int(data.get('iteration', -1))
                                    if iteration < 0:
                                        continue
                                    
                                    # Try different reward field names
                                    reward = data.get('total_reward', data.get('reward', 0))
                                    policy_loss = data.get('policy_loss', 0)
                                    value_loss = data.get('value_loss', 0)
                                    
                                    self.iterations.append(iteration)
                                    self.rewards.append(float(reward))
                                    self.policy_losses.append(float(policy_loss))
                                    self.value_losses.append(float(value_loss))
                                    
                            except Exception as parse_error:
                                # Skip malformed lines
                                continue
                                
                except Exception as file_error:
                    print(f"Error reading log file {log_path}: {file_error}")
                    continue
            
            # Update plots
            if not self.iterations:
                wx.MessageBox("No valid training data found in log files", "Data Error", wx.OK | wx.ICON_WARNING)
                return
                
            print(f"Loaded {len(self.iterations)} training data points")
            self.update_plots()
            
            # Populate manual model list if in manual mode
            if self.model_selection_choice.GetSelection() == 2:  # Manual Select
                self.populate_manual_model_list()
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            wx.MessageBox(f"Error loading training data: {str(e)}\n\nDetails:\n{error_details}", 
                        "Data Loading Error", wx.OK | wx.ICON_ERROR)
    
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
    
    def find_best_model_file(self, pod_key, model_path):
        """Find the best model file based on selection strategy - simplified to work with any .pt files"""
        selection = self.model_selection_choice.GetSelection()
        strategy = self.model_selection_choice.GetString(selection)
        
        if strategy == "Manual Select":
            return self.get_manual_selected_model(pod_key, model_path)
        elif strategy == "Latest Iteration":
            return self.find_latest_model_file(pod_key, model_path)
        else:  # Best Performance
            return self.find_best_performance_model_file(pod_key, model_path)
    
    def pod_key_to_role(self, pod_key):
        """Convert pod_key to role name"""
        if "pod0" in pod_key:
            return "runner"
        elif "pod1" in pod_key:
            return "blocker"
        return pod_key  # Return as-is if already in new format

    def find_latest_model_file(self, pod_key, model_path):
        """Find the latest model file by iteration number - simplified"""
        role = self.pod_key_to_role(pod_key)
        
        # Find all .pt files that contain the role name
        all_pt_files = list(model_path.glob("*.pt"))
        role_files = [f for f in all_pt_files if role in f.name.lower()]
        
        if not role_files:
            # If no role-specific files found, try any .pt file
            role_files = all_pt_files
        
        if not role_files:
            return None
        
        # Try to find files with iteration numbers
        iter_files = []
        for file in role_files:
            iter_num = self.extract_iteration_number(file.name)
            if iter_num > 0:
                iter_files.append((iter_num, file))
        
        if iter_files:
            # Sort by iteration number and return the latest
            iter_files.sort(key=lambda x: x[0], reverse=True)
            return iter_files[0][1]
        
        # If no iteration files, return the first role file
        return role_files[0]
        
    def find_best_performance_model_file(self, pod_key, model_path):
        """Find the best model file based on training performance - simplified"""
        role = self.pod_key_to_role(pod_key)
        
        # Find all .pt files that contain the role name
        all_pt_files = list(model_path.glob("*.pt"))
        role_files = [f for f in all_pt_files if role in f.name.lower()]
        
        if not role_files:
            # If no role-specific files found, try any .pt file
            role_files = all_pt_files
        
        if not role_files:
            return None
        
        # Try to find the best model based on training logs
        best_model = self.find_best_model_from_logs(role_files, model_path)
        if best_model:
            return best_model
        
        # Fallback: return the latest iteration file
        return self.find_latest_model_file(pod_key, model_path)

    def extract_iteration_number(self, filename):
        """Extract iteration number from filename - improved to handle various formats"""
        try:
            # Look for patterns like _iter123, iter123, _123.pt, etc.
            import re
            
            # Try different patterns
            patterns = [
                r'_iter(\d+)',      # _iter123
                r'iter(\d+)',       # iter123
                r'_(\d+)\.pt$',     # _123.pt
                r'(\d+)\.pt$'       # 123.pt
            ]
            
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    return int(match.group(1))
                    
        except (ValueError, IndexError):
            pass
        return 0

    def populate_manual_model_list(self):
        """Populate the manual model selection dropdown - simplified"""
        model_dir = self.model_dir_ctrl.GetValue()
        if not model_dir:
            return
            
        model_path = Path(model_dir)
        if not model_path.exists():
            return
        
        # Clear existing choices
        self.manual_model_choice.Clear()
        
        # Find all .pt files
        model_files = list(model_path.glob("*.pt"))
        
        if not model_files:
            return
        
        # Sort files by name and add to dropdown
        model_files.sort(key=lambda x: x.name)
        
        for model_file in model_files:
            # Extract iteration number for display
            iter_num = self.extract_iteration_number(model_file.name)
            if iter_num > 0:
                display_name = f"Iter {iter_num}: {model_file.name}"
            else:
                display_name = f"Model: {model_file.name}"
            
            self.manual_model_choice.Append(display_name)
        
        if model_files:
            self.manual_model_choice.SetSelection(0)

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
            # If no logs found, default to the latest iteration
            wx.CallAfter(self.sim_results.AppendText, "No training logs found. Defaulting to latest iteration.\n")
            return self.find_latest_iteration_model(model_files)
        
        # Parse logs to find best performing iterations
        best_iteration = None
        best_reward = float('-inf')
        iteration_rewards = {}
        
        for log_file in log_files:
            try:
                wx.CallAfter(self.sim_results.AppendText, f"Parsing log file: {log_file.name}\n")
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            parts = line.strip().split(',')
                            if len(parts) >= 2:
                                # Extract iteration number
                                iter_part = [p for p in parts if p.startswith('iteration=')]
                                if not iter_part:
                                    continue
                                iteration = int(iter_part[0].split('=')[1])
                                
                                # Extract reward - handle both old and new formats
                                reward_part = [p for p in parts if p.startswith('reward=') or p.startswith('total_reward=')]
                                if not reward_part:
                                    continue
                                reward = float(reward_part[0].split('=')[1])
                                
                                # Track the best reward and its iteration
                                if reward > best_reward:
                                    best_reward = reward
                                    best_iteration = iteration
                                    wx.CallAfter(self.sim_results.AppendText, 
                                            f"New best: iteration {iteration}, reward {reward:.4f}\n")
                                
                                # Store all iteration rewards for averaging
                                if iteration not in iteration_rewards:
                                    iteration_rewards[iteration] = []
                                iteration_rewards[iteration].append(reward)
                                
                        except (ValueError, IndexError) as e:
                            continue
            except Exception as e:
                wx.CallAfter(self.sim_results.AppendText, f"Error parsing log {log_file.name}: {str(e)}\n")
                continue
        
        # If we found a best iteration, look for corresponding model file
        if best_iteration is not None:
            wx.CallAfter(self.sim_results.AppendText, f"Best iteration found: {best_iteration} with reward {best_reward:.4f}\n")
            
            # First try exact match
            for model_file in model_files:
                if f'_iter{best_iteration}.' in str(model_file):
                    wx.CallAfter(self.sim_results.AppendText, 
                            f"Selected best model: {model_file.name} (iteration {best_iteration}, reward {best_reward:.4f})\n")
                    return model_file
            
            # If no exact match, find the closest iteration
            closest_iter_file = self.find_closest_iteration_model(model_files, best_iteration)
            if closest_iter_file:
                wx.CallAfter(self.sim_results.AppendText, 
                        f"Selected closest model to best iteration: {closest_iter_file.name}\n")
                return closest_iter_file
        
        # Fallback to latest iteration
        wx.CallAfter(self.sim_results.AppendText, "No best model found from logs. Using latest iteration.\n")
        return self.find_latest_iteration_model(model_files)

    def find_closest_iteration_model(self, model_files, target_iteration):
        """Find the model file with iteration closest to target_iteration"""
        iter_files = []
        for model_file in model_files:
            try:
                if '_iter' in str(model_file):
                    iter_num = self.extract_iteration_number(model_file.name)
                    iter_files.append((iter_num, model_file))
            except (ValueError, IndexError):
                continue
        
        if not iter_files:
            return None
        
        # Sort by distance to target iteration
        iter_files.sort(key=lambda x: abs(x[0] - target_iteration))
        return iter_files[0][1]

    def find_latest_iteration_model(self, model_files):
        """Find the model file with the highest iteration number"""
        iter_files = []
        for model_file in model_files:
            try:
                if '_iter' in str(model_file):
                    iter_num = self.extract_iteration_number(model_file.name)
                    iter_files.append((iter_num, model_file))
            except (ValueError, IndexError):
                continue
        
        if not iter_files:
            # If no iteration files, return the first model file
            return model_files[0] if model_files else None
        
        # Sort by iteration number (descending)
        iter_files.sort(key=lambda x: x[0], reverse=True)
        wx.CallAfter(self.sim_results.AppendText, 
                f"Selected latest model: {iter_files[0][1].name} (iteration {iter_files[0][0]})\n")
        return iter_files[0][1]

    def on_visualize_race(self, event):
        """Start race visualization - simplified model checking"""
        model_dir = self.model_dir_ctrl.GetValue()
        if not model_dir:
            wx.MessageBox("Please select a model directory first", "Visualization Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Check if model directory exists
        model_path = Path(model_dir)
        if not model_path.exists():
            wx.MessageBox(f"Model directory does not exist: {model_dir}", "Visualization Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Check for any .pt files
        model_files = list(model_path.glob("*.pt"))
        if not model_files:
            wx.MessageBox(f"No .pt model files found in directory: {model_dir}", "Visualization Error", wx.OK | wx.ICON_ERROR)
            return
        
        print(f"Found {len(model_files)} .pt files for visualization")
        
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
            
            # Debug: Print initial observation keys
            wx.CallAfter(self.sim_results.AppendText, f"Initial observation keys: {list(observations.keys())}\n")
            
            # Get observation dimension from the environment
            obs_dim = next(iter(observations.values())).shape[1]

            # Load player 0 models - support both new and old naming conventions
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
                
                # Get head sizes from checkpoint - check all possible key names
                policy_hidden_size = 16  # Default
                value_hidden_size = 16   # Default
                action_hidden_size = 16  # Default
                
                # Check for policy head
                if 'policy_head.0.weight' in checkpoint:
                    policy_hidden_size = checkpoint['policy_head.0.weight'].shape[0]
                
                # Check for value head
                if 'value_head.0.weight' in checkpoint:
                    value_hidden_size = checkpoint['value_head.0.weight'].shape[0]
                
                # Check for action head (try multiple possible key names for backward compatibility)
                action_head_keys = [
                    'action_head.0.weight',           # Current name
                    'special_action_head.0.weight',   # Legacy name
                    'special_head.0.weight'           # Another possible legacy name
                ]
                
                for key in action_head_keys:
                    if key in checkpoint:
                        action_hidden_size = checkpoint[key].shape[0]
                        break
                
                # Create network with matching architecture
                network = PodNetwork(
                    observation_dim=obs_dim,
                    hidden_layers=hidden_layers,
                    policy_hidden_size=policy_hidden_size,
                    value_hidden_size=value_hidden_size,
                    action_hidden_size=action_hidden_size  # Use action_hidden_size instead of special_hidden_size
                )
                
                return network, checkpoint
            
            # Load player 0 models - try both new and old naming conventions
            for i in range(2):
                # Determine role and pod_key
                role = "runner" if i == 0 else "blocker"
                pod_key = f"player0_pod{i}"
                
                # Try to find model file using the new find_best_model_file method
                model_file = self.find_best_model_file(pod_key, model_path)
                
                if model_file:
                    try:
                        network, checkpoint = create_network_from_checkpoint(model_file)
                        network.load_state_dict(checkpoint)
                        network.eval()
                        pod_networks[pod_key] = network
                        wx.CallAfter(self.sim_results.AppendText, f"Loaded {role} model from {model_file.name}\n")
                    except Exception as e:
                        wx.CallAfter(self.sim_results.AppendText, f"Error loading {role} from {model_file}: {str(e)}\n")
                        # Create a default network as fallback
                        pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
                else:
                    wx.CallAfter(self.sim_results.AppendText, f"No model found for {role} (pod{i})\n")
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
                    role = "runner" if i == 0 else "blocker"
                    model_file = self.find_best_model_file(source_pod_key, model_path)
                    if model_file:
                        try:
                            network, checkpoint = create_network_from_checkpoint(model_file)
                            network.load_state_dict(checkpoint)
                            network.eval()
                            pod_networks[pod_key] = network
                        except Exception as e:
                            wx.CallAfter(self.sim_results.AppendText, f"Error loading {role} for player1: {str(e)}\n")
                            # Use default network as fallback
                            pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
                    else:
                        # Use default network
                        pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
                else:
                    # Use default network
                    pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
            
            # Debug: Print loaded pod networks
            wx.CallAfter(self.sim_results.AppendText, f"Loaded pod networks for keys: {list(pod_networks.keys())}\n")
            
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
            max_steps = 600  # Limit steps to avoid infinite loops
            
            for step in range(max_steps):
                if done:
                    wx.CallAfter(self.sim_results.AppendText, f"Race ended at step {step} - done={done}\n")
                    break
                
                # Get actions from all pod networks
                actions = {}
                with torch.no_grad():
                    # Debug: Print observation keys at each step
                    if step > 0:  # Only print for steps near the error
                        wx.CallAfter(self.sim_results.AppendText, f"Step {step} observation keys: {list(observations.keys())}\n")
                    
                    for pod_key, network in pod_networks.items():
                        # Get deterministic action from network
                        if pod_key in observations:
                            obs = observations[pod_key]
                            action = network.get_actions(obs, deterministic=True)
                            actions[pod_key] = action
                        elif step > 0:  # Debug near error
                            wx.CallAfter(self.sim_results.AppendText, f"Step {step}: Missing observation for {pod_key}\n")
                
                # Debug: Print action keys
                if step > 0:
                    wx.CallAfter(self.sim_results.AppendText, f"Step {step} action keys: {list(actions.keys())}\n")
                
                # Step the environment
                try:
                    observations, rewards, done, info = env.step(actions)
                    
                    # Debug: Print info after step
                    if step > 0:
                        wx.CallAfter(self.sim_results.AppendText, f"Step {step} info keys: {list(info.keys())}\n")
                        if "checkpoint_progress" in info:
                            wx.CallAfter(self.sim_results.AppendText, f"Step {step} checkpoint_progress shape: {info['checkpoint_progress'].shape}\n")
                    
                except KeyError as e:
                    wx.CallAfter(self.sim_results.AppendText, f"Error during step {step}: KeyError {str(e)}\n")
                    wx.CallAfter(self.sim_results.AppendText, f"Available action keys: {list(actions.keys())}\n")
                    wx.CallAfter(self.sim_results.AppendText, f"Available observation keys: {list(observations.keys())}\n")
                    wx.CallAfter(self.sim_results.AppendText, "Continuing with visualization...\n")
                    
                    # FIX: Create a fallback visualization using current pod states
                    pod_states = env.get_pod_states()
                    race_state['pods'] = []
                    for pod_idx, pod_state in enumerate(pod_states):
                        pod_key = f"player{pod_idx//2}_pod{pod_idx%2}"
                        if pod_key in pod_networks:
                            position = pod_state['position']
                            pod_trails[pod_key].append(position)
                            pod_info = {
                                'position': position,
                                'angle': pod_state['angle'],
                                'trail': pod_trails[pod_key]
                            }
                            race_state['pods'].append(pod_info)
                    wx.CallAfter(self.update_race_visualization_throttled, race_state, 'visualization')
                    continue
                except ValueError as e:
                    wx.CallAfter(self.sim_results.AppendText, f"Error during step {step}: ValueError {str(e)}\n")
                    wx.CallAfter(self.sim_results.AppendText, "Continuing with visualization...\n")
                    # FIX: Create a fallback visualization using current pod states
                    pod_states = env.get_pod_states()
                    race_state['pods'] = []
                    for pod_idx, pod_state in enumerate(pod_states):
                        pod_key = f"player{pod_idx//2}_pod{pod_idx%2}"
                        if pod_key in pod_networks:
                            position = pod_state['position']
                            pod_trails[pod_key].append(position)
                            pod_info = {
                                'position': position,
                                'angle': pod_state['angle'],
                                'trail': pod_trails[pod_key]
                            }
                            race_state['pods'].append(pod_info)
                    wx.CallAfter(self.update_race_visualization_throttled, race_state, 'visualization')
                    continue
                
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
                    wx.CallAfter(self.update_race_visualization_throttled, race_state, 'visualization')
                    print(f"Updated visualization at step {step}")
                    
                    # Small delay to make visualization visible
                    time.sleep(0.1) 
            
            # Final update
            wx.CallAfter(self.update_race_visualization_throttled, race_state, 'visualization')
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            wx.CallAfter(wx.MessageBox, f"Error during race visualization: {str(e)}\n\nDetails:\n{error_details}", 
                        "Visualization Error", wx.OK | wx.ICON_ERROR)
        
        finally:
            wx.CallAfter(self.visualize_race_btn.Enable)

    def on_run_simulation(self, event):
        """Run a simulation with the selected model - simplified model checking"""
        model_dir = self.model_dir_ctrl.GetValue()
        if not model_dir:
            wx.MessageBox("Please select a model directory first", "Simulation Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Check if model directory exists
        model_path = Path(model_dir)
        if not model_path.exists():
            wx.MessageBox(f"Model directory does not exist: {model_dir}", "Simulation Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Check for any .pt files
        model_files = list(model_path.glob("*.pt"))
        if not model_files:
            wx.MessageBox(f"No .pt model files found in directory: {model_dir}", "Simulation Error", wx.OK | wx.ICON_ERROR)
            return
        
        print(f"Found {len(model_files)} .pt files for simulation")
        
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
            
            # Load models for player 0 - support both new and old naming conventions
            for i in range(2):
                # Determine role and pod_key
                role = "runner" if i == 0 else "blocker"
                pod_key = f"player0_pod{i}"
                
                # Try to find model file using the new find_best_model_file method
                model_file = self.find_best_model_file(pod_key, model_path)
                
                if model_file:
                    try:
                        network, checkpoint = create_network_from_checkpoint(model_file)
                        network.load_state_dict(checkpoint)
                        network.eval()
                        pod_networks[pod_key] = network
                        wx.CallAfter(self.sim_results.AppendText, f"Loaded {role} model from {model_file.name}\n")
                    except Exception as e:
                        wx.CallAfter(self.sim_results.AppendText, f"Error loading {role} from {model_file}: {str(e)}\n")
                        # Create a default network as fallback
                        pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
                else:
                    wx.CallAfter(self.sim_results.AppendText, f"No model found for {role} (pod{i}), using random initialization\n")
                    # Create a default network
                    pod_networks[pod_key] = PodNetwork(observation_dim=obs_dim).to('cpu')
            
            # Use the same model for opponent (player1)
            for i in range(2):
                pod_key = f"player1_pod{i}"
                source_pod_key = f"player0_pod{i}"
                
                # Use player0's model for player1
                role = "runner" if i == 0 else "blocker"
                model_file = self.find_best_model_file(source_pod_key, model_path)
                
                if model_file:
                    try:
                        network, checkpoint = create_network_from_checkpoint(model_file)
                        network.load_state_dict(checkpoint)
                        network.eval()
                        pod_networks[pod_key] = network
                    except Exception as e:
                        wx.CallAfter(self.sim_results.AppendText, f"Error loading {role} for player1: {str(e)}\n")
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
            
            # Track role-specific performance
            runner_wins = 0
            blocker_wins = 0
            total_runner_progress = 0
            total_blocker_progress = 0
            
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
                
                # Determine winner and track role-specific performance
                player0_runner_progress = info["checkpoint_progress"][0][0].item()  # player0_pod0 (runner)
                player0_blocker_progress = info["checkpoint_progress"][1][0].item()  # player0_pod1 (blocker)
                player1_runner_progress = info["checkpoint_progress"][2][0].item()   # player1_pod0 (runner)
                player1_blocker_progress = info["checkpoint_progress"][3][0].item()  # player1_pod1 (blocker)
                
                # Track overall progress for roles
                total_runner_progress += max(player0_runner_progress, player1_runner_progress)
                total_blocker_progress += max(player0_blocker_progress, player1_blocker_progress)
                
                # Determine which player's best pod performed better
                player0_best_progress = max(player0_runner_progress, player0_blocker_progress)
                player1_best_progress = max(player1_runner_progress, player1_blocker_progress)
                
                if player0_best_progress > player1_best_progress:
                    player0_wins += 1
                    # Determine which role won for player0
                    if player0_runner_progress > player0_blocker_progress:
                        runner_wins += 1
                        winning_role = "runner"
                    else:
                        blocker_wins += 1
                        winning_role = "blocker"
                    wx.CallAfter(self.sim_results.AppendText, 
                               f"  Result: Player 0 wins with {winning_role}! "
                               f"Progress: {player0_best_progress:.3f} vs {player1_best_progress:.3f}\n")
                else:
                    player1_wins += 1
                    # Determine which role won for player1
                    if player1_runner_progress > player1_blocker_progress:
                        runner_wins += 1
                        winning_role = "runner"
                    else:
                        blocker_wins += 1
                        winning_role = "blocker"
                    wx.CallAfter(self.sim_results.AppendText, 
                               f"  Result: Player 1 wins with {winning_role}! "
                               f"Progress: {player1_best_progress:.3f} vs {player0_best_progress:.3f}\n")
                
                # Show detailed progress for this race
                wx.CallAfter(self.sim_results.AppendText, 
                           f"    P0 Runner: {player0_runner_progress:.3f}, P0 Blocker: {player0_blocker_progress:.3f}\n")
                wx.CallAfter(self.sim_results.AppendText, 
                           f"    P1 Runner: {player1_runner_progress:.3f}, P1 Blocker: {player1_blocker_progress:.3f}\n")
            
            # Calculate averages
            avg_runner_progress = total_runner_progress / iterations
            avg_blocker_progress = total_blocker_progress / iterations
            
            # Show final results
            wx.CallAfter(self.sim_results.AppendText, f"\n" + "="*50 + "\n")
            wx.CallAfter(self.sim_results.AppendText, f"FINAL SIMULATION RESULTS\n")
            wx.CallAfter(self.sim_results.AppendText, f"="*50 + "\n")
            
            wx.CallAfter(self.sim_results.AppendText, f"Player Performance:\n")
            wx.CallAfter(self.sim_results.AppendText, f"  Player 0 wins: {player0_wins}/{iterations} ({player0_wins/iterations*100:.1f}%)\n")
            wx.CallAfter(self.sim_results.AppendText, f"  Player 1 wins: {player1_wins}/{iterations} ({player1_wins/iterations*100:.1f}%)\n")
            
            wx.CallAfter(self.sim_results.AppendText, f"\nRole Performance:\n")
            wx.CallAfter(self.sim_results.AppendText, f"  Runner wins: {runner_wins}/{iterations} ({runner_wins/iterations*100:.1f}%)\n")
            wx.CallAfter(self.sim_results.AppendText, f"  Blocker wins: {blocker_wins}/{iterations} ({blocker_wins/iterations*100:.1f}%)\n")
            
            wx.CallAfter(self.sim_results.AppendText, f"\nAverage Progress:\n")
            wx.CallAfter(self.sim_results.AppendText, f"  Average Runner Progress: {avg_runner_progress:.3f}\n")
            wx.CallAfter(self.sim_results.AppendText, f"  Average Blocker Progress: {avg_blocker_progress:.3f}\n")
            
            # Determine which role is more effective
            if avg_runner_progress > avg_blocker_progress:
                wx.CallAfter(self.sim_results.AppendText, f"\nRunner strategy appears more effective (+{avg_runner_progress - avg_blocker_progress:.3f})\n")
            elif avg_blocker_progress > avg_runner_progress:
                wx.CallAfter(self.sim_results.AppendText, f"\nBlocker strategy appears more effective (+{avg_blocker_progress - avg_runner_progress:.3f})\n")
            else:
                wx.CallAfter(self.sim_results.AppendText, f"\nBoth strategies perform equally well\n")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            wx.CallAfter(self.sim_results.AppendText, f"Error during simulation: {str(e)}\n")
            wx.CallAfter(self.sim_results.AppendText, f"Details:\n{error_details}\n")
        
        finally:
            wx.CallAfter(self.sim_btn.Enable)
