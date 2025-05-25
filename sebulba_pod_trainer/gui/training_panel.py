import wx
import torch
import os
from pathlib import Path

class TrainingPanel(wx.Panel):
    def __init__(self, parent, main_frame):
        super(TrainingPanel, self).__init__(parent)
        
        self.main_frame = main_frame
        
        # Create UI elements
        self.create_ui()
        
        # Update UI from config
        self.update_from_config()
    
    def create_ui(self):
        # Main sizer for the panel
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create a scrolled window to contain all the content
        self.scroll_panel = wx.ScrolledWindow(self, style=wx.VSCROLL | wx.HSCROLL)
        self.scroll_panel.SetScrollRate(20, 20)  # Set scroll increments
        
        # Create the content sizer inside the scrolled window
        content_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Training parameters section
        training_box = wx.StaticBox(self.scroll_panel, label="Training Parameters")
        training_sizer = wx.StaticBoxSizer(training_box, wx.VERTICAL)
        
        # Grid for parameters
        param_grid = wx.FlexGridSizer(rows=10, cols=2, vgap=10, hgap=10)
        param_grid.AddGrowableCol(1, 1)
        
        # Batch size
        param_grid.Add(wx.StaticText(self.scroll_panel, label="Batch Size:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.batch_size_ctrl = wx.SpinCtrl(self.scroll_panel, min=1, max=1024, initial=64)
        param_grid.Add(self.batch_size_ctrl, 0, wx.EXPAND)
        
        # Learning rate
        param_grid.Add(wx.StaticText(self.scroll_panel, label="Learning Rate:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.learning_rate_ctrl = wx.TextCtrl(self.scroll_panel, value="0.0003")
        param_grid.Add(self.learning_rate_ctrl, 0, wx.EXPAND)
        
        # Number of iterations
        param_grid.Add(wx.StaticText(self.scroll_panel, label="Number of Iterations:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.iterations_ctrl = wx.SpinCtrl(self.scroll_panel, min=1, max=100000, initial=1000)
        param_grid.Add(self.iterations_ctrl, 0, wx.EXPAND)
        
        # Steps per iteration
        param_grid.Add(wx.StaticText(self.scroll_panel, label="Steps per Iteration:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.steps_ctrl = wx.SpinCtrl(self.scroll_panel, min=1, max=10000, initial=128)
        param_grid.Add(self.steps_ctrl, 0, wx.EXPAND)
        
        # Save interval
        param_grid.Add(wx.StaticText(self.scroll_panel, label="Save Interval:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.save_interval_ctrl = wx.SpinCtrl(self.scroll_panel, min=1, max=1000, initial=50)
        param_grid.Add(self.save_interval_ctrl, 0, wx.EXPAND)
        
        # Save directory
        param_grid.Add(wx.StaticText(self.scroll_panel, label="Save Directory:"), 0, wx.ALIGN_CENTER_VERTICAL)
        save_dir_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.save_dir_ctrl = wx.TextCtrl(self.scroll_panel, value="models")
        save_dir_sizer.Add(self.save_dir_ctrl, 1, wx.EXPAND)
        self.browse_btn = wx.Button(self.scroll_panel, label="Browse...")
        save_dir_sizer.Add(self.browse_btn, 0, wx.LEFT, 5)
        param_grid.Add(save_dir_sizer, 0, wx.EXPAND)
        
        # Optimized environment option
        param_grid.Add(wx.StaticText(self.scroll_panel, label="Use Optimized Environment:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.optimized_env_check = wx.CheckBox(self.scroll_panel, label="")
        param_grid.Add(self.optimized_env_check, 0)
        
        # Add parameters to sizer
        training_sizer.Add(param_grid, 0, wx.EXPAND | wx.ALL, 10)
        
        # PPO parameters section
        ppo_box = wx.StaticBox(self.scroll_panel, label="PPO Parameters")
        ppo_sizer = wx.StaticBoxSizer(ppo_box, wx.VERTICAL)
        
        # Grid for PPO parameters
        ppo_grid = wx.FlexGridSizer(rows=8, cols=2, vgap=10, hgap=10)
        ppo_grid.AddGrowableCol(1, 1)
        
        # Mini batch size
        ppo_grid.Add(wx.StaticText(self.scroll_panel, label="Mini Batch Size:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.mini_batch_size_ctrl = wx.SpinCtrl(self.scroll_panel, min=1, max=512, initial=16)
        ppo_grid.Add(self.mini_batch_size_ctrl, 0, wx.EXPAND)
        
        # PPO epochs
        ppo_grid.Add(wx.StaticText(self.scroll_panel, label="PPO Epochs:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.ppo_epochs_ctrl = wx.SpinCtrl(self.scroll_panel, min=1, max=100, initial=10)
        ppo_grid.Add(self.ppo_epochs_ctrl, 0, wx.EXPAND)
        
        # Clip parameter
        ppo_grid.Add(wx.StaticText(self.scroll_panel, label="Clip Parameter:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.clip_param_ctrl = wx.TextCtrl(self.scroll_panel, value="0.2")
        ppo_grid.Add(self.clip_param_ctrl, 0, wx.EXPAND)
        
        # Value coefficient
        ppo_grid.Add(wx.StaticText(self.scroll_panel, label="Value Coefficient:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.value_coef_ctrl = wx.TextCtrl(self.scroll_panel, value="0.5")
        ppo_grid.Add(self.value_coef_ctrl, 0, wx.EXPAND)
        
        # Entropy coefficient
        ppo_grid.Add(wx.StaticText(self.scroll_panel, label="Entropy Coefficient:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.entropy_coef_ctrl = wx.TextCtrl(self.scroll_panel, value="0.01")
        ppo_grid.Add(self.entropy_coef_ctrl, 0, wx.EXPAND)
        
        # Max gradient norm
        ppo_grid.Add(wx.StaticText(self.scroll_panel, label="Max Gradient Norm:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.max_grad_norm_ctrl = wx.TextCtrl(self.scroll_panel, value="0.5")
        ppo_grid.Add(self.max_grad_norm_ctrl, 0, wx.EXPAND)
        
        # Gamma
        ppo_grid.Add(wx.StaticText(self.scroll_panel, label="Gamma (Discount Factor):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.gamma_ctrl = wx.TextCtrl(self.scroll_panel, value="0.99")
        ppo_grid.Add(self.gamma_ctrl, 0, wx.EXPAND)
        
        # GAE Lambda
        ppo_grid.Add(wx.StaticText(self.scroll_panel, label="GAE Lambda:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.gae_lambda_ctrl = wx.TextCtrl(self.scroll_panel, value="0.95")
        ppo_grid.Add(self.gae_lambda_ctrl, 0, wx.EXPAND)
        
        # Add PPO parameters to sizer
        ppo_sizer.Add(ppo_grid, 0, wx.EXPAND | wx.ALL, 10)
        
        # Device configuration section
        device_box = wx.StaticBox(self.scroll_panel, label="Device Configuration")
        device_sizer = wx.StaticBoxSizer(device_box, wx.VERTICAL)
        
        # Device selection
        device_grid = wx.FlexGridSizer(rows=0, cols=2, vgap=10, hgap=10)
        device_grid.AddGrowableCol(1, 1)
        
        device_grid.Add(wx.StaticText(self.scroll_panel, label="Training Device:"), 0, wx.ALIGN_CENTER_VERTICAL)
        
        # Create device choices
        device_choices = ["CPU"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_choices.append(f"CUDA:{i} ({torch.cuda.get_device_name(i)})")
        
        self.device_choice = wx.Choice(self.scroll_panel, choices=device_choices)
        self.device_choice.SetSelection(0)  # Default to CPU
        device_grid.Add(self.device_choice, 0, wx.EXPAND)
        
        # Multi-GPU option
        if torch.cuda.device_count() > 1:
            device_grid.Add(wx.StaticText(self.scroll_panel, label="Use Multiple GPUs:"), 0, wx.ALIGN_CENTER_VERTICAL)
            self.multi_gpu_check = wx.CheckBox(self.scroll_panel, label="")
            device_grid.Add(self.multi_gpu_check, 0)
            
            # GPU selection
            device_grid.Add(wx.StaticText(self.scroll_panel, label="Select GPUs:"), 0, wx.ALIGN_CENTER_VERTICAL)
            
            # Create a panel for GPU checkboxes
            gpu_panel = wx.Panel(self.scroll_panel)
            gpu_sizer = wx.BoxSizer(wx.VERTICAL)
            
            self.gpu_checks = []
            for i in range(torch.cuda.device_count()):
                gpu_check = wx.CheckBox(gpu_panel, label=f"GPU {i}: {torch.cuda.get_device_name(i)}")
                gpu_sizer.Add(gpu_check, 0, wx.BOTTOM, 5)
                self.gpu_checks.append(gpu_check)
            
            gpu_panel.SetSizer(gpu_sizer)
            device_grid.Add(gpu_panel, 0, wx.EXPAND)
            
            # Bind multi-GPU checkbox event
            self.multi_gpu_check.Bind(wx.EVT_CHECKBOX, self.on_multi_gpu_toggle)
        
        # Mixed precision option
        device_grid.Add(wx.StaticText(self.scroll_panel, label="Use Mixed Precision:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.mixed_precision_check = wx.CheckBox(self.scroll_panel, label="")
        device_grid.Add(self.mixed_precision_check, 0)
        
        # Parallel training option
        device_grid.Add(wx.StaticText(self.scroll_panel, label="Use Parallel Training:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.parallel_check = wx.CheckBox(self.scroll_panel, label="")
        device_grid.Add(self.parallel_check, 0)
        
        # Environments per GPU
        device_grid.Add(wx.StaticText(self.scroll_panel, label="Environments per GPU:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.envs_per_gpu_ctrl = wx.SpinCtrl(self.scroll_panel, min=1, max=16, initial=8)
        device_grid.Add(self.envs_per_gpu_ctrl, 0, wx.EXPAND)
        
        # Add device configuration to sizer
        device_sizer.Add(device_grid, 0, wx.EXPAND | wx.ALL, 10)
        
        # Parallel Training Synchronization section
        sync_box = wx.StaticBox(self.scroll_panel, label="Parallel Training Synchronization")
        sync_sizer = wx.StaticBoxSizer(sync_box, wx.VERTICAL)
        
        # Grid for synchronization parameters
        sync_grid = wx.FlexGridSizer(rows=6, cols=2, vgap=10, hgap=10)
        sync_grid.AddGrowableCol(1, 1)
        
        # Sync interval
        sync_grid.Add(wx.StaticText(self.scroll_panel, label="Model Sync Interval:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.sync_interval_ctrl = wx.SpinCtrl(self.scroll_panel, min=1, max=100, initial=5)
        sync_grid.Add(self.sync_interval_ctrl, 0, wx.EXPAND)
        
        # Parameter server
        sync_grid.Add(wx.StaticText(self.scroll_panel, label="Use Parameter Server:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.parameter_server_check = wx.CheckBox(self.scroll_panel, label="")
        self.parameter_server_check.SetValue(True)
        sync_grid.Add(self.parameter_server_check, 0)
        
        # Shared experience buffer
        sync_grid.Add(wx.StaticText(self.scroll_panel, label="Shared Experience Buffer:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.shared_experience_check = wx.CheckBox(self.scroll_panel, label="")
        self.shared_experience_check.SetValue(False)
        sync_grid.Add(self.shared_experience_check, 0)
        
        # Gradient aggregation
        sync_grid.Add(wx.StaticText(self.scroll_panel, label="Gradient Aggregation:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.gradient_aggregation_check = wx.CheckBox(self.scroll_panel, label="")
        self.gradient_aggregation_check.SetValue(True)
        sync_grid.Add(self.gradient_aggregation_check, 0)
        
        # Adaptive learning rate
        sync_grid.Add(wx.StaticText(self.scroll_panel, label="Adaptive Learning Rate:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.adaptive_lr_check = wx.CheckBox(self.scroll_panel, label="")
        self.adaptive_lr_check.SetValue(True)
        sync_grid.Add(self.adaptive_lr_check, 0)
        
        # Add help text for synchronization options
        help_text = wx.StaticText(self.scroll_panel, label=
            "• Model Sync Interval: How often workers synchronize model parameters\n"
            "• Parameter Server: Enable model parameter sharing across workers\n"
            "• Shared Experience: Share training experiences between workers\n"
            "• Gradient Aggregation: Average gradients across workers\n"
            "• Adaptive Learning Rate: Coordinate learning rates based on global performance")
        help_text.SetFont(wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_NORMAL))
        
        # Add synchronization parameters to sizer
        sync_sizer.Add(sync_grid, 0, wx.EXPAND | wx.ALL, 10)
        sync_sizer.Add(help_text, 0, wx.EXPAND | wx.ALL, 5)
        
        # Add all sections to content sizer (inside scrolled window)
        content_sizer.Add(training_sizer, 0, wx.EXPAND | wx.ALL, 10)
        content_sizer.Add(ppo_sizer, 0, wx.EXPAND | wx.ALL, 10)
        content_sizer.Add(device_sizer, 0, wx.EXPAND | wx.ALL, 10)
        content_sizer.Add(sync_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Set the sizer for the scrolled window
        self.scroll_panel.SetSizer(content_sizer)
        
        # Add the scrolled window to the main sizer
        main_sizer.Add(self.scroll_panel, 1, wx.EXPAND | wx.ALL, 5)
        
        # Start training button - keep this outside the scroll area at the bottom
        button_panel = wx.Panel(self)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.train_btn = wx.Button(button_panel, label="Start Training", size=(150, 40))
        button_sizer.Add(self.train_btn, 0, wx.ALIGN_CENTER)
        
        button_panel.SetSizer(button_sizer)
        main_sizer.Add(button_panel, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        # Set the main sizer for the panel
        self.SetSizer(main_sizer)
        
        # Bind events
        self.browse_btn.Bind(wx.EVT_BUTTON, self.on_browse)
        self.train_btn.Bind(wx.EVT_BUTTON, self.on_train)
        self.device_choice.Bind(wx.EVT_CHOICE, self.on_device_choice)
        self.parallel_check.Bind(wx.EVT_CHECKBOX, self.on_parallel_toggle)
        
        # Initially disable sync options if parallel training is not enabled
        self.on_parallel_toggle(None)
    
    def on_multi_gpu_toggle(self, event):
        """Enable/disable GPU selection based on multi-GPU checkbox"""
        enabled = self.multi_gpu_check.GetValue()
        for check in self.gpu_checks:
            check.Enable(enabled)
    
    def on_parallel_toggle(self, event):
        """Enable/disable parallel training options based on parallel checkbox"""
        enabled = self.parallel_check.GetValue()
        
        # Enable/disable environments per GPU control
        self.envs_per_gpu_ctrl.Enable(enabled)
        
        # Enable/disable synchronization options
        self.sync_interval_ctrl.Enable(enabled)
        self.parameter_server_check.Enable(enabled)
        self.shared_experience_check.Enable(enabled)
        self.gradient_aggregation_check.Enable(enabled)
        self.adaptive_lr_check.Enable(enabled)
        
        # If parallel training is disabled, also disable some sync options that only make sense with multiple workers
        if not enabled:
            self.parameter_server_check.SetValue(False)
            self.shared_experience_check.SetValue(False)
            self.gradient_aggregation_check.SetValue(False)
        else:
            # Re-enable sensible defaults when parallel training is enabled
            self.parameter_server_check.SetValue(True)
            self.gradient_aggregation_check.SetValue(True)
            self.adaptive_lr_check.SetValue(True)
    
    def on_device_choice(self, event):
        """Handle device selection change"""
        # If CPU is selected, disable multi-GPU and parallel options
        if self.device_choice.GetSelection() == 0:  # CPU
            if hasattr(self, 'multi_gpu_check'):
                self.multi_gpu_check.SetValue(False)
                self.multi_gpu_check.Enable(False)
                self.on_multi_gpu_toggle(None)
            
            self.parallel_check.SetValue(False)
            self.parallel_check.Enable(False)
            self.on_parallel_toggle(None)
        else:
            if hasattr(self, 'multi_gpu_check'):
                self.multi_gpu_check.Enable(True)
            
            self.parallel_check.Enable(True)
    
    def on_browse(self, event):
        """Open directory browser for save directory"""
        dlg = wx.DirDialog(self, "Choose Save Directory", defaultPath=self.save_dir_ctrl.GetValue())
        if dlg.ShowModal() == wx.ID_OK:
            self.save_dir_ctrl.SetValue(dlg.GetPath())
        dlg.Destroy()
    
    def on_train(self, event):
        """Start training"""
        # Update configuration from UI
        self.update_config()
        
        # Start training
        self.main_frame.on_start_training(event)
    
    def update_from_config(self):
        """Update UI elements from configuration"""
        config = self.main_frame.config
        
        # Training parameters
        self.batch_size_ctrl.SetValue(config.get('batch_size', 64))
        self.learning_rate_ctrl.SetValue(str(config.get('learning_rate', 0.0003)))
        self.iterations_ctrl.SetValue(config.get('num_iterations', 1000))
        self.steps_ctrl.SetValue(config.get('steps_per_iteration', 128))
        self.save_interval_ctrl.SetValue(config.get('save_interval', 50))
        self.save_dir_ctrl.SetValue(config.get('save_dir', 'models'))
        
        # Optimized environment option
        self.optimized_env_check.SetValue(config.get('use_optimized_env', True))
        
        # PPO parameters
        ppo_params = config.get('ppo_params', {})
        self.mini_batch_size_ctrl.SetValue(ppo_params.get('mini_batch_size', 16))
        self.ppo_epochs_ctrl.SetValue(ppo_params.get('ppo_epochs', 10))
        self.clip_param_ctrl.SetValue(str(ppo_params.get('clip_param', 0.2)))
        self.value_coef_ctrl.SetValue(str(ppo_params.get('value_coef', 0.5)))
        self.entropy_coef_ctrl.SetValue(str(ppo_params.get('entropy_coef', 0.01)))
        self.max_grad_norm_ctrl.SetValue(str(ppo_params.get('max_grad_norm', 0.5)))
        self.gamma_ctrl.SetValue(str(ppo_params.get('gamma', 0.99)))
        self.gae_lambda_ctrl.SetValue(str(ppo_params.get('gae_lambda', 0.95)))
        
        # Device configuration
        device = config.get('device', 'cpu')
        
        if device == 'cpu':
            self.device_choice.SetSelection(0)
        elif device.startswith('cuda:'):
            try:
                gpu_idx = int(device.split(':')[1])
                self.device_choice.SetSelection(gpu_idx + 1)  # +1 because CPU is first
            except (IndexError, ValueError):
                self.device_choice.SetSelection(1)  # Default to first GPU
        
        # Multi-GPU configuration
        if hasattr(self, 'multi_gpu_check'):
            multi_gpu = config.get('multi_gpu', False)
            self.multi_gpu_check.SetValue(multi_gpu)
            
            # Update GPU checkboxes
            devices = config.get('devices', [])
            for i, check in enumerate(self.gpu_checks):
                check.SetValue(i in devices)
            
            # Enable/disable GPU selection
            self.on_multi_gpu_toggle(None)
        
        # Mixed precision
        self.mixed_precision_check.SetValue(config.get('use_mixed_precision', True))
        
        # Parallel training
        self.parallel_check.SetValue(config.get('use_parallel', False))
        self.envs_per_gpu_ctrl.SetValue(config.get('envs_per_gpu', 8))
        
        # Synchronization parameters
        self.sync_interval_ctrl.SetValue(config.get('sync_interval', 5))
        self.parameter_server_check.SetValue(config.get('use_parameter_server', True))
        self.shared_experience_check.SetValue(config.get('shared_experience_buffer', False))
        self.gradient_aggregation_check.SetValue(config.get('gradient_aggregation', True))
        self.adaptive_lr_check.SetValue(config.get('adaptive_lr', True))
        
        # Update parallel training dependent UI
        self.on_parallel_toggle(None)
        
        # Update device-dependent UI
        self.on_device_choice(None)
    
    def update_config(self):
        """Update configuration from UI elements"""
        config = self.main_frame.config
        
        # Training parameters
        config['batch_size'] = self.batch_size_ctrl.GetValue()
        
        try:
            config['learning_rate'] = float(self.learning_rate_ctrl.GetValue())
        except ValueError:
            config['learning_rate'] = 0.0003
        
        config['num_iterations'] = self.iterations_ctrl.GetValue()
        config['steps_per_iteration'] = self.steps_ctrl.GetValue()
        config['save_interval'] = self.save_interval_ctrl.GetValue()
        config['save_dir'] = self.save_dir_ctrl.GetValue()
        
        # Optimized environment option
        config['use_optimized_env'] = self.optimized_env_check.GetValue()
        
        # PPO parameters
        ppo_params = {}
        ppo_params['mini_batch_size'] = self.mini_batch_size_ctrl.GetValue()
        ppo_params['ppo_epochs'] = self.ppo_epochs_ctrl.GetValue()
        
        try:
            ppo_params['clip_param'] = float(self.clip_param_ctrl.GetValue())
        except ValueError:
            ppo_params['clip_param'] = 0.2
        
        try:
            ppo_params['value_coef'] = float(self.value_coef_ctrl.GetValue())
        except ValueError:
            ppo_params['value_coef'] = 0.5
        
        try:
            ppo_params['entropy_coef'] = float(self.entropy_coef_ctrl.GetValue())
        except ValueError:
            ppo_params['entropy_coef'] = 0.01
        
        try:
            ppo_params['max_grad_norm'] = float(self.max_grad_norm_ctrl.GetValue())
        except ValueError:
            ppo_params['max_grad_norm'] = 0.5
        
        try:
            ppo_params['gamma'] = float(self.gamma_ctrl.GetValue())
        except ValueError:
            ppo_params['gamma'] = 0.99
        
        try:
            ppo_params['gae_lambda'] = float(self.gae_lambda_ctrl.GetValue())
        except ValueError:
            ppo_params['gae_lambda'] = 0.95
        
        config['ppo_params'] = ppo_params
        
        # Device configuration
        device_idx = self.device_choice.GetSelection()
        if device_idx == 0:
            config['device'] = 'cpu'
        else:
            config['device'] = f'cuda:{device_idx - 1}'  # -1 because CPU is first
        
        # Multi-GPU configuration
        if hasattr(self, 'multi_gpu_check'):
            config['multi_gpu'] = self.multi_gpu_check.GetValue()
            
            # Get selected GPUs
            devices = []
            if config['multi_gpu']:
                for i, check in enumerate(self.gpu_checks):
                    if check.GetValue():
                        devices.append(i)
            
            config['devices'] = devices
        
        # Mixed precision
        config['use_mixed_precision'] = self.mixed_precision_check.GetValue()
        
        # Parallel training
        config['use_parallel'] = self.parallel_check.GetValue()
        config['envs_per_gpu'] = self.envs_per_gpu_ctrl.GetValue()
        
        # Synchronization parameters
        config['sync_interval'] = self.sync_interval_ctrl.GetValue()
        config['use_parameter_server'] = self.parameter_server_check.GetValue()
        config['shared_experience_buffer'] = self.shared_experience_check.GetValue()
        config['gradient_aggregation'] = self.gradient_aggregation_check.GetValue()
        config['adaptive_lr'] = self.adaptive_lr_check.GetValue()
        
        print(f"Updated config: {config}")
