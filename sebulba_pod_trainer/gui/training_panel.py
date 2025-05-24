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
        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Training parameters section
        training_box = wx.StaticBox(self, label="Training Parameters")
        training_sizer = wx.StaticBoxSizer(training_box, wx.VERTICAL)
        
        # Grid for parameters
        param_grid = wx.FlexGridSizer(rows=10, cols=2, vgap=10, hgap=10)
        param_grid.AddGrowableCol(1, 1)
        
        # Batch size
        param_grid.Add(wx.StaticText(self, label="Batch Size:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.batch_size_ctrl = wx.SpinCtrl(self, min=1, max=1024, initial=64)
        param_grid.Add(self.batch_size_ctrl, 0, wx.EXPAND)
        
        # Learning rate
        param_grid.Add(wx.StaticText(self, label="Learning Rate:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.learning_rate_ctrl = wx.TextCtrl(self, value="0.0003")
        param_grid.Add(self.learning_rate_ctrl, 0, wx.EXPAND)
        
        # Number of iterations
        param_grid.Add(wx.StaticText(self, label="Number of Iterations:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.iterations_ctrl = wx.SpinCtrl(self, min=1, max=100000, initial=1000)
        param_grid.Add(self.iterations_ctrl, 0, wx.EXPAND)
        
        # Steps per iteration
        param_grid.Add(wx.StaticText(self, label="Steps per Iteration:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.steps_ctrl = wx.SpinCtrl(self, min=1, max=10000, initial=128)
        param_grid.Add(self.steps_ctrl, 0, wx.EXPAND)
        
        # Save interval
        param_grid.Add(wx.StaticText(self, label="Save Interval:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.save_interval_ctrl = wx.SpinCtrl(self, min=1, max=1000, initial=50)
        param_grid.Add(self.save_interval_ctrl, 0, wx.EXPAND)
        
        # Save directory
        param_grid.Add(wx.StaticText(self, label="Save Directory:"), 0, wx.ALIGN_CENTER_VERTICAL)
        save_dir_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.save_dir_ctrl = wx.TextCtrl(self, value="models")
        save_dir_sizer.Add(self.save_dir_ctrl, 1, wx.EXPAND)
        self.browse_btn = wx.Button(self, label="Browse...")
        save_dir_sizer.Add(self.browse_btn, 0, wx.LEFT, 5)
        param_grid.Add(save_dir_sizer, 0, wx.EXPAND)
        
        # Optimized environment option
        param_grid.Add(wx.StaticText(self, label="Use Optimized Environment:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.optimized_env_check = wx.CheckBox(self, label="")
        param_grid.Add(self.optimized_env_check, 0)
        
        # Add parameters to sizer
        training_sizer.Add(param_grid, 0, wx.EXPAND | wx.ALL, 10)
        
        # PPO parameters section
        ppo_box = wx.StaticBox(self, label="PPO Parameters")
        ppo_sizer = wx.StaticBoxSizer(ppo_box, wx.VERTICAL)
        
        # Grid for PPO parameters
        ppo_grid = wx.FlexGridSizer(rows=8, cols=2, vgap=10, hgap=10)
        ppo_grid.AddGrowableCol(1, 1)
        
        # Mini batch size
        ppo_grid.Add(wx.StaticText(self, label="Mini Batch Size:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.mini_batch_size_ctrl = wx.SpinCtrl(self, min=1, max=512, initial=16)
        ppo_grid.Add(self.mini_batch_size_ctrl, 0, wx.EXPAND)
        
        # PPO epochs
        ppo_grid.Add(wx.StaticText(self, label="PPO Epochs:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.ppo_epochs_ctrl = wx.SpinCtrl(self, min=1, max=100, initial=10)
        ppo_grid.Add(self.ppo_epochs_ctrl, 0, wx.EXPAND)
        
        # Clip parameter
        ppo_grid.Add(wx.StaticText(self, label="Clip Parameter:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.clip_param_ctrl = wx.TextCtrl(self, value="0.2")
        ppo_grid.Add(self.clip_param_ctrl, 0, wx.EXPAND)
        
        # Value coefficient
        ppo_grid.Add(wx.StaticText(self, label="Value Coefficient:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.value_coef_ctrl = wx.TextCtrl(self, value="0.5")
        ppo_grid.Add(self.value_coef_ctrl, 0, wx.EXPAND)
        
        # Entropy coefficient
        ppo_grid.Add(wx.StaticText(self, label="Entropy Coefficient:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.entropy_coef_ctrl = wx.TextCtrl(self, value="0.01")
        ppo_grid.Add(self.entropy_coef_ctrl, 0, wx.EXPAND)
        
        # Max gradient norm
        ppo_grid.Add(wx.StaticText(self, label="Max Gradient Norm:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.max_grad_norm_ctrl = wx.TextCtrl(self, value="0.5")
        ppo_grid.Add(self.max_grad_norm_ctrl, 0, wx.EXPAND)
        
        # Gamma
        ppo_grid.Add(wx.StaticText(self, label="Gamma (Discount Factor):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.gamma_ctrl = wx.TextCtrl(self, value="0.99")
        ppo_grid.Add(self.gamma_ctrl, 0, wx.EXPAND)
        
        # GAE Lambda
        ppo_grid.Add(wx.StaticText(self, label="GAE Lambda:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.gae_lambda_ctrl = wx.TextCtrl(self, value="0.95")
        ppo_grid.Add(self.gae_lambda_ctrl, 0, wx.EXPAND)
        
        # Add PPO parameters to sizer
        ppo_sizer.Add(ppo_grid, 0, wx.EXPAND | wx.ALL, 10)
        
        # Device configuration section
        device_box = wx.StaticBox(self, label="Device Configuration")
        device_sizer = wx.StaticBoxSizer(device_box, wx.VERTICAL)
        
        # Device selection
        device_grid = wx.FlexGridSizer(rows=0, cols=2, vgap=10, hgap=10)
        device_grid.AddGrowableCol(1, 1)
        
        device_grid.Add(wx.StaticText(self, label="Training Device:"), 0, wx.ALIGN_CENTER_VERTICAL)
        
        # Create device choices
        device_choices = ["CPU"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_choices.append(f"CUDA:{i} ({torch.cuda.get_device_name(i)})")
        
        self.device_choice = wx.Choice(self, choices=device_choices)
        self.device_choice.SetSelection(0)  # Default to CPU
        device_grid.Add(self.device_choice, 0, wx.EXPAND)
        
        # Multi-GPU option
        if torch.cuda.device_count() > 1:
            device_grid.Add(wx.StaticText(self, label="Use Multiple GPUs:"), 0, wx.ALIGN_CENTER_VERTICAL)
            self.multi_gpu_check = wx.CheckBox(self, label="")
            device_grid.Add(self.multi_gpu_check, 0)
            
            # GPU selection
            device_grid.Add(wx.StaticText(self, label="Select GPUs:"), 0, wx.ALIGN_CENTER_VERTICAL)
            
            # Create a panel for GPU checkboxes
            gpu_panel = wx.Panel(self)
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
        device_grid.Add(wx.StaticText(self, label="Use Mixed Precision:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.mixed_precision_check = wx.CheckBox(self, label="")
        device_grid.Add(self.mixed_precision_check, 0)
        
        # Parallel training option
        device_grid.Add(wx.StaticText(self, label="Use Parallel Training:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.parallel_check = wx.CheckBox(self, label="")
        device_grid.Add(self.parallel_check, 0)
        
        # Environments per GPU
        device_grid.Add(wx.StaticText(self, label="Environments per GPU:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.envs_per_gpu_ctrl = wx.SpinCtrl(self, min=1, max=16, initial=8)
        device_grid.Add(self.envs_per_gpu_ctrl, 0, wx.EXPAND)
        
        # Add device configuration to sizer
        device_sizer.Add(device_grid, 0, wx.EXPAND | wx.ALL, 10)
        
        # Add all sections to main sizer
        main_sizer.Add(training_sizer, 0, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(ppo_sizer, 0, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(device_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Start training button
        self.train_btn = wx.Button(self, label="Start Training")
        main_sizer.Add(self.train_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        # Set sizer
        self.SetSizer(main_sizer)
        
        # Bind events
        self.browse_btn.Bind(wx.EVT_BUTTON, self.on_browse)
        self.train_btn.Bind(wx.EVT_BUTTON, self.on_train)
        self.device_choice.Bind(wx.EVT_CHOICE, self.on_device_choice)
        self.parallel_check.Bind(wx.EVT_CHECKBOX, self.on_parallel_toggle)
    
    def on_multi_gpu_toggle(self, event):
        """Enable/disable GPU selection based on multi-GPU checkbox"""
        enabled = self.multi_gpu_check.GetValue()
        for check in self.gpu_checks:
            check.Enable(enabled)
    
    def on_parallel_toggle(self, event):
        """Enable/disable environments per GPU control based on parallel checkbox"""
        enabled = self.parallel_check.GetValue()
        self.envs_per_gpu_ctrl.Enable(enabled)
    
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
        
        print(f"Updated config: {config}")
