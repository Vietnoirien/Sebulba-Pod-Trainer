import wx
import torch

class PreferencesDialog(wx.Dialog):
    def __init__(self, parent):
        super(PreferencesDialog, self).__init__(
            parent, title="Preferences", size=(500, 400)
        )
        
        self.parent = parent
        
        # Create UI elements
        self.create_ui()
        
        # Update UI from config
        self.update_from_config()
    
    def create_ui(self):
        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
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
        
        # Optimized environment option
        device_grid.Add(wx.StaticText(self, label="Use Optimized Environment:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.optimized_env_check = wx.CheckBox(self, label="")
        device_grid.Add(self.optimized_env_check, 0)
        
        # Add device configuration to sizer
        device_sizer.Add(device_grid, 0, wx.EXPAND | wx.ALL, 10)
        
        # Add all sections to main sizer
        main_sizer.Add(device_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Buttons
        button_sizer = wx.StdDialogButtonSizer()
        self.ok_button = wx.Button(self, wx.ID_OK)
        self.ok_button.SetDefault()
        button_sizer.AddButton(self.ok_button)
        
        self.cancel_button = wx.Button(self, wx.ID_CANCEL)
        button_sizer.AddButton(self.cancel_button)
        button_sizer.Realize()
        
        main_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        # Set sizer
        self.SetSizer(main_sizer)
        
        # Bind events
        self.ok_button.Bind(wx.EVT_BUTTON, self.on_ok)
        self.device_choice.Bind(wx.EVT_CHOICE, self.on_device_choice)
    
    def on_multi_gpu_toggle(self, event):
        """Enable/disable GPU selection based on multi-GPU checkbox"""
        enabled = self.multi_gpu_check.GetValue()
        for check in self.gpu_checks:
            check.Enable(enabled)
    
    def on_device_choice(self, event):
        """Handle device selection change"""
        # If CPU is selected, disable multi-GPU option
        if self.device_choice.GetSelection() == 0:  # CPU
            if hasattr(self, 'multi_gpu_check'):
                self.multi_gpu_check.SetValue(False)
                self.multi_gpu_check.Enable(False)
                self.on_multi_gpu_toggle(None)
        else:
            if hasattr(self, 'multi_gpu_check'):
                self.multi_gpu_check.Enable(True)
    
    def update_from_config(self):
        """Update UI elements from configuration"""
        config = self.parent.config
        
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
        
        # Optimized environment
        self.optimized_env_check.SetValue(config.get('use_optimized_env', True))
        
        # Update device-dependent UI
        self.on_device_choice(None)
    
    def on_ok(self, event):
        """Update configuration and close dialog"""
        config = self.parent.config
        
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
        
        # Optimized environment
        config['use_optimized_env'] = self.optimized_env_check.GetValue()
        
        # Update parent's device status
        self.parent.update_device_status()
        
        # Close dialog
        self.EndModal(wx.ID_OK)
