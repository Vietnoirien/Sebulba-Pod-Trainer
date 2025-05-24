import wx
import wx.grid
import torch
import threading
import time
from pathlib import Path

class LeaguePanel(wx.Panel):
    def __init__(self, parent, main_frame):
        super(LeaguePanel, self).__init__(parent)
        
        self.main_frame = main_frame
        self.config = main_frame.config
        
        # League state
        self.league_running = False
        
        # Create UI components
        self.create_ui()
        
        # Update UI from config
        self.update_from_config()
    
    def create_ui(self):
        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # League configuration section
        config_box = wx.StaticBox(self, label="League Configuration")
        config_sizer = wx.StaticBoxSizer(config_box, wx.VERTICAL)
        
        # Create a grid sizer for configuration
        param_grid = wx.FlexGridSizer(rows=0, cols=2, vgap=10, hgap=10)
        param_grid.AddGrowableCol(1)
        
        # League size
        size_label = wx.StaticText(self, label="League Size:")
        self.size_ctrl = wx.SpinCtrl(self, min=2, max=50, initial=10)
        param_grid.Add(size_label, 0, wx.ALIGN_CENTER_VERTICAL)
        param_grid.Add(self.size_ctrl, 0, wx.EXPAND)
        
        # Base save directory
        dir_label = wx.StaticText(self, label="League Save Directory:")
        dir_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.dir_ctrl = wx.TextCtrl(self, value="league_models")
        self.dir_btn = wx.Button(self, label="Browse...")
        dir_sizer.Add(self.dir_ctrl, 1, wx.EXPAND | wx.RIGHT, 5)
        dir_sizer.Add(self.dir_btn, 0)
        param_grid.Add(dir_label, 0, wx.ALIGN_CENTER_VERTICAL)
        param_grid.Add(dir_sizer, 0, wx.EXPAND)
        
        # Evolution iterations
        evolution_label = wx.StaticText(self, label="Evolution Iterations:")
        self.evolution_ctrl = wx.SpinCtrl(self, min=1, max=1000, initial=10)
        param_grid.Add(evolution_label, 0, wx.ALIGN_CENTER_VERTICAL)
        param_grid.Add(self.evolution_ctrl, 0, wx.EXPAND)
        
        # Training iterations per member
        training_label = wx.StaticText(self, label="Training Iterations per Member:")
        self.training_ctrl = wx.SpinCtrl(self, min=1, max=1000, initial=100)
        param_grid.Add(training_label, 0, wx.ALIGN_CENTER_VERTICAL)
        param_grid.Add(self.training_ctrl, 0, wx.EXPAND)
        
        # Tournament frequency
        tournament_label = wx.StaticText(self, label="Tournament Frequency:")
        self.tournament_ctrl = wx.SpinCtrl(self, min=1, max=100, initial=2)
        param_grid.Add(tournament_label, 0, wx.ALIGN_CENTER_VERTICAL)
        param_grid.Add(self.tournament_ctrl, 0, wx.EXPAND)
        
        # Batch size
        batch_label = wx.StaticText(self, label="Batch Size:")
        self.batch_ctrl = wx.SpinCtrl(self, min=8, max=512, initial=64)
        param_grid.Add(batch_label, 0, wx.ALIGN_CENTER_VERTICAL)
        param_grid.Add(self.batch_ctrl, 0, wx.EXPAND)
        
        # GPU Optimization section
        gpu_box = wx.StaticBox(self, label="GPU Optimization")
        gpu_sizer = wx.StaticBoxSizer(gpu_box, wx.VERTICAL)
        
        # Multi-GPU checkbox
        self.multi_gpu_cb = wx.CheckBox(self, label="Use Multiple GPUs")
        gpu_sizer.Add(self.multi_gpu_cb, 0, wx.ALL, 5)
        
        # GPU devices
        devices_label = wx.StaticText(self, label="GPU Devices (comma-separated indices):")
        self.devices_ctrl = wx.TextCtrl(self, value="0")
        gpu_sizer.Add(devices_label, 0, wx.ALL, 5)
        gpu_sizer.Add(self.devices_ctrl, 0, wx.EXPAND | wx.ALL, 5)
        
        # Optimized environment checkbox
        self.opt_env_cb = wx.CheckBox(self, label="Use Optimized Environment")
        self.opt_env_cb.SetValue(True)
        gpu_sizer.Add(self.opt_env_cb, 0, wx.ALL, 5)
        
        # Optimized trainer checkbox
        self.opt_trainer_cb = wx.CheckBox(self, label="Use Optimized Trainer")
        self.opt_trainer_cb.SetValue(True)
        gpu_sizer.Add(self.opt_trainer_cb, 0, wx.ALL, 5)
        
        # Mixed precision checkbox
        self.mixed_precision_cb = wx.CheckBox(self, label="Use Mixed Precision Training")
        self.mixed_precision_cb.SetValue(True)
        gpu_sizer.Add(self.mixed_precision_cb, 0, wx.ALL, 5)
        
        # Add GPU optimization section to config sizer
        config_sizer.Add(param_grid, 0, wx.EXPAND | wx.ALL, 10)
        config_sizer.Add(gpu_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # League members section
        members_box = wx.StaticBox(self, label="League Members")
        members_sizer = wx.StaticBoxSizer(members_box, wx.VERTICAL)
        
        # Create grid for league members
        self.members_grid = wx.grid.Grid(self)
        self.members_grid.CreateGrid(10, 5)  # Default 10 rows, 5 columns
        
        # Set column labels
        self.members_grid.SetColLabelValue(0, "ID")
        self.members_grid.SetColLabelValue(1, "Name")
        self.members_grid.SetColLabelValue(2, "ELO")
        self.members_grid.SetColLabelValue(3, "Win Rate")
        self.members_grid.SetColLabelValue(4, "Generation")
        
        # Set column widths
        self.members_grid.SetColSize(0, 50)
        self.members_grid.SetColSize(1, 150)
        self.members_grid.SetColSize(2, 80)
        self.members_grid.SetColSize(3, 100)
        self.members_grid.SetColSize(4, 100)
        
        # Make grid read-only
        for row in range(self.members_grid.GetNumberRows()):
            for col in range(self.members_grid.GetNumberCols()):
                self.members_grid.SetReadOnly(row, col)
        
        members_sizer.Add(self.members_grid, 1, wx.EXPAND | wx.ALL, 5)
        
        # League control buttons
        league_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.init_league_btn = wx.Button(self, label="Initialize League")
        self.start_league_btn = wx.Button(self, label="Start League Training")
        self.stop_league_btn = wx.Button(self, label="Stop League Training")
        self.tournament_btn = wx.Button(self, label="Run Tournament")
        
        league_btn_sizer.Add(self.init_league_btn, 0, wx.RIGHT, 5)
        league_btn_sizer.Add(self.start_league_btn, 0, wx.RIGHT, 5)
        league_btn_sizer.Add(self.stop_league_btn, 0, wx.RIGHT, 5)
        league_btn_sizer.Add(self.tournament_btn, 0)
        
        members_sizer.Add(league_btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        
        # League status section
        status_box = wx.StaticBox(self, label="League Status")
        status_sizer = wx.StaticBoxSizer(status_box, wx.VERTICAL)
        
        self.status_text = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 150))
        status_sizer.Add(self.status_text, 1, wx.EXPAND | wx.ALL, 5)
        
        # Add sections to main sizer
        main_sizer.Add(config_sizer, 0, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(members_sizer, 1, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(status_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Set sizer
        self.SetSizer(main_sizer)
        
        # Bind events
        self.dir_btn.Bind(wx.EVT_BUTTON, self.on_browse_dir)
        self.init_league_btn.Bind(wx.EVT_BUTTON, self.on_init_league)
        self.start_league_btn.Bind(wx.EVT_BUTTON, self.on_start_league)
        self.stop_league_btn.Bind(wx.EVT_BUTTON, self.on_stop_league)
        self.tournament_btn.Bind(wx.EVT_BUTTON, self.on_run_tournament)
        self.multi_gpu_cb.Bind(wx.EVT_CHECKBOX, self.on_multi_gpu_changed)
        
        # Initially disable some buttons
        self.stop_league_btn.Disable()
        self.on_multi_gpu_changed(None)  # Initialize GPU controls state
    
    def on_multi_gpu_changed(self, event):
        """Enable/disable GPU device selection based on multi-GPU checkbox"""
        is_multi_gpu = self.multi_gpu_cb.GetValue()
        self.devices_ctrl.Enable(is_multi_gpu)
    
    def update_from_config(self):
        """Update UI elements from the current configuration"""
        league_config = self.config.get('league_config', {})
        
        self.size_ctrl.SetValue(league_config.get('league_size', 10))
        self.dir_ctrl.SetValue(league_config.get('base_save_dir', 'league_models'))
        self.evolution_ctrl.SetValue(league_config.get('evolution_iterations', 10))
        self.training_ctrl.SetValue(league_config.get('training_iterations', 100))
        self.tournament_ctrl.SetValue(league_config.get('tournament_frequency', 2))
        self.batch_ctrl.SetValue(league_config.get('batch_size', 64))
        
        # GPU optimization settings
        self.multi_gpu_cb.SetValue(league_config.get('multi_gpu', False))
        
        # Parse devices list
        devices = league_config.get('devices', [0])
        if isinstance(devices, list):
            self.devices_ctrl.SetValue(','.join(map(str, devices)))
        else:
            self.devices_ctrl.SetValue('0')
        
        self.opt_env_cb.SetValue(league_config.get('use_optimized_env', True))
        self.opt_trainer_cb.SetValue(league_config.get('use_optimized_trainer', True))
        self.mixed_precision_cb.SetValue(league_config.get('use_mixed_precision', True))
        
        # Update GPU controls state
        self.on_multi_gpu_changed(None)
        
        # Load league members if available
        self.load_league_members()
    
    def update_config(self):
        """Update configuration from UI elements"""
        # Parse GPU devices
        devices_str = self.devices_ctrl.GetValue()
        try:
            devices = [int(d.strip()) for d in devices_str.split(',') if d.strip()]
            if not devices:
                devices = [0]
        except ValueError:
            devices = [0]
            self.status_text.AppendText("Invalid GPU device indices, using default [0]\n")
        
        league_config = {
            'league_size': self.size_ctrl.GetValue(),
            'base_save_dir': self.dir_ctrl.GetValue(),
            'evolution_iterations': self.evolution_ctrl.GetValue(),
            'training_iterations': self.training_ctrl.GetValue(),
            'tournament_frequency': self.tournament_ctrl.GetValue(),
            'batch_size': self.batch_ctrl.GetValue(),
            'multi_gpu': self.multi_gpu_cb.GetValue(),
            'devices': devices,
            'use_optimized_env': self.opt_env_cb.GetValue(),
            'use_optimized_trainer': self.opt_trainer_cb.GetValue(),
            'use_mixed_precision': self.mixed_precision_cb.GetValue()
        }
        
        self.config['league_config'] = league_config
    
    def on_browse_dir(self, event):
        """Open directory browser for league save directory"""
        with wx.DirDialog(self, "Choose League Save Directory", defaultPath=self.dir_ctrl.GetValue()) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.dir_ctrl.SetValue(dlg.GetPath())
                # Try to load league members from this directory
                self.load_league_members()
    
    def load_league_members(self):
        """Load league members from the save directory"""
        try:
            import json
            from pathlib import Path
            
            save_dir = Path(self.dir_ctrl.GetValue())
            metadata_path = save_dir / 'league_metadata.json'
            
            if not metadata_path.exists():
                # Clear grid if no metadata exists
                self.members_grid.ClearGrid()
                return
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            members = metadata.get('members', [])
            
            # Resize grid if needed
            if self.members_grid.GetNumberRows() != len(members):
                if self.members_grid.GetNumberRows() > len(members):
                    self.members_grid.DeleteRows(0, self.members_grid.GetNumberRows() - len(members))
                else:
                    self.members_grid.AppendRows(len(members) - self.members_grid.GetNumberRows())
            
            # Update grid with member data
            for i, member in enumerate(members):
                self.members_grid.SetCellValue(i, 0, str(member.get('id', i)))
                self.members_grid.SetCellValue(i, 1, member.get('name', f"agent_{i}"))
                self.members_grid.SetCellValue(i, 2, str(round(member.get('elo', 1000), 1)))
                
                wins = member.get('wins', 0)
                matches = member.get('matches', 0)
                win_rate = wins / max(1, matches)
                self.members_grid.SetCellValue(i, 3, f"{win_rate:.2f} ({wins}/{matches})")
                
                self.members_grid.SetCellValue(i, 4, str(member.get('generation', 0)))
            
            # Make grid read-only
            for row in range(self.members_grid.GetNumberRows()):
                for col in range(self.members_grid.GetNumberCols()):
                    self.members_grid.SetReadOnly(row, col)
            
            self.status_text.AppendText(f"Loaded {len(members)} league members from {metadata_path}\n")
            
        except Exception as e:
            self.status_text.AppendText(f"Error loading league members: {str(e)}\n")
    
    def on_init_league(self, event):
        """Initialize a new league"""
        dlg = wx.MessageDialog(self, 
                              "Initialize a new league? This will overwrite any existing league in the selected directory.",
                              "Initialize League",
                              wx.YES_NO | wx.ICON_QUESTION)
        result = dlg.ShowModal()
        dlg.Destroy()
        
        if result == wx.ID_YES:
            # Update config
            self.update_config()
            
            # Start initialization in a separate thread
            self.status_text.Clear()
            self.status_text.AppendText("Initializing league...\n")
            
            init_thread = threading.Thread(target=self.initialize_league)
            init_thread.daemon = True
            init_thread.start()
    
    def initialize_league(self):
        """Initialize a new league in a separate thread"""
        try:
            from ..training.league import PodLeague
            
            # Get configuration
            league_config = self.config.get('league_config', {})
            
            # Parse devices
            devices = league_config.get('devices', [0])
            
            # Create league with GPU optimization settings
            league = PodLeague(
                device=torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
                league_size=league_config.get('league_size', 10),
                base_save_dir=league_config.get('base_save_dir', 'league_models'),
                multi_gpu=league_config.get('multi_gpu', False),
                devices=devices,
                use_optimized_env=league_config.get('use_optimized_env', True),
                use_optimized_trainer=league_config.get('use_optimized_trainer', True),
                batch_size=league_config.get('batch_size', 64),
                use_mixed_precision=league_config.get('use_mixed_precision', True)
            )
            
            # Force initialization
            league.initialize_league()
            
            # Update UI
            wx.CallAfter(self.status_text.AppendText, f"League initialized with {league_config.get('league_size', 10)} members\n")
            wx.CallAfter(self.load_league_members)
            
        except Exception as e:
            wx.CallAfter(self.status_text.AppendText, f"Error initializing league: {str(e)}\n")
    
    def on_start_league(self, event):
        """Start league training"""
        if self.league_running:
            wx.MessageBox("League training is already running", "League Training", wx.OK | wx.ICON_INFORMATION)
            return
        
        # Update config
        self.update_config()
        
        # Start league training in a separate thread
        self.league_running = True
        self.status_text.AppendText("Starting league training...\n")
        
        # Update button states
        self.start_league_btn.Disable()
        self.init_league_btn.Disable()
        self.stop_league_btn.Enable()
        
        league_thread = threading.Thread(target=self.start_league_training)
        league_thread.daemon = True
        league_thread.start()
    
    def start_league_training(self):
        """Start league training in a separate thread"""
        try:
            from ..training.league import PodLeague
            
            # Get configuration
            league_config = self.config.get('league_config', {})
            
            # Parse devices
            devices = league_config.get('devices', [0])
            
            # Create league with GPU optimization settings
            league = PodLeague(
                device=torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
                league_size=league_config.get('league_size', 10),
                base_save_dir=league_config.get('base_save_dir', 'league_models'),
                multi_gpu=league_config.get('multi_gpu', False),
                devices=devices,
                use_optimized_env=league_config.get('use_optimized_env', True),
                use_optimized_trainer=league_config.get('use_optimized_trainer', True),
                batch_size=league_config.get('batch_size', 64),
                use_mixed_precision=league_config.get('use_mixed_precision', True)
            )
            
            # Start evolution
            wx.CallAfter(self.status_text.AppendText, "Starting league evolution...\n")
            
            # Store league reference for stopping
            self.active_league = league
            
            # Run evolution
            league.evolve_league(
                iterations=league_config.get('evolution_iterations', 10),
                training_iterations=league_config.get('training_iterations', 100),
                tournament_frequency=league_config.get('tournament_frequency', 2)
            )
            
            # Update UI when done
            wx.CallAfter(self.on_league_training_complete)
            
        except Exception as e:
            wx.CallAfter(self.on_league_training_error, str(e))
    
    def on_league_training_complete(self):
        """Called when league training completes"""
        self.league_running = False
        self.status_text.AppendText("League training completed\n")
        
        # Update button states
        self.start_league_btn.Enable()
        self.init_league_btn.Enable()
        self.stop_league_btn.Disable()
        
        # Reload league members
        self.load_league_members()
    
    def on_league_training_error(self, error_msg):
        """Called when league training encounters an error"""
        self.league_running = False
        self.status_text.AppendText(f"League training error: {error_msg}\n")
        
        # Update button states
        self.start_league_btn.Enable()
        self.init_league_btn.Enable()
        self.stop_league_btn.Disable()
    
    def on_stop_league(self, event):
        """Stop league training"""
        if not self.league_running:
            return
        
        dlg = wx.MessageDialog(self, 
                              "Stop league training? The current iteration will complete before stopping.",
                              "Stop League Training",
                              wx.YES_NO | wx.ICON_QUESTION)
        result = dlg.ShowModal()
        dlg.Destroy()
        
        if result == wx.ID_YES:
            # Set flag to stop training
            if hasattr(self, 'active_league'):
                self.status_text.AppendText("Stopping league training after current iteration completes...\n")
                self.active_league.stop_requested = True
            else:
                self.status_text.AppendText("No active league found to stop\n")
    
    def on_run_tournament(self, event):
        """Run a tournament with the current league members"""
        if self.league_running:
            wx.MessageBox("Cannot run tournament while league training is in progress", 
                         "Tournament Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Update config
        self.update_config()
        
        # Start tournament in a separate thread
        self.status_text.AppendText("Starting tournament...\n")
        
        tournament_thread = threading.Thread(target=self.run_tournament)
        tournament_thread.daemon = True
        tournament_thread.start()
    
    def run_tournament(self):
        """Run a tournament in a separate thread"""
        try:
            from ..training.league import PodLeague
            
            # Get configuration
            league_config = self.config.get('league_config', {})
            
            # Parse devices
            devices = league_config.get('devices', [0])
            
            # Create league with GPU optimization settings
            league = PodLeague(
                device=torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
                league_size=league_config.get('league_size', 10),
                base_save_dir=league_config.get('base_save_dir', 'league_models'),
                multi_gpu=league_config.get('multi_gpu', False),
                devices=devices,
                use_optimized_env=league_config.get('use_optimized_env', True),
                use_optimized_trainer=league_config.get('use_optimized_trainer', True),
                batch_size=league_config.get('batch_size', 64),
                use_mixed_precision=league_config.get('use_mixed_precision', True)
            )
            
            # Run tournament
            wx.CallAfter(self.status_text.AppendText, "Running tournament...\n")
            league.run_tournament(matches_per_pair=1)
            
            # Update UI when done
            wx.CallAfter(self.status_text.AppendText, "Tournament completed\n")
            wx.CallAfter(self.load_league_members)
            
        except Exception as e:
            wx.CallAfter(self.status_text.AppendText, f"Tournament error: {str(e)}\n")
