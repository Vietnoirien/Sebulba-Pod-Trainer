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
        self.active_league = None
        
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
        
        # Mixed precision checkbox
        self.mixed_precision_cb = wx.CheckBox(self, label="Use Mixed Precision Training")
        self.mixed_precision_cb.SetValue(True)
        gpu_sizer.Add(self.mixed_precision_cb, 0, wx.ALL, 5)
        
        # Parallel training options
        parallel_box = wx.StaticBox(self, label="Parallel Training")
        parallel_sizer = wx.StaticBoxSizer(parallel_box, wx.VERTICAL)
        
        # Enable parallel training checkbox
        self.parallel_training_cb = wx.CheckBox(self, label="Enable Parallel Training (Train All Members Simultaneously)")
        self.parallel_training_cb.SetValue(True)
        parallel_sizer.Add(self.parallel_training_cb, 0, wx.ALL, 5)
        
        # Max concurrent trainings
        concurrent_label = wx.StaticText(self, label="Max Concurrent Trainings (0 = auto):")
        self.concurrent_ctrl = wx.SpinCtrl(self, min=0, max=16, initial=0)
        concurrent_sizer = wx.BoxSizer(wx.HORIZONTAL)
        concurrent_sizer.Add(concurrent_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        concurrent_sizer.Add(self.concurrent_ctrl, 0)
        parallel_sizer.Add(concurrent_sizer, 0, wx.ALL, 5)
        
        # Add sections to config sizer
        config_sizer.Add(param_grid, 0, wx.EXPAND | wx.ALL, 10)
        config_sizer.Add(gpu_sizer, 0, wx.EXPAND | wx.ALL, 10)
        config_sizer.Add(parallel_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # League members section
        members_box = wx.StaticBox(self, label="League Members")
        members_sizer = wx.StaticBoxSizer(members_box, wx.VERTICAL)
        
        # Create grid for league members
        self.members_grid = wx.grid.Grid(self)
        self.members_grid.CreateGrid(10, 7)
        
        # Set column labels
        self.members_grid.SetColLabelValue(0, "ID")
        self.members_grid.SetColLabelValue(1, "Name")
        self.members_grid.SetColLabelValue(2, "ELO")
        self.members_grid.SetColLabelValue(3, "Win Rate")
        self.members_grid.SetColLabelValue(4, "Generation")
        self.members_grid.SetColLabelValue(5, "Source")
        self.members_grid.SetColLabelValue(6, "Status")
        
        # Set column widths
        self.members_grid.SetColSize(0, 50)
        self.members_grid.SetColSize(1, 150)
        self.members_grid.SetColSize(2, 80)
        self.members_grid.SetColSize(3, 100)
        self.members_grid.SetColSize(4, 100)
        self.members_grid.SetColSize(5, 120)
        self.members_grid.SetColSize(6, 100)
        
        # Make grid read-only
        for row in range(self.members_grid.GetNumberRows()):
            for col in range(self.members_grid.GetNumberCols()):
                self.members_grid.SetReadOnly(row, col)
        
        members_sizer.Add(self.members_grid, 1, wx.EXPAND | wx.ALL, 5)
        
        # League control buttons
        league_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.init_league_btn = wx.Button(self, label="Initialize League")
        self.import_model_btn = wx.Button(self, label="Import Trained Model")
        self.export_model_btn = wx.Button(self, label="Export Model")
        self.start_league_btn = wx.Button(self, label="Start League Training")
        self.stop_league_btn = wx.Button(self, label="Stop League Training")
        self.tournament_btn = wx.Button(self, label="Run Tournament")
        self.train_all_btn = wx.Button(self, label="Train All Members")
        
        league_btn_sizer.Add(self.init_league_btn, 0, wx.RIGHT, 5)
        league_btn_sizer.Add(self.import_model_btn, 0, wx.RIGHT, 5)
        league_btn_sizer.Add(self.export_model_btn, 0, wx.RIGHT, 5)
        league_btn_sizer.Add(self.train_all_btn, 0, wx.RIGHT, 5)
        league_btn_sizer.Add(self.start_league_btn, 0, wx.RIGHT, 5)
        league_btn_sizer.Add(self.stop_league_btn, 0, wx.RIGHT, 5)
        league_btn_sizer.Add(self.tournament_btn, 0)
        
        members_sizer.Add(league_btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        
        # League status section
        status_box = wx.StaticBox(self, label="League Status")
        status_sizer = wx.StaticBoxSizer(status_box, wx.VERTICAL)
        
        self.status_text = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 150))
        status_sizer.Add(self.status_text, 1, wx.EXPAND | wx.ALL, 5)
        
        # Training progress section
        progress_box = wx.StaticBox(self, label="Training Progress")
        progress_sizer = wx.StaticBoxSizer(progress_box, wx.VERTICAL)
        
        # Progress gauge
        self.progress_gauge = wx.Gauge(self, range=100)
        progress_sizer.Add(self.progress_gauge, 0, wx.EXPAND | wx.ALL, 5)
        
        # Progress text
        self.progress_text = wx.StaticText(self, label="Ready")
        progress_sizer.Add(self.progress_text, 0, wx.ALL, 5)
        
        # Add sections to main sizer
        main_sizer.Add(config_sizer, 0, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(members_sizer, 1, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(status_sizer, 0, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(progress_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Set sizer
        self.SetSizer(main_sizer)
        
        # Bind events
        self.dir_btn.Bind(wx.EVT_BUTTON, self.on_browse_dir)
        self.init_league_btn.Bind(wx.EVT_BUTTON, self.on_init_league)
        self.import_model_btn.Bind(wx.EVT_BUTTON, self.on_import_model)
        self.export_model_btn.Bind(wx.EVT_BUTTON, self.on_export_model)
        self.train_all_btn.Bind(wx.EVT_BUTTON, self.on_train_all_members)
        self.start_league_btn.Bind(wx.EVT_BUTTON, self.on_start_league)
        self.stop_league_btn.Bind(wx.EVT_BUTTON, self.on_stop_league)
        self.tournament_btn.Bind(wx.EVT_BUTTON, self.on_run_tournament)
        self.multi_gpu_cb.Bind(wx.EVT_CHECKBOX, self.on_multi_gpu_changed)
        self.parallel_training_cb.Bind(wx.EVT_CHECKBOX, self.on_parallel_training_changed)
        
        # Initially disable some buttons
        self.stop_league_btn.Disable()
        self.on_multi_gpu_changed(None)  # Initialize GPU controls state
        self.on_parallel_training_changed(None)  # Initialize parallel training controls state
        
        # Timer for updating member status during training
        self.update_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_update_timer, self.update_timer)
    
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
                
                # Show source information
                if 'imported_from' in member:
                    source_info = f"Imported ({member.get('import_format', 'unknown')})"
                else:
                    source_info = "League"
                self.members_grid.SetCellValue(i, 5, source_info)
                
                # Status column (will be updated during training)
                self.members_grid.SetCellValue(i, 6, "Ready")
            
            # Make grid read-only
            for row in range(self.members_grid.GetNumberRows()):
                for col in range(self.members_grid.GetNumberCols()):
                    self.members_grid.SetReadOnly(row, col)
            
            self.status_text.AppendText(f"Loaded {len(members)} league members from {metadata_path}\n")
            
        except Exception as e:
            self.status_text.AppendText(f"Error loading league members: {str(e)}\n")
    
    def update_member_status(self, member_idx: int, status: str):
        """Update the status of a specific member in the grid"""
        if member_idx < self.members_grid.GetNumberRows():
            self.members_grid.SetCellValue(member_idx, 6, status)
            self.members_grid.Refresh()
    
    def on_update_timer(self, event):
        """Timer event to update UI during training"""
        if self.league_running and self.active_league:
            # Update progress based on training threads
            if hasattr(self.active_league, 'training_threads'):
                total_members = len(self.active_league.league_members)
                active_threads = len([t for t in self.active_league.training_threads if t.is_alive()])
                
                if total_members > 0:
                    progress = max(0, min(100, int((total_members - active_threads) / total_members * 100)))
                    self.progress_gauge.SetValue(progress)
                    self.progress_text.SetLabel(f"Training: {active_threads} active threads, {progress}% complete")
                    
                    # Update member statuses
                    for i in range(total_members):
                        if i < len(self.active_league.training_threads):
                            thread = self.active_league.training_threads[i]
                            if thread.is_alive():
                                self.update_member_status(i, "Training")
                            else:
                                self.update_member_status(i, "Complete")
                        else:
                            self.update_member_status(i, "Waiting")
    
    def on_import_model(self, event):
        """Import a trained model from external source"""
        with wx.DirDialog(self, "Choose directory containing trained models") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                source_dir = dlg.GetPath()
                
                # Ask for member name
                with wx.TextEntryDialog(self, "Enter name for imported model:", "Import Model", "imported_model") as name_dlg:
                    if name_dlg.ShowModal() == wx.ID_OK:
                        member_name = name_dlg.GetValue()
                        
                        # Start import in a separate thread
                        self.status_text.AppendText(f"Importing model from {source_dir}...\n")
                        
                        import_thread = threading.Thread(
                            target=self.import_model_thread,
                            args=(source_dir, member_name)
                        )
                        import_thread.daemon = True
                        import_thread.start()
    
    def import_model_thread(self, source_dir, member_name):
        """Import model in a separate thread"""
        try:
            from ..training.league import PodLeague
            
            # Update config first
            wx.CallAfter(self.update_config)
            
            # Get configuration
            league_config = self.config.get('league_config', {})
            
            # Parse devices
            devices = league_config.get('devices', [0])
            
            # Create league instance
            league = PodLeague(
                device=torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
                league_size=league_config.get('league_size', 10),
                base_save_dir=league_config.get('base_save_dir', 'league_models'),
                multi_gpu=league_config.get('multi_gpu', False),
                devices=devices,
                batch_size=league_config.get('batch_size', 64),
                use_mixed_precision=league_config.get('use_mixed_precision', True)
            )
            
            # Import the model
            member_idx = league.import_trained_model(source_dir, member_name)
            
            # Update UI
            wx.CallAfter(self.status_text.AppendText, f"Successfully imported {member_name} as member {member_idx}\n")
            wx.CallAfter(self.load_league_members)
            
        except Exception as e:
            wx.CallAfter(self.status_text.AppendText, f"Error importing model: {str(e)}\n")
    
    def on_export_model(self, event):
        """Export a league member to trainer format"""
        # Get selected member from grid
        selected_row = self.members_grid.GetGridCursorRow()
        if selected_row < 0:
            wx.MessageBox("Please select a league member to export", "Export Model", wx.OK | wx.ICON_INFORMATION)
            return
        
        # Get member ID from grid
        try:
            member_id = int(self.members_grid.GetCellValue(selected_row, 0))
        except (ValueError, IndexError):
            wx.MessageBox("Invalid member selection", "Export Model", wx.OK | wx.ICON_ERROR)
            return
        
        # Choose export directory
        with wx.DirDialog(self, "Choose export directory") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                export_dir = dlg.GetPath()
                
                # Start export in a separate thread
                self.status_text.AppendText(f"Exporting member {member_id} to {export_dir}...\n")
                
                export_thread = threading.Thread(
                    target=self.export_model_thread,
                    args=(member_id, export_dir)
                )
                export_thread.daemon = True
                export_thread.start()
    
    def export_model_thread(self, member_id, export_dir):
        """Export model in a separate thread"""
        try:
            from ..training.league import PodLeague
            
            # Update config first
            wx.CallAfter(self.update_config)
            
            # Get configuration
            league_config = self.config.get('league_config', {})
            
            # Parse devices
            devices = league_config.get('devices', [0])
            
            # Create league instance
            league = PodLeague(
                device=torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
                league_size=league_config.get('league_size', 10),
                base_save_dir=league_config.get('base_save_dir', 'league_models'),
                multi_gpu=league_config.get('multi_gpu', False),
                devices=devices,
                batch_size=league_config.get('batch_size', 64),
                use_mixed_precision=league_config.get('use_mixed_precision', True)
            )
            
            # Export the model
            league.export_member_to_trainer_format(member_id, export_dir)
            
            # Update UI
            wx.CallAfter(self.status_text.AppendText, f"Successfully exported member {member_id} to {export_dir}\n")
            
        except Exception as e:
            wx.CallAfter(self.status_text.AppendText, f"Error exporting model: {str(e)}\n")
    
    def on_multi_gpu_changed(self, event):
        """Enable/disable GPU device selection based on multi-GPU checkbox"""
        is_multi_gpu = self.multi_gpu_cb.GetValue()
        self.devices_ctrl.Enable(is_multi_gpu)
    
    def on_parallel_training_changed(self, event):
        """Enable/disable parallel training controls"""
        is_parallel = self.parallel_training_cb.GetValue()
        self.concurrent_ctrl.Enable(is_parallel)
    
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
        
        self.mixed_precision_cb.SetValue(league_config.get('use_mixed_precision', True))
        
        # Parallel training settings
        self.parallel_training_cb.SetValue(league_config.get('parallel_training', True))
        self.concurrent_ctrl.SetValue(league_config.get('max_concurrent_trainings', 0))
        
        # Update controls state
        self.on_multi_gpu_changed(None)
        self.on_parallel_training_changed(None)
        
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
            'use_mixed_precision': self.mixed_precision_cb.GetValue(),
            'parallel_training': self.parallel_training_cb.GetValue(),
            'max_concurrent_trainings': self.concurrent_ctrl.GetValue()
        }
        
        self.config['league_config'] = league_config
    
    def on_browse_dir(self, event):
        """Open directory browser for league save directory"""
        with wx.DirDialog(self, "Choose League Save Directory", defaultPath=self.dir_ctrl.GetValue()) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.dir_ctrl.SetValue(dlg.GetPath())
                # Try to load league members from this directory
                self.load_league_members()
    
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
            
            # Create league with updated configuration
            league = PodLeague(
                device=torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
                league_size=league_config.get('league_size', 10),
                base_save_dir=league_config.get('base_save_dir', 'league_models'),
                multi_gpu=league_config.get('multi_gpu', False),
                devices=devices,
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
    
    def on_train_all_members(self, event):
        """Train all league members simultaneously"""
        if self.league_running:
            wx.MessageBox("League training is already running", "Training Error", wx.OK | wx.ICON_INFORMATION)
            return
        
        # Update config
        self.update_config()
        
        # Start training in a separate thread
        self.league_running = True
        self.status_text.AppendText("Starting parallel training for all members...\n")
        
        # Update button states
        self.start_league_btn.Disable()
        self.train_all_btn.Disable()
        self.init_league_btn.Disable()
        self.import_model_btn.Disable()
        self.export_model_btn.Disable()
        self.stop_league_btn.Enable()
        
        # Start progress updates
        self.progress_gauge.SetValue(0)
        self.progress_text.SetLabel("Starting training...")
        self.update_timer.Start(1000)  # Update every second
        
        train_thread = threading.Thread(target=self.train_all_members_thread)
        train_thread.daemon = True
        train_thread.start()
    
    def train_all_members_thread(self):
        """Train all members in a separate thread"""
        try:
            from ..training.league import PodLeague
            
            # Get configuration
            league_config = self.config.get('league_config', {})
            
            # Parse devices
            devices = league_config.get('devices', [0])
            
            # Create league with updated configuration
            league = PodLeague(
                device=torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
                league_size=league_config.get('league_size', 10),
                base_save_dir=league_config.get('base_save_dir', 'league_models'),
                multi_gpu=league_config.get('multi_gpu', False),
                devices=devices,
                batch_size=league_config.get('batch_size', 64),
                use_mixed_precision=league_config.get('use_mixed_precision', True)
            )
            
            # Store league reference for stopping and progress tracking
            self.active_league = league
            
            # Determine max concurrent trainings
            max_concurrent = league_config.get('max_concurrent_trainings', 0)
            if max_concurrent == 0:
                max_concurrent = None  # Let the league decide
            
            # Start parallel training
            wx.CallAfter(self.status_text.AppendText, "Starting parallel training for all members...\n")
            
            completed, failed = league.train_all_members_parallel(
                num_iterations=league_config.get('training_iterations', 100),
                max_concurrent_trainings=max_concurrent
            )
            
            # Update UI when done
            wx.CallAfter(self.on_training_complete, completed, failed)
            
        except Exception as e:
            wx.CallAfter(self.on_training_error, str(e))
    
    def on_start_league(self, event):
        """Start league evolution training"""
        if self.league_running:
            wx.MessageBox("League training is already running", "League Training", wx.OK | wx.ICON_INFORMATION)
            return
        
        # Update config
        self.update_config()
        
        # Start league training in a separate thread
        self.league_running = True
        self.status_text.AppendText("Starting league evolution...\n")
        
        # Update button states
        self.start_league_btn.Disable()
        self.train_all_btn.Disable()
        self.init_league_btn.Disable()
        self.import_model_btn.Disable()
        self.export_model_btn.Disable()
        self.stop_league_btn.Enable()
        
        # Start progress updates
        self.progress_gauge.SetValue(0)
        self.progress_text.SetLabel("Starting league evolution...")
        self.update_timer.Start(1000)  # Update every second
        
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
            
            # Create league with updated configuration
            league = PodLeague(
                device=torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
                league_size=league_config.get('league_size', 10),
                base_save_dir=league_config.get('base_save_dir', 'league_models'),
                multi_gpu=league_config.get('multi_gpu', False),
                devices=devices,
                batch_size=league_config.get('batch_size', 64),
                use_mixed_precision=league_config.get('use_mixed_precision', True)
            )
            
            # Start evolution
            wx.CallAfter(self.status_text.AppendText, "Starting league evolution...\n")
            
            # Store league reference for stopping
            self.active_league = league
            
            # Run evolution with parallel training option
            league.evolve_league(
                iterations=league_config.get('evolution_iterations', 10),
                training_iterations=league_config.get('training_iterations', 100),
                tournament_frequency=league_config.get('tournament_frequency', 2),
                parallel_training=league_config.get('parallel_training', True)
            )
            
            # Update UI when done
            wx.CallAfter(self.on_league_training_complete)
            
        except Exception as e:
            wx.CallAfter(self.on_league_training_error, str(e))
    
    def on_training_complete(self, completed: int, failed: int):
        """Called when parallel training of all members completes"""
        self.league_running = False
        self.update_timer.Stop()
        
        self.status_text.AppendText(f"Parallel training completed: {completed} successful, {failed} failed\n")
        
        # Update progress
        self.progress_gauge.SetValue(100)
        self.progress_text.SetLabel(f"Training complete: {completed} successful, {failed} failed")
        
        # Reset member statuses
        for i in range(self.members_grid.GetNumberRows()):
            self.update_member_status(i, "Ready")
        
        # Update button states
        self.start_league_btn.Enable()
        self.train_all_btn.Enable()
        self.init_league_btn.Enable()
        self.import_model_btn.Enable()
        self.export_model_btn.Enable()
        self.stop_league_btn.Disable()
        
        # Reload league members
        self.load_league_members()
    
    def on_training_error(self, error_msg: str):
        """Called when training encounters an error"""
        self.league_running = False
        self.update_timer.Stop()
        
        self.status_text.AppendText(f"Training error: {error_msg}\n")
        
        # Update progress
        self.progress_gauge.SetValue(0)
        self.progress_text.SetLabel("Training failed")
        
        # Reset member statuses
        for i in range(self.members_grid.GetNumberRows()):
            self.update_member_status(i, "Error")
        
        # Update button states
        self.start_league_btn.Enable()
        self.train_all_btn.Enable()
        self.init_league_btn.Enable()
        self.import_model_btn.Enable()
        self.export_model_btn.Enable()
        self.stop_league_btn.Disable()
    
    def on_league_training_complete(self):
        """Called when league evolution training completes"""
        self.league_running = False
        self.update_timer.Stop()
        
        self.status_text.AppendText("League evolution completed\n")
        
        # Update progress
        self.progress_gauge.SetValue(100)
        self.progress_text.SetLabel("League evolution complete")
        
        # Reset member statuses
        for i in range(self.members_grid.GetNumberRows()):
            self.update_member_status(i, "Ready")
        
        # Update button states
        self.start_league_btn.Enable()
        self.train_all_btn.Enable()
        self.init_league_btn.Enable()
        self.import_model_btn.Enable()
        self.export_model_btn.Enable()
        self.stop_league_btn.Disable()
        
        # Reload league members
        self.load_league_members()
    
    def on_league_training_error(self, error_msg):
        """Called when league training encounters an error"""
        self.league_running = False
        self.update_timer.Stop()
        
        self.status_text.AppendText(f"League training error: {error_msg}\n")
        
        # Update progress
        self.progress_gauge.SetValue(0)
        self.progress_text.SetLabel("League training failed")
        
        # Reset member statuses
        for i in range(self.members_grid.GetNumberRows()):
            self.update_member_status(i, "Error")
        
        # Update button states
        self.start_league_btn.Enable()
        self.train_all_btn.Enable()
        self.init_league_btn.Enable()
        self.import_model_btn.Enable()
        self.export_model_btn.Enable()
        self.stop_league_btn.Disable()
    
    def on_stop_league(self, event):
        """Stop league training"""
        if not self.league_running:
            return
        
        dlg = wx.MessageDialog(self, 
                              "Stop league training? The current operations will complete before stopping.",
                              "Stop League Training",
                              wx.YES_NO | wx.ICON_QUESTION)
        result = dlg.ShowModal()
        dlg.Destroy()
        
        if result == wx.ID_YES:
            # Set flag to stop training
            if hasattr(self, 'active_league') and self.active_league:
                self.status_text.AppendText("Stopping league training after current operations complete...\n")
                self.active_league.stop_requested = True
                
                # If we have parallel training, stop all threads
                if hasattr(self.active_league, 'stop_all_training'):
                    self.active_league.stop_all_training()
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
        self.progress_text.SetLabel("Running tournament...")
        
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
            
            # Create league with updated configuration
            league = PodLeague(
                device=torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
                league_size=league_config.get('league_size', 10),
                base_save_dir=league_config.get('base_save_dir', 'league_models'),
                multi_gpu=league_config.get('multi_gpu', False),
                devices=devices,
                batch_size=league_config.get('batch_size', 64),
                use_mixed_precision=league_config.get('use_mixed_precision', True)
            )
            
            # Run tournament
            wx.CallAfter(self.status_text.AppendText, "Running tournament...\n")
            league.run_tournament(matches_per_pair=1)
            
            # Update UI when done
            wx.CallAfter(self.status_text.AppendText, "Tournament completed\n")
            wx.CallAfter(self.progress_text.SetLabel, "Tournament complete")
            wx.CallAfter(self.load_league_members)
            
        except Exception as e:
            wx.CallAfter(self.status_text.AppendText, f"Tournament error: {str(e)}\n")
            wx.CallAfter(self.progress_text.SetLabel, "Tournament failed")
