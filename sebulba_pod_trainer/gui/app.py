import wx
import torch
import os
from pathlib import Path
import threading
import json
from copy import deepcopy

from ..models.neural_pod import PodNetwork
from ..environment.race_env import RaceEnvironment
from ..environment.optimized_race_env import OptimizedRaceEnvironment
from ..training.trainer import PPOTrainer
from ..training.optimized_trainer import OptimizedPPOTrainer
from ..training.league import PodLeague
from ..export.model_exporter import ModelExporter

from .preferences_dialog import PreferencesDialog
from .training_panel import TrainingPanel
from .visualization_panel import VisualizationPanel
from .league_panel import LeaguePanel
from .model_panel import ModelPanel
from .export_panel import ExportPanel

class SebulbaPodTrainerApp(wx.App):
    def OnInit(self):
        self.frame = MainFrame(None, title="Sebulba Pod Trainer")
        self.frame.Show()
        return True

class MainFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MainFrame, self).__init__(*args, **kw)
        
        # Set up the main frame
        self.SetSize((1200, 800))
        self.SetMinSize((800, 600))
        
        # Initialize configuration
        self.config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'multi_gpu': False,
            'devices': [],
            'batch_size': 64,
            'learning_rate': 3e-4,
            'save_dir': 'models',
            'use_mixed_precision': True,
            'use_optimized_env': True,  # New option for optimized environment
            'network_config': {
                'observation_dim': 25,
                'hidden_layers': [
                    {'type': 'Linear+ReLU', 'size': 32},
                    {'type': 'Linear+ReLU', 'size': 32}
                ],
                'policy_hidden_size': 32,
                'value_hidden_size': 32,
                'special_hidden_size': 32
            },
            'use_parallel': False,
            'envs_per_gpu': 8,
            'num_iterations': 1000,
            'steps_per_iteration': 128,
            'save_interval': 50,
            'ppo_params': {
                'mini_batch_size': 16,
                'ppo_epochs': 10,
                'clip_param': 0.2,
                'value_coef': 0.5,
                'entropy_coef': 0.01,
                'max_grad_norm': 0.5,
                'gamma': 0.99,
                'gae_lambda': 0.95
            }
        }
        
        # If multi_gpu is not explicitly enabled, use only one GPU
        if not self.config.get('multi_gpu', False):
            # Force single GPU
            self.config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.config['devices'] = [0] if torch.cuda.is_available() else []
        
        # Check available GPUs
        self.available_gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                self.available_gpus.append((i, gpu_name))
        
        # Create UI components
        self.create_menu_bar()
        self.create_status_bar()
        self.create_main_panel()
        
        # Bind events
        self.Bind(wx.EVT_CLOSE, self.on_close)
        
        # Load config if exists
        self.config_path = Path("sebulba_config.json")
        if self.config_path.exists():
            self.load_config()
        
    def create_menu_bar(self):
        menu_bar = wx.MenuBar()
        
        # File menu
        file_menu = wx.Menu()
        
        new_item = file_menu.Append(wx.ID_NEW, "&New Project", "Create a new training project")
        open_item = file_menu.Append(wx.ID_OPEN, "&Open Project", "Open an existing training project")
        save_item = file_menu.Append(wx.ID_SAVE, "&Save Project", "Save the current project")
        file_menu.AppendSeparator()
        export_item = file_menu.Append(wx.ID_ANY, "E&xport Model", "Export model for CodinGame submission")
        file_menu.AppendSeparator()
        exit_item = file_menu.Append(wx.ID_EXIT, "E&xit", "Exit the application")
        
        # Edit menu
        edit_menu = wx.Menu()
        preferences_item = edit_menu.Append(wx.ID_PREFERENCES, "&Preferences", "Edit application preferences")
        
        # Training menu
        training_menu = wx.Menu()
        start_training_item = training_menu.Append(wx.ID_ANY, "&Start Training", "Start the training process")
        stop_training_item = training_menu.Append(wx.ID_ANY, "S&top Training", "Stop the training process")
        training_menu.AppendSeparator()
        league_item = training_menu.Append(wx.ID_ANY, "&League Training", "Configure and start league training")
        
        # Help menu
        help_menu = wx.Menu()
        about_item = help_menu.Append(wx.ID_ABOUT, "&About", "About Sebulba Pod Trainer")
        
        # Add menus to menu bar
        menu_bar.Append(file_menu, "&File")
        menu_bar.Append(edit_menu, "&Edit")
        menu_bar.Append(training_menu, "&Training")
        menu_bar.Append(help_menu, "&Help")
        
        # Set the menu bar
        self.SetMenuBar(menu_bar)
        
        # Bind events
        self.Bind(wx.EVT_MENU, self.on_new_project, new_item)
        self.Bind(wx.EVT_MENU, self.on_open_project, open_item)
        self.Bind(wx.EVT_MENU, self.on_save_project, save_item)
        self.Bind(wx.EVT_MENU, self.on_export_model, export_item)
        self.Bind(wx.EVT_MENU, self.on_exit, exit_item)
        self.Bind(wx.EVT_MENU, self.on_preferences, preferences_item)
        self.Bind(wx.EVT_MENU, self.on_start_training, start_training_item)
        self.Bind(wx.EVT_MENU, self.on_stop_training, stop_training_item)
        self.Bind(wx.EVT_MENU, self.on_league_training, league_item)
        self.Bind(wx.EVT_MENU, self.on_about, about_item)
    
    def create_status_bar(self):
        self.status_bar = self.CreateStatusBar(2)
        self.status_bar.SetStatusWidths([-2, -1])
        self.status_bar.SetStatusText("Ready", 0)
        self.update_device_status()
    
    def update_device_status(self):
        device_text = f"Device: {self.config['device']}"
        if self.config['multi_gpu'] and len(self.config['devices']) > 0:
            device_text += f" (Multi-GPU: {len(self.config['devices'])} devices)"
        self.status_bar.SetStatusText(device_text, 1)
    
    def create_main_panel(self):
        # Create notebook for tabbed interface
        self.notebook = wx.Notebook(self)
        
        # Create tabs
        self.model_panel = ModelPanel(self.notebook, self)
        self.training_panel = TrainingPanel(self.notebook, self)
        self.visualization_panel = VisualizationPanel(self.notebook, self)
        self.league_panel = LeaguePanel(self.notebook, self)
        self.export_panel = ExportPanel(self.notebook, self)

        # Add tabs to notebook
        self.notebook.AddPage(self.model_panel, "Model Architecture")
        self.notebook.AddPage(self.training_panel, "Training Configuration")
        self.notebook.AddPage(self.visualization_panel, "Visualization")
        self.notebook.AddPage(self.league_panel, "League Training")
        self.notebook.AddPage(self.export_panel, "Export")
        
        # Create sizer for the notebook
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.notebook, 1, wx.EXPAND)
        
        # Set sizer
        self.SetSizer(sizer)
    
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Merge with default config to ensure all keys exist
            self.config.update(loaded_config)
            
            # Ensure network_config has proper structure
            if 'network_config' not in self.config:
                self.config['network_config'] = {
                    'observation_dim': 25,
                    'hidden_layers': [
                        {'type': 'Linear+ReLU', 'size': 32},
                        {'type': 'Linear+ReLU', 'size': 32}
                    ],
                    'policy_hidden_size': 32,
                    'value_hidden_size': 32,
                    'special_hidden_size': 32
                }
            
            # Ensure hidden_layers exists and is not empty
            if 'hidden_layers' not in self.config['network_config'] or len(self.config['network_config']['hidden_layers']) == 0:
                self.config['network_config']['hidden_layers'] = [
                    {'type': 'Linear+ReLU', 'size': 32},
                    {'type': 'Linear+ReLU', 'size': 32}
                ]
            
            # Ensure device settings are properly set
            if self.config.get('multi_gpu', False) and len(self.config.get('devices', [])) > 0:
                print(f"Loaded multi-GPU configuration with devices: {self.config['devices']}")
            else:
                # Force single GPU if multi_gpu is not enabled
                self.config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                self.config['devices'] = [0] if torch.cuda.is_available() else []
                
            print(f"Loaded config with network_config: {self.config['network_config']}")
            
            self.update_device_status()
            self.model_panel.update_from_config()
            self.training_panel.update_from_config()
            self.league_panel.update_from_config()
        except Exception as e:
            wx.MessageBox(f"Error loading configuration: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
    
    def save_config(self):
        try:
            print(f"Saving config with network_config: {self.config['network_config']}")
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            wx.MessageBox(f"Error saving configuration: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
    
    def on_new_project(self, event):
        dlg = wx.MessageDialog(self, 
                              "Create a new project? This will reset all current settings.",
                              "New Project",
                              wx.YES_NO | wx.ICON_QUESTION)
        result = dlg.ShowModal()
        dlg.Destroy()
        
        if result == wx.ID_YES:
            # Reset configuration to defaults
            self.config = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'multi_gpu': False,
                'devices': [],
                'batch_size': 64,
                'learning_rate': 3e-4,
                'save_dir': 'models',
                'use_mixed_precision': True,
                'use_optimized_env': True,  # Default to optimized environment
                'network_config': {
                    'observation_dim': 27,
                    'hidden_layers': [
                        {'type': 'Linear+ReLU', 'size': 32},
                        {'type': 'Linear+ReLU', 'size': 32}
                    ],
                    'policy_hidden_size': 32,
                    'value_hidden_size': 32,
                    'special_hidden_size': 32
                }
            }
            
            # Update UI
            self.update_device_status()
            self.model_panel.update_from_config()
            self.training_panel.update_from_config()
            self.league_panel.update_from_config()
            
            # Save new config
            self.save_config()
    
    def on_open_project(self, event):
        with wx.FileDialog(self, "Open Project Configuration", wildcard="JSON files (*.json)|*.json",
                          style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            
            # Load the selected file
            path = fileDialog.GetPath()
            try:
                with open(path, 'r') as f:
                    self.config = json.load(f)
                
                # Update UI
                self.update_device_status()
                self.model_panel.update_from_config()
                self.training_panel.update_from_config()
                self.league_panel.update_from_config()
                
                # Save path for future saves
                self.config_path = Path(path)
                
            except Exception as e:
                wx.MessageBox(f"Error loading project: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
    
    def on_save_project(self, event):
        with wx.FileDialog(self, "Save Project Configuration", wildcard="JSON files (*.json)|*.json",
                          style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            
            # Save to the selected file
            path = fileDialog.GetPath()
            try:
                with open(path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                
                # Update config path
                self.config_path = Path(path)
                
            except Exception as e:
                wx.MessageBox(f"Error saving project: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
    
    def on_export_model(self, event):
        with wx.DirDialog(self, "Select Model Directory", style=wx.DD_DEFAULT_STYLE) as dirDialog:
            if dirDialog.ShowModal() == wx.ID_CANCEL:
                return
            
            model_dir = dirDialog.GetPath()
            
            with wx.FileDialog(self, "Save Exported Model", wildcard="Python files (*.py)|*.py",
                              style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
                
                if fileDialog.ShowModal() == wx.ID_CANCEL:
                    return
                
                output_path = fileDialog.GetPath()
                
                try:
                    exporter = ModelExporter(model_dir, output_path)
                    exporter.export()
                    wx.MessageBox(f"Model successfully exported to {output_path}", "Export Complete", wx.OK | wx.ICON_INFORMATION)
                except Exception as e:
                    wx.MessageBox(f"Error exporting model: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
    
    def on_exit(self, event):
        self.Close()
    
    def on_preferences(self, event):
        dlg = PreferencesDialog(self)
        dlg.ShowModal()
        dlg.Destroy()
        self.update_device_status()
    
    def on_start_training(self, event, config_updated=False):
        # Get training configuration from panels
        print("Getting training configuration from panels...")
        
        # Debug: Print configuration before any updates
        print("Configuration BEFORE panel updates:", self.config)
        print(f"Network config BEFORE: {self.config.get('network_config', {})}")
        
        # Only update config if it hasn't been updated already
        if not config_updated:
            print("Updating config from model panel...")
            self.model_panel.update_config()
            print("After model panel update:", self.config['network_config'])
            
            print("Updating config from training panel...")
            self.training_panel.update_config()
            print("After training panel update:", self.config)
        else:
            print("Config already updated from training panel")
        
        # Ensure critical parameters exist
        if 'num_iterations' not in self.config:
            print("Adding missing num_iterations parameter")
            self.config['num_iterations'] = 1000
        
        if 'steps_per_iteration' not in self.config:
            print("Adding missing steps_per_iteration parameter")
            self.config['steps_per_iteration'] = 128
        
        if 'save_interval' not in self.config:
            print("Adding missing save_interval parameter")
            self.config['save_interval'] = 50
        
        # Validate network config
        network_config = self.config.get('network_config', {})
        print(f"Final network config being used: {network_config}")
        
        if 'hidden_layers' not in network_config or len(network_config['hidden_layers']) == 0:
            print("WARNING: No hidden layers found, adding default layers")
            network_config['hidden_layers'] = [
                {'type': 'Linear+ReLU', 'size': 32},
                {'type': 'Linear+ReLU', 'size': 32}
            ]
            self.config['network_config'] = network_config
        
        # Create save directory if it doesn't exist
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        # Switch to visualization tab if live monitoring is enabled
        self.notebook.SetSelection(2)  # Index 2 is the visualization tab

        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self.run_training)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        # Update UI
        self.status_bar.SetStatusText("Training...", 0)

    def run_training(self):
        print("Starting training...")
        try:
            # Make a deep copy of the config to avoid any reference issues
            training_config = deepcopy(self.config)
            
            print(f"Training config: {training_config}")
            print(f"Network config in run_training: {training_config.get('network_config', {})}")
            
            # Ensure critical parameters exist
            if 'num_iterations' not in training_config:
                print("WARNING: num_iterations not found, using default value of 1000")
                training_config['num_iterations'] = 1000
                
            if 'steps_per_iteration' not in training_config:
                print("WARNING: steps_per_iteration not found, using default value of 128")
                training_config['steps_per_iteration'] = 128
                
            if 'save_interval' not in training_config:
                print("WARNING: save_interval not found, using default value of 50")
                training_config['save_interval'] = 50
            
            # Get visualization panel for real-time updates
            visualization_panel = self.visualization_panel
            
            # Check if we should use parallel training
            print(f"Config in run_training: multi_gpu={training_config.get('multi_gpu')}, use_parallel={training_config.get('use_parallel')}")
        
            # Modified condition: use parallel training if use_parallel is True, regardless of multi_gpu setting
            if training_config.get('use_parallel', False):
                print("Using parallel training...")
                from ..training.trainer_parallel import ParallelPPOTrainer
                
                # Print debug information
                print("=== STARTING PARALLEL TRAINING ===")
                print(f"Configuration: {training_config}")
                
                # Create environment config for parallel training
                env_config = {
                    'batch_size': training_config['batch_size'],
                    # Don't specify device here, let each worker set it
                    'num_checkpoints': training_config.get('num_checkpoints', 3),
                    'laps': training_config.get('laps', 3)
                }
                
                print(f"Environment config: {env_config}")
                
                # Create trainer args with explicit parameters
                trainer_args = {}
                
                # Copy basic parameters
                trainer_args['learning_rate'] = training_config['learning_rate']
                trainer_args['batch_size'] = training_config['batch_size']
                
                # If multi_gpu is enabled, use the devices list, otherwise use a list with just the primary device
                if training_config.get('multi_gpu', False) and training_config.get('devices'):
                    trainer_args['devices'] = training_config['devices']
                else:
                    # Extract device index for single GPU case
                    device_str = training_config.get('device', 'cuda:0')
                    if device_str.startswith('cuda:'):
                        try:
                            device_idx = int(device_str.split(':')[1])
                            trainer_args['devices'] = [device_idx]
                        except (IndexError, ValueError):
                            trainer_args['devices'] = [0]  # Default to first GPU
                    else:
                        trainer_args['devices'] = [0]  # Default to first GPU if device string is unusual
                
                trainer_args['use_mixed_precision'] = training_config.get('use_mixed_precision', True)
                
                # CRITICAL: Explicitly copy training parameters
                trainer_args['num_iterations'] = int(training_config['num_iterations'])
                trainer_args['steps_per_iteration'] = int(training_config['steps_per_iteration'])
                trainer_args['save_interval'] = int(training_config['save_interval'])
                trainer_args['save_dir'] = training_config['save_dir']
                
                # Add PPO parameters if they exist
                if 'ppo_params' in training_config:
                    for key, value in training_config['ppo_params'].items():
                        trainer_args[key] = value

                # Add network configuration
                trainer_args['network_config'] = training_config.get('network_config', {})
                print(f"Network config being passed to parallel trainer: {trainer_args['network_config']}")

                print(f"Final trainer args: {trainer_args}")
                print(f"num_iterations: {trainer_args.get('num_iterations')}")
                print(f"steps_per_iteration: {trainer_args.get('steps_per_iteration')}")
                print(f"Using devices: {trainer_args['devices']}")
                print(f"network_config: {trainer_args['network_config']}")
                
                # Create parallel trainer
                print("Creating ParallelPPOTrainer...")
                trainer = ParallelPPOTrainer(
                    env_config=env_config,
                    trainer_args=trainer_args,
                    envs_per_gpu=training_config.get('envs_per_gpu', 8)
                )
                
                # Start parallel training with visualization
                print("Starting parallel training...")
                trainer.train(visualization_panel)
                print("Parallel training completed")

            else:
                print("Using standard (non-parallel) training...")
                # For multi-GPU but non-parallel case, we need special handling
                if training_config.get('multi_gpu', False) and len(training_config.get('devices', [])) > 1:
                    # Create an environment on the primary device only
                    primary_device = f"cuda:{training_config['devices'][0]}"
                    print(f"Creating environment on primary device: {primary_device}")
                    
                    # Check if we should use the optimized environment
                    if training_config.get('use_optimized_env', True):
                        print("Using optimized race environment")
                        env = OptimizedRaceEnvironment(
                            batch_size=training_config['batch_size'],
                            device=torch.device(primary_device)
                        )
                    else:
                        print("Using standard race environment")
                        env = RaceEnvironment(
                            batch_size=training_config['batch_size'],
                            device=torch.device(primary_device)
                        )
                else:
                    # Single GPU or CPU case
                    print(f"Creating environment on device: {training_config['device']}")
                    
                    # Check if we should use the optimized environment
                    if training_config.get('use_optimized_env', True):
                        print("Using optimized race environment")
                        env = OptimizedRaceEnvironment(
                            batch_size=training_config['batch_size'],
                            device=torch.device(training_config['device'])
                        )
                    else:
                        print("Using standard race environment")
                        env = RaceEnvironment(
                            batch_size=training_config['batch_size'],
                            device=torch.device(training_config['device'])
                        )

                # Create a custom network using the configuration
                network_config = training_config.get('network_config', {})
                print(f"Using network config for standard training: {network_config}")

                # Create trainer - use optimized trainer if using optimized environment
                if training_config.get('use_optimized_env', True):
                    print("Creating OptimizedPPOTrainer...")
                    trainer = OptimizedPPOTrainer(
                        env=env,
                        learning_rate=training_config['learning_rate'],
                        batch_size=training_config['batch_size'],
                        device=torch.device(training_config['device']),
                        multi_gpu=training_config.get('multi_gpu', False),
                        devices=training_config.get('devices', []),
                        use_mixed_precision=training_config.get('use_mixed_precision', True),
                        network_config=network_config  # Pass the network config
                    )
                else:
                    print("Creating standard PPOTrainer...")
                    trainer = PPOTrainer(
                        env=env,
                        learning_rate=training_config['learning_rate'],
                        batch_size=training_config['batch_size'],
                        device=torch.device(training_config['device']),
                        multi_gpu=training_config.get('multi_gpu', False),
                        devices=training_config.get('devices', []),
                        use_mixed_precision=training_config.get('use_mixed_precision', True),
                        network_config=network_config  # Pass the network config
                    )
                
                # Add PPO parameters if they exist
                if 'ppo_params' in training_config:
                    print(f"Adding PPO parameters: {training_config['ppo_params']}")
                    for key, value in training_config['ppo_params'].items():
                        setattr(trainer, key, value)
                
                # Start training with visualization panel
                print("Starting standard training...")
                print(f"Training with num_iterations={training_config['num_iterations']}, steps_per_iteration={training_config['steps_per_iteration']}, save_interval={training_config['save_interval']}")
                trainer.train(
                    num_iterations=training_config['num_iterations'],
                    steps_per_iteration=training_config['steps_per_iteration'],
                    save_interval=training_config['save_interval'],
                    save_dir=training_config['save_dir'],
                    visualization_panel=visualization_panel
                )
                print("Standard training completed")
            
            # Update UI when done
            wx.CallAfter(self.on_training_complete)
            
        except Exception as e:
            import traceback
            print(f"Exception in run_training: {str(e)}")
            print(traceback.format_exc())
            wx.CallAfter(self.on_training_error, str(e))    
    def on_training_complete(self):
        self.status_bar.SetStatusText("Training complete", 0)
        wx.MessageBox("Training completed successfully!", "Training Complete", wx.OK | wx.ICON_INFORMATION)
    
    def on_training_error(self, error_msg):
        self.status_bar.SetStatusText("Training failed", 0)
        wx.MessageBox(f"Training failed: {error_msg}", "Training Error", wx.OK | wx.ICON_ERROR)
    
    def on_stop_training(self, event):
        # This is a placeholder - we would need a way to signal the training process to stop
        wx.MessageBox("Training stop requested. The current iteration will complete before stopping.", 
                     "Stop Training", wx.OK | wx.ICON_INFORMATION)
    
    def on_league_training(self, event):
        # Switch to league tab
        self.notebook.SetSelection(3)
        
        # Start league training
        self.league_panel.start_league_training()
    
    def on_about(self, event):
        info = wx.adv.AboutDialogInfo()
        info.SetName("Sebulba Pod Trainer")
        info.SetVersion("1.0")
        info.SetDescription("A training application for pod racing AI using reinforcement learning")
        info.SetCopyright("(C) 2023")
        info.SetWebSite("https://github.com/yourusername/sebulba_pod_trainer")
        
        wx.adv.AboutBox(info)
    
    def on_close(self, event):
        dlg = wx.MessageDialog(self, 
                              "Do you want to save your configuration before exiting?",
                              "Save Configuration",
                              wx.YES_NO | wx.CANCEL | wx.ICON_QUESTION)
        result = dlg.ShowModal()
        dlg.Destroy()
        
        if result == wx.ID_YES:
            self.save_config()
            self.Destroy()
        elif result == wx.ID_NO:
            self.Destroy()
        else:  # wx.ID_CANCEL
            event.Veto()

def main():
    app = SebulbaPodTrainerApp()
    app.MainLoop()

if __name__ == "__main__":
    main()
