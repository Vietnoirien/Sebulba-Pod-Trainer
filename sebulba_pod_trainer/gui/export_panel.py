import wx
import os
from pathlib import Path

class ExportPanel(wx.Panel):
    def __init__(self, parent, main_frame):
        super(ExportPanel, self).__init__(parent)
        self.main_frame = main_frame
        
        # Create main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create model selection section
        model_box = wx.StaticBox(self, label="Model Selection")
        model_sizer = wx.StaticBoxSizer(model_box, wx.VERTICAL)
        
        # Model directory selection
        dir_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.model_dir_label = wx.StaticText(self, label="Model Directory:")
        self.model_dir_text = wx.TextCtrl(self)
        self.browse_button = wx.Button(self, label="Browse...")
        
        dir_sizer.Add(self.model_dir_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        dir_sizer.Add(self.model_dir_text, 1, wx.EXPAND | wx.RIGHT, 5)
        dir_sizer.Add(self.browse_button, 0)
        
        model_sizer.Add(dir_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Model selection strategy (similar to visualization panel)
        strategy_sizer = wx.BoxSizer(wx.HORIZONTAL)
        strategy_label = wx.StaticText(self, label="Model Selection:")
        self.model_selection_choice = wx.Choice(self, choices=["Best Performance", "Latest Iteration", "Manual Select"])
        self.model_selection_choice.SetSelection(0)  # Default to best performance
        self.model_selection_choice.Bind(wx.EVT_CHOICE, self.on_model_selection_changed)
        
        strategy_sizer.Add(strategy_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        strategy_sizer.Add(self.model_selection_choice, 0)
        
        model_sizer.Add(strategy_sizer, 0, wx.ALL, 5)
        
        # Manual model selection (initially hidden)
        manual_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.manual_runner_label = wx.StaticText(self, label="Runner Model:")
        self.manual_runner_choice = wx.Choice(self, choices=[])
        self.manual_blocker_label = wx.StaticText(self, label="Blocker Model:")
        self.manual_blocker_choice = wx.Choice(self, choices=[])
        
        manual_sizer.Add(self.manual_runner_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        manual_sizer.Add(self.manual_runner_choice, 1, wx.EXPAND | wx.RIGHT, 5)
        manual_sizer.Add(self.manual_blocker_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        manual_sizer.Add(self.manual_blocker_choice, 1, wx.EXPAND)
        
        # Initially hide manual controls
        self.manual_runner_label.Hide()
        self.manual_runner_choice.Hide()
        self.manual_blocker_label.Hide()
        self.manual_blocker_choice.Hide()
        
        model_sizer.Add(manual_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Legacy model file selection (kept for backward compatibility, but hidden by default)
        self.runner_label = wx.StaticText(self, label="Runner Model:")
        self.runner_choice = wx.Choice(self)
        self.blocker_label = wx.StaticText(self, label="Blocker Model:")
        self.blocker_choice = wx.Choice(self)
        
        # Hide legacy controls
        self.runner_label.Hide()
        self.runner_choice.Hide()
        self.blocker_label.Hide()
        self.blocker_choice.Hide()
        
        model_sizer.Add(self.runner_label, 0, wx.ALL, 5)
        model_sizer.Add(self.runner_choice, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        model_sizer.Add(self.blocker_label, 0, wx.ALL, 5)
        model_sizer.Add(self.blocker_choice, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        
        # Add model sizer to main sizer
        main_sizer.Add(model_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Create export options section
        export_box = wx.StaticBox(self, label="Export Options")
        export_sizer = wx.StaticBoxSizer(export_box, wx.VERTICAL)
        
        # Quantization option
        self.quantize_check = wx.CheckBox(self, label="Quantize Weights (reduces file size)")
        self.quantize_check.SetValue(True)
        export_sizer.Add(self.quantize_check, 0, wx.ALL, 5)
        
        # Precision slider
        precision_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.precision_label = wx.StaticText(self, label="Precision:")
        self.precision_slider = wx.Slider(self, value=8, minValue=1, maxValue=16, 
                                         style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.precision_value = wx.StaticText(self, label="3 decimal places")
        
        precision_sizer.Add(self.precision_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        precision_sizer.Add(self.precision_slider, 1, wx.EXPAND | wx.RIGHT, 5)
        precision_sizer.Add(self.precision_value, 0, wx.ALIGN_CENTER_VERTICAL)
        
        export_sizer.Add(precision_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Output file selection
        output_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.output_label = wx.StaticText(self, label="Output File:")
        self.output_text = wx.TextCtrl(self, value="codingame_submission.py")
        self.output_browse = wx.Button(self, label="Browse...")
        
        output_sizer.Add(self.output_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        output_sizer.Add(self.output_text, 1, wx.EXPAND | wx.RIGHT, 5)
        output_sizer.Add(self.output_browse, 0)
        
        export_sizer.Add(output_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Add export sizer to main sizer
        main_sizer.Add(export_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Export button
        self.export_button = wx.Button(self, label="Export Model")
        main_sizer.Add(self.export_button, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        # Status information
        self.status_text = wx.StaticText(self, label="")
        main_sizer.Add(self.status_text, 0, wx.EXPAND | wx.ALL, 10)
        
        # Preview section
        preview_box = wx.StaticBox(self, label="Export Preview")
        preview_sizer = wx.StaticBoxSizer(preview_box, wx.VERTICAL)
        
        self.preview_text = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        self.preview_text.SetMinSize((400, 200))
        preview_sizer.Add(self.preview_text, 1, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(preview_sizer, 1, wx.EXPAND | wx.ALL, 10)
        
        # Set the main sizer
        self.SetSizer(main_sizer)
        
        # Bind events
        self.browse_button.Bind(wx.EVT_BUTTON, self.on_browse_model_dir)
        self.output_browse.Bind(wx.EVT_BUTTON, self.on_browse_output)
        self.export_button.Bind(wx.EVT_BUTTON, self.on_export)
        self.precision_slider.Bind(wx.EVT_SLIDER, self.on_precision_change)
        self.quantize_check.Bind(wx.EVT_CHECKBOX, self.on_quantize_change)
        
        # Initialize with default values
        self.update_precision_label()
    
    def on_model_selection_changed(self, event):
        """Handle model selection strategy change"""
        selection = self.model_selection_choice.GetSelection()
        if selection != wx.NOT_FOUND:
            strategy = self.model_selection_choice.GetString(selection)
            
            if strategy == "Manual Select":
                # Show manual controls and populate model list
                self.manual_runner_label.Show()
                self.manual_runner_choice.Show()
                self.manual_blocker_label.Show()
                self.manual_blocker_choice.Show()
                self.populate_manual_model_lists()
            else:
                # Hide manual controls
                self.manual_runner_label.Hide()
                self.manual_runner_choice.Hide()
                self.manual_blocker_label.Hide()
                self.manual_blocker_choice.Hide()
            
            # Refresh layout and preview
            self.Layout()
            self.update_preview()
    
    def populate_manual_model_lists(self):
        """Populate the manual model selection dropdowns"""
        model_dir = self.model_dir_text.GetValue()
        if not model_dir:
            return
            
        model_path = Path(model_dir)
        if not model_path.exists():
            return
        
        # Clear existing choices
        self.manual_runner_choice.Clear()
        self.manual_blocker_choice.Clear()
        
        # Find all model files for each pod
        for pod_name in ["player0_pod0", "player0_pod1"]:
            model_files = []
            
            # Standard files
            standard_file = model_path / f"{pod_name}.pt"
            if standard_file.exists():
                model_files.append(f"Standard: {pod_name}.pt")
            
            # GPU-specific files
            gpu_files = list(model_path.glob(f"{pod_name}_gpu*.pt"))
            for gpu_file in sorted(gpu_files):
                model_files.append(f"GPU: {gpu_file.name}")
            
            # Iteration files
            iter_files = list(model_path.glob(f"{pod_name}_gpu*_iter*.pt"))
            for iter_file in sorted(iter_files, key=lambda x: self.extract_iteration_number(x.name)):
                iteration = self.extract_iteration_number(iter_file.name)
                model_files.append(f"Iter {iteration}: {iter_file.name}")
            
            # Add to appropriate dropdown
            choice_ctrl = self.manual_runner_choice if pod_name == "player0_pod0" else self.manual_blocker_choice
            for model_file in model_files:
                choice_ctrl.Append(model_file)
            
            if model_files:
                choice_ctrl.SetSelection(0)
    
    def extract_iteration_number(self, filename):
        """Extract iteration number from filename"""
        try:
            if '_iter' in filename:
                return int(filename.split('_iter')[-1].split('.')[0])
        except (ValueError, IndexError):
            pass
        return 0
    
    def find_best_model_file(self, pod_name, model_path):
        """Find the best model file based on selection strategy (copied from visualization panel)"""
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
        if pod_name == "player0_pod0":
            choice_ctrl = self.manual_runner_choice
        elif pod_name == "player0_pod1":
            choice_ctrl = self.manual_blocker_choice
        else:
            return None
            
        selection = choice_ctrl.GetSelection()
        if selection == wx.NOT_FOUND:
            return None
            
        selected_text = choice_ctrl.GetString(selection)
        
        # Extract filename from the display text
        if ": " in selected_text:
            filename = selected_text.split(": ", 1)[1]
            model_file = model_path / filename
            
            if model_file.exists():
                return model_file
        
        # Fallback to best performance if manual selection doesn't work
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
                    return model_file
        
        # Alternative: find iteration with best average reward over multiple runs
        if iteration_rewards:
            avg_rewards = {iter_num: sum(rewards)/len(rewards) 
                          for iter_num, rewards in iteration_rewards.items()}
            best_avg_iteration = max(avg_rewards.keys(), key=lambda k: avg_rewards[k])
            
            for model_file in model_files:
                if f'_iter{best_avg_iteration}.' in str(model_file):
                    return model_file
        
        return None
    
    def on_browse_model_dir(self, event):
        """Handle browsing for model directory"""
        with wx.DirDialog(self, "Select Model Directory", style=wx.DD_DEFAULT_STYLE) as dirDialog:
            if dirDialog.ShowModal() == wx.ID_CANCEL:
                return
            
            model_dir = dirDialog.GetPath()
            self.model_dir_text.SetValue(model_dir)
            self.update_model_choices(model_dir)
    
    def update_model_choices(self, model_dir):
        """Update the model choice dropdowns based on available models"""
        try:
            model_path = Path(model_dir)
            model_files = list(model_path.glob("*.pt"))
            
            # Clear current choices (legacy dropdowns)
            self.runner_choice.Clear()
            self.blocker_choice.Clear()
            
            # Add model files to choices (legacy)
            for model_file in model_files:
                self.runner_choice.Append(model_file.name)
                self.blocker_choice.Append(model_file.name)
            
            # Try to select appropriate defaults (legacy)
            self.select_default_model(self.runner_choice, "pod0")
            self.select_default_model(self.blocker_choice, "pod1")
            
            # Update manual model lists if in manual mode
            if self.model_selection_choice.GetSelection() == 2:  # Manual Select
                self.populate_manual_model_lists()
            
            # Update status
            self.status_text.SetLabel(f"Found {len(model_files)} model files in directory")
            
            # Update preview
            self.update_preview()
            
        except Exception as e:
            self.status_text.SetLabel(f"Error loading models: {str(e)}")
    
    def select_default_model(self, choice_ctrl, pod_name):
        """Try to select a default model based on pod name (legacy function)"""
        # Try to find a model with the pod name
        for i in range(choice_ctrl.GetCount()):
            item = choice_ctrl.GetString(i)
            if pod_name in item:
                choice_ctrl.SetSelection(i)
                return
        
        # If no matching model found, select the first one if available
        if choice_ctrl.GetCount() > 0:
            choice_ctrl.SetSelection(0)
    
    def on_browse_output(self, event):
        """Handle browsing for output file"""
        with wx.FileDialog(self, "Save Exported Model", wildcard="Python files (*.py)|*.py",
                          style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            
            output_path = fileDialog.GetPath()
            self.output_text.SetValue(output_path)
    
    def on_precision_change(self, event):
        """Handle precision slider change"""
        self.update_precision_label()
        self.update_preview()
    
    def on_quantize_change(self, event):
        """Handle quantize checkbox change"""
        # Enable/disable precision controls based on quantize checkbox
        is_quantize = self.quantize_check.GetValue()
        self.precision_label.Enable(is_quantize)
        self.precision_slider.Enable(is_quantize)
        self.precision_value.Enable(is_quantize)
        
        self.update_preview()
    
    def update_precision_label(self):
        """Update the precision value label"""
        precision = self.precision_slider.GetValue()
        self.precision_value.SetLabel(f"{precision} decimal places")
    
    def get_selected_model_info(self):
        """Get information about the selected models"""
        model_dir = self.model_dir_text.GetValue()
        if not model_dir or not os.path.exists(model_dir):
            return None, None, "Please select a valid model directory first"
        
        model_path = Path(model_dir)
        
        # Get runner model
        runner_model = self.find_best_model_file("player0_pod0", model_path)
        runner_name = runner_model.name if runner_model else "Not found"
        
        # Get blocker model
        blocker_model = self.find_best_model_file("player0_pod1", model_path)
        blocker_name = blocker_model.name if blocker_model else "Not found"
        
        return runner_name, blocker_name, None
    
    def update_preview(self):
        """Update the export preview"""
        runner_name, blocker_name, error = self.get_selected_model_info()
        
        if error:
            self.preview_text.SetValue(error)
            return
        
        model_dir = self.model_dir_text.GetValue()
        quantize = self.quantize_check.GetValue()
        precision = self.precision_slider.GetValue()
        strategy = self.model_selection_choice.GetStringSelection()
        
        # Calculate approximate file size based on settings
        if quantize:
            estimated_size = f"Estimated file size: ~{200 / precision:.1f} KB (with quantization at precision {precision})"
        else:
            estimated_size = "Estimated file size: ~600 KB (without quantization)"
        
        preview = f"""Export Configuration:
Model Directory: {model_dir}
Selection Strategy: {strategy}
Runner Model: {runner_name}
Blocker Model: {blocker_name}
Output File: {self.output_text.GetValue()}
Quantization: {"Enabled" if quantize else "Disabled"}
Precision: {precision} decimal places

{estimated_size}

The exported model will include:
- Neural network weights (quantized to {precision} decimal places)
- Activation functions (ReLU, tanh, sigmoid)
- Observation processing code
- Game loop and decision logic
"""
        self.preview_text.SetValue(preview)
    
    def on_export(self, event):
        """Handle export button click"""
        # Validate inputs
        model_dir = self.model_dir_text.GetValue()
        if not model_dir or not os.path.exists(model_dir):
            wx.MessageBox("Please select a valid model directory", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        output_path = self.output_text.GetValue()
        if not output_path:
            wx.MessageBox("Please specify an output file path", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Verify that we can find the required models
        model_path = Path(model_dir)
        runner_model = self.find_best_model_file("player0_pod0", model_path)
        blocker_model = self.find_best_model_file("player0_pod1", model_path)
        
        if not runner_model or not blocker_model:
            wx.MessageBox(f"Could not find required model files:\n"
                         f"Runner (pod0): {'Found' if runner_model else 'Not found'}\n"
                         f"Blocker (pod1): {'Found' if blocker_model else 'Not found'}", 
                         "Model Files Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Get export settings
        quantize = self.quantize_check.GetValue()
        precision = self.precision_slider.GetValue()
        
        try:
            # Show busy cursor
            wx.BeginBusyCursor()
            
            # Import the model exporter
            from ..export.model_exporter import ModelExporter
            
            # Get network configuration from main frame if available
            network_config = None
            if hasattr(self.main_frame, 'config') and 'network_config' in self.main_frame.config:
                network_config = self.main_frame.config['network_config']
            
            # Create and run the exporter with specific model files
            exporter = ModelExporter(model_dir, output_path, network_config=network_config)
            
            # Set specific model files if using advanced selection
            if hasattr(exporter, 'set_model_files'):
                exporter.set_model_files({
                    'player0_pod0': runner_model,
                    'player0_pod1': blocker_model
                })
            
            exporter.export(quantize=quantize, precision=precision)
            
            # Show success message with model info
            file_size = os.path.getsize(output_path) / 1024  # Size in KB
            wx.MessageBox(f"Model successfully exported to {output_path}\n"
                         f"File size: {file_size:.2f} KB\n"
                         f"Runner model: {runner_model.name}\n"
                         f"Blocker model: {blocker_model.name}", 
                         "Export Complete", wx.OK | wx.ICON_INFORMATION)
            
        except Exception as e:
            wx.MessageBox(f"Error exporting model: {str(e)}", "Export Error", wx.OK | wx.ICON_ERROR)
        
        finally:
            # Restore normal cursor
            if wx.IsBusy():
                wx.EndBusyCursor()
    
    def update_from_config(self):
        """Update panel from main frame config"""
        config = self.main_frame.config
        
        # Set model directory if available
        if 'save_dir' in config:
            self.model_dir_text.SetValue(config['save_dir'])
            self.update_model_choices(config['save_dir'])
        
        # Update preview
        self.update_preview()
