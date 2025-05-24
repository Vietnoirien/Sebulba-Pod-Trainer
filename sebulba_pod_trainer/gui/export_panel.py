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
        
        # Model file selection
        self.runner_label = wx.StaticText(self, label="Runner Model:")
        self.runner_choice = wx.Choice(self)
        self.blocker_label = wx.StaticText(self, label="Blocker Model:")
        self.blocker_choice = wx.Choice(self)
        
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
        self.precision_slider = wx.Slider(self, value=3, minValue=1, maxValue=5, 
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
            
            # Clear current choices
            self.runner_choice.Clear()
            self.blocker_choice.Clear()
            
            # Add model files to choices
            for model_file in model_files:
                self.runner_choice.Append(model_file.name)
                self.blocker_choice.Append(model_file.name)
            
            # Try to select appropriate defaults
            self.select_default_model(self.runner_choice, "pod0")
            self.select_default_model(self.blocker_choice, "pod1")
            
            # Update status
            self.status_text.SetLabel(f"Found {len(model_files)} model files in directory")
            
        except Exception as e:
            self.status_text.SetLabel(f"Error loading models: {str(e)}")
    
    def select_default_model(self, choice_ctrl, pod_name):
        """Try to select a default model based on pod name"""
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
    
    def update_preview(self):
        """Update the export preview"""
        # This would show a preview of the exported file size and structure
        model_dir = self.model_dir_text.GetValue()
        if not model_dir or not os.path.exists(model_dir):
            self.preview_text.SetValue("Please select a valid model directory first")
            return
        
        quantize = self.quantize_check.GetValue()
        precision = self.precision_slider.GetValue()
        
        # Calculate approximate file size based on settings
        if quantize:
            estimated_size = f"Estimated file size: ~{200 / precision:.1f} KB (with quantization at precision {precision})"
        else:
            estimated_size = "Estimated file size: ~600 KB (without quantization)"
        
        preview = f"""Export Configuration:
Model Directory: {model_dir}
Runner Model: {self.runner_choice.GetStringSelection() if self.runner_choice.GetSelection() != wx.NOT_FOUND else "Not selected"}
Blocker Model: {self.blocker_choice.GetStringSelection() if self.blocker_choice.GetSelection() != wx.NOT_FOUND else "Not selected"}
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
        
        # Get export settings
        quantize = self.quantize_check.GetValue()
        precision = self.precision_slider.GetValue()
        
        try:
            # Show busy cursor
            wx.BeginBusyCursor()
            
            # Import the model exporter
            from ..export.model_exporter import ModelExporter
            
            # Create and run the exporter
            exporter = ModelExporter(model_dir, output_path)
            exporter.export(quantize=quantize, precision=precision)
            
            # Show success message
            file_size = os.path.getsize(output_path) / 1024  # Size in KB
            wx.MessageBox(f"Model successfully exported to {output_path}\nFile size: {file_size:.2f} KB", 
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