import wx
import wx.grid
import torch
import numpy as np
from typing import Dict, List, Any

class ModelPanel(wx.Panel):
    def __init__(self, parent, main_frame):
        super(ModelPanel, self).__init__(parent)
        
        self.main_frame = main_frame
        self.config = main_frame.config
        self.visualization_data = None  # Store visualization data
        
        # Create UI components
        self.create_ui()
        
        # Update UI from config
        self.update_from_config()
    
    def create_ui(self):
        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Network architecture section
        arch_box = wx.StaticBox(self, label="Network Architecture")
        arch_sizer = wx.StaticBoxSizer(arch_box, wx.VERTICAL)
        
        # Input dimension
        input_sizer = wx.BoxSizer(wx.HORIZONTAL)
        input_label = wx.StaticText(self, label="Observation Dimension:")
        self.input_dim_ctrl = wx.SpinCtrl(self, min=1, max=100, initial=56)  # Updated for new observation format
        input_sizer.Add(input_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        input_sizer.Add(self.input_dim_ctrl, 0, wx.EXPAND)
        arch_sizer.Add(input_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Hidden layers
        hidden_sizer = wx.BoxSizer(wx.VERTICAL)
        hidden_label = wx.StaticText(self, label="Hidden Layers:")
        hidden_sizer.Add(hidden_label, 0, wx.BOTTOM, 5)
        
        # Grid for hidden layers - start with at least 2 rows
        self.layers_grid = wx.grid.Grid(self)
        self.layers_grid.CreateGrid(2, 2)  # Start with 2 hidden layers
        self.layers_grid.SetColLabelValue(0, "Layer Type")
        self.layers_grid.SetColLabelValue(1, "Size")
        
        # Set default values
        self.layers_grid.SetCellValue(0, 0, "Linear+ReLU")
        self.layers_grid.SetCellValue(0, 1, "24")  # Reduced from 32
        self.layers_grid.SetCellValue(1, 0, "Linear+ReLU")
        self.layers_grid.SetCellValue(1, 1, "16")  # Reduced from 32
        
        # Create dropdown editor for layer type
        layer_types = ["Linear", "Linear+ReLU", "Linear+Tanh", "Linear+Sigmoid"]
        self.layers_grid.SetCellEditor(0, 0, wx.grid.GridCellChoiceEditor(layer_types))
        self.layers_grid.SetCellEditor(1, 0, wx.grid.GridCellChoiceEditor(layer_types))
        
        # Set column widths
        self.layers_grid.SetColSize(0, 150)
        self.layers_grid.SetColSize(1, 100)
        
        hidden_sizer.Add(self.layers_grid, 1, wx.EXPAND | wx.BOTTOM, 5)
        
        # Buttons for adding/removing layers
        layer_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.add_layer_btn = wx.Button(self, label="Add Layer")
        self.remove_layer_btn = wx.Button(self, label="Remove Layer")
        layer_btn_sizer.Add(self.add_layer_btn, 0, wx.RIGHT, 5)
        layer_btn_sizer.Add(self.remove_layer_btn, 0)
        hidden_sizer.Add(layer_btn_sizer, 0, wx.ALIGN_RIGHT)
        
        arch_sizer.Add(hidden_sizer, 1, wx.EXPAND | wx.ALL, 5)
        
        # Output section
        output_box = wx.StaticBox(self, label="Output Configuration")
        output_sizer = wx.StaticBoxSizer(output_box, wx.VERTICAL)
        
        # Policy head
        policy_sizer = wx.BoxSizer(wx.HORIZONTAL)
        policy_label = wx.StaticText(self, label="Policy Head Hidden Size:")
        self.policy_size_ctrl = wx.SpinCtrl(self, min=8, max=128, initial=12)  # Reduced from 16
        policy_sizer.Add(policy_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        policy_sizer.Add(self.policy_size_ctrl, 0, wx.EXPAND)
        output_sizer.Add(policy_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Value head
        value_sizer = wx.BoxSizer(wx.HORIZONTAL)
        value_label = wx.StaticText(self, label="Value Head Hidden Size:")
        self.value_size_ctrl = wx.SpinCtrl(self, min=8, max=128, initial=12)  # Reduced from 16
        value_sizer.Add(value_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        value_sizer.Add(self.value_size_ctrl, 0, wx.EXPAND)
        output_sizer.Add(value_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Special action head
        special_sizer = wx.BoxSizer(wx.HORIZONTAL)
        special_label = wx.StaticText(self, label="Special Action Head Hidden Size:")
        self.special_size_ctrl = wx.SpinCtrl(self, min=8, max=128, initial=12)  # Reduced from 16
        special_sizer.Add(special_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        special_sizer.Add(self.special_size_ctrl, 0, wx.EXPAND)
        output_sizer.Add(special_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Export options
        export_box = wx.StaticBox(self, label="Export Configuration")
        export_sizer = wx.StaticBoxSizer(export_box, wx.VERTICAL)
        
        # Quantization precision
        precision_sizer = wx.BoxSizer(wx.HORIZONTAL)
        precision_label = wx.StaticText(self, label="Weight Precision (decimal places):")
        self.precision_ctrl = wx.SpinCtrl(self, min=1, max=16, initial=16)  # Reduced from 3
        precision_sizer.Add(precision_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        precision_sizer.Add(self.precision_ctrl, 0, wx.EXPAND)
        export_sizer.Add(precision_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Add sections to main sizer
        main_sizer.Add(arch_sizer, 1, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(output_sizer, 0, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(export_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Visualization section
        viz_box = wx.StaticBox(self, label="Network Visualization")
        viz_sizer = wx.StaticBoxSizer(viz_box, wx.VERTICAL)
        
        self.viz_btn = wx.Button(self, label="Visualize Network")
        viz_sizer.Add(self.viz_btn, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        
        # Add visualization panel (placeholder for network diagram)
        self.viz_panel = VisualizationPanel(self)
        self.viz_panel.SetBackgroundColour(wx.Colour(240, 240, 240))
        self.viz_panel.SetMinSize((-1, 200))
        viz_sizer.Add(self.viz_panel, 1, wx.EXPAND | wx.ALL, 5)
        
        # Add model size estimation
        size_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.size_label = wx.StaticText(self, label="Estimated model size: 0 parameters, 0 KB")
        size_sizer.Add(self.size_label, 0, wx.ALIGN_CENTER_VERTICAL)
        viz_sizer.Add(size_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(viz_sizer, 1, wx.EXPAND | wx.ALL, 10)
        
        # Apply button
        self.apply_btn = wx.Button(self, label="Apply Changes")
        main_sizer.Add(self.apply_btn, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        
        # Set sizer
        self.SetSizer(main_sizer)
        
        # Bind events
        self.add_layer_btn.Bind(wx.EVT_BUTTON, self.on_add_layer)
        self.remove_layer_btn.Bind(wx.EVT_BUTTON, self.on_remove_layer)
        self.viz_btn.Bind(wx.EVT_BUTTON, self.on_visualize)
        self.apply_btn.Bind(wx.EVT_BUTTON, self.on_apply)
        
        # Bind events for updating size estimation
        self.layers_grid.Bind(wx.grid.EVT_GRID_CELL_CHANGED, self.update_size_estimation)
        self.input_dim_ctrl.Bind(wx.EVT_SPINCTRL, self.update_size_estimation)
        self.policy_size_ctrl.Bind(wx.EVT_SPINCTRL, self.update_size_estimation)
        self.value_size_ctrl.Bind(wx.EVT_SPINCTRL, self.update_size_estimation)
        self.special_size_ctrl.Bind(wx.EVT_SPINCTRL, self.update_size_estimation)
        self.precision_ctrl.Bind(wx.EVT_SPINCTRL, self.update_size_estimation)
    
    def update_from_config(self):
        """Update UI elements from the current configuration"""
        network_config = self.config.get('network_config', {})
        
        # Update observation dimension
        self.input_dim_ctrl.SetValue(network_config.get('observation_dim', 56))
        
        # Update hidden layers
        hidden_layers = network_config.get('hidden_layers', [
            {'type': 'Linear+ReLU', 'size': 24},
            {'type': 'Linear+ReLU', 'size': 16}
        ])
        
        # Ensure we have at least one hidden layer
        if len(hidden_layers) == 0:
            hidden_layers = [{'type': 'Linear+ReLU', 'size': 24}]
        
        # Resize grid if needed
        current_rows = self.layers_grid.GetNumberRows()
        target_rows = len(hidden_layers)
        
        if current_rows < target_rows:
            self.layers_grid.AppendRows(target_rows - current_rows)
        elif current_rows > target_rows:
            self.layers_grid.DeleteRows(0, current_rows - target_rows)
        
        # Update grid values
        for i, layer in enumerate(hidden_layers):
            self.layers_grid.SetCellValue(i, 0, layer.get('type', 'Linear+ReLU'))
            self.layers_grid.SetCellValue(i, 1, str(layer.get('size', 24)))
            
            # Set cell editor for layer type
            layer_types = ["Linear", "Linear+ReLU", "Linear+Tanh", "Linear+Sigmoid"]
            self.layers_grid.SetCellEditor(i, 0, wx.grid.GridCellChoiceEditor(layer_types))
        
        # Update head sizes
        self.policy_size_ctrl.SetValue(network_config.get('policy_hidden_size', 12))
        self.value_size_ctrl.SetValue(network_config.get('value_hidden_size', 12))
        self.special_size_ctrl.SetValue(network_config.get('special_hidden_size', 12))
        
        # Update export settings
        export_config = self.config.get('export_config', {})
        self.precision_ctrl.SetValue(export_config.get('precision', 16))
        
        # Update size estimation
        self.update_size_estimation()
    
    def update_config(self):
        """Update configuration from UI elements"""
        # Get observation dimension
        observation_dim = self.input_dim_ctrl.GetValue()
        
        # Get hidden layers
        hidden_layers = []
        num_rows = self.layers_grid.GetNumberRows()
        
        for i in range(num_rows):
            layer_type = self.layers_grid.GetCellValue(i, 0)
            try:
                layer_size = int(self.layers_grid.GetCellValue(i, 1))
            except ValueError:
                layer_size = 32  # Default if invalid
            
            hidden_layers.append({
                'type': layer_type,
                'size': layer_size
            })
        
        # Ensure we have at least one hidden layer
        if len(hidden_layers) == 0:
            hidden_layers = [{'type': 'Linear+ReLU', 'size': 24}]
        
        # Get head sizes
        policy_hidden_size = self.policy_size_ctrl.GetValue()
        value_hidden_size = self.value_size_ctrl.GetValue()
        special_hidden_size = self.special_size_ctrl.GetValue()
        
        # Get export settings
        precision = self.precision_ctrl.GetValue()
        
        # Update config
        self.config['network_config'] = {
            'observation_dim': observation_dim,
            'hidden_layers': hidden_layers,
            'policy_hidden_size': policy_hidden_size,
            'value_hidden_size': value_hidden_size,
            'special_hidden_size': special_hidden_size
        }
        
        self.config['export_config'] = {
            'precision': precision
        }
        
        print(f"Updated network config: {self.config['network_config']}")  # Debug print
    
    def calculate_model_size(self):
        """Calculate the number of parameters in the model and estimated file size"""
        observation_dim = self.input_dim_ctrl.GetValue()
        
        # Get hidden layers
        hidden_layers = []
        num_rows = self.layers_grid.GetNumberRows()
        
        for i in range(num_rows):
            try:
                layer_size = int(self.layers_grid.GetCellValue(i, 1))
            except ValueError:
                layer_size = 32  # Default if invalid
            
            hidden_layers.append(layer_size)
        
        # Ensure we have at least one hidden layer
        if len(hidden_layers) == 0:
            hidden_layers = [32]
        
        # Get head sizes
        policy_hidden_size = self.policy_size_ctrl.GetValue()
        value_hidden_size = self.value_size_ctrl.GetValue()
        special_hidden_size = self.special_size_ctrl.GetValue()
        
        # Calculate parameters
        total_params = 0
        
        # Encoder layers
        input_size = observation_dim
        for layer_size in hidden_layers:
            # Weights + biases
            total_params += (input_size * layer_size) + layer_size
            input_size = layer_size
        
        # Last hidden layer size
        last_hidden_size = hidden_layers[-1] if hidden_layers else observation_dim
        
        # Policy head
        total_params += (last_hidden_size * policy_hidden_size) + policy_hidden_size  # First layer
        total_params += (policy_hidden_size * 2) + 2  # Output layer (2 outputs)
        
        # Value head
        total_params += (last_hidden_size * value_hidden_size) + value_hidden_size  # First layer
        total_params += (value_hidden_size * 1) + 1  # Output layer (1 output)
        
        # Special action head
        total_params += (last_hidden_size * special_hidden_size) + special_hidden_size  # First layer
        total_params += (special_hidden_size * 2) + 2  # Output layer (2 outputs)
        
        # Estimate file size (rough approximation)
        precision = self.precision_ctrl.GetValue()
        chars_per_param = precision + 2  # Decimal point, sign, and digits
        estimated_size = total_params * chars_per_param
        
        # Add overhead for code
        code_overhead = 8200  # Approximate size of the code without weights
        estimated_size += code_overhead
        
        return total_params, estimated_size
    
    def update_size_estimation(self, event=None):
        """Update the size estimation label"""
        try:
            total_params, estimated_size = self.calculate_model_size()
            
            # Update the label
            self.size_label.SetLabel(
                f"Estimated model size: {total_params:,} parameters, {estimated_size/1000:.2f} KB"
            )
            
            # Highlight in red if over the limit
            if estimated_size > 100000:
                self.size_label.SetForegroundColour(wx.Colour(255, 0, 0))
            else:
                self.size_label.SetForegroundColour(wx.Colour(0, 0, 0))
                
        except Exception as e:
            self.size_label.SetLabel(f"Error estimating size: {str(e)}")
            self.size_label.SetForegroundColour(wx.Colour(255, 0, 0))
    
    def on_add_layer(self, event):
        """Add a new hidden layer to the grid"""
        # Add a new row
        self.layers_grid.AppendRows(1)
        current_rows = self.layers_grid.GetNumberRows()
        
        # Set default values for the new row
        self.layers_grid.SetCellValue(current_rows - 1, 0, "Linear+ReLU")
        self.layers_grid.SetCellValue(current_rows - 1, 1, "32")
        
        # Set cell editor
        layer_types = ["Linear", "Linear+ReLU", "Linear+Tanh", "Linear+Sigmoid"]
        self.layers_grid.SetCellEditor(current_rows - 1, 0, wx.grid.GridCellChoiceEditor(layer_types))
        
        # Update size estimation
        self.update_size_estimation()
    
    def on_remove_layer(self, event):
        """Remove the last hidden layer from the grid"""
        current_rows = self.layers_grid.GetNumberRows()
        if current_rows > 1:  # Keep at least one hidden layer
            self.layers_grid.DeleteRows(current_rows - 1, 1)
            
            # Update size estimation
            self.update_size_estimation()
    
    def on_visualize(self, event):
        """Visualize the network architecture"""
        try:
            # Update config from UI
            self.update_config()
            
            # Prepare visualization data
            network_config = self.config['network_config']
            
            # Send data to visualization panel
            self.viz_panel.set_network_config(network_config)
            
            # Refresh the panel to trigger a redraw
            self.viz_panel.Refresh()
            
        except Exception as e:
            wx.MessageBox(f"Error visualizing network: {str(e)}", "Visualization Error", wx.OK | wx.ICON_ERROR)
    
    def on_apply(self, event):
        """Apply changes to the network configuration"""
        self.update_config()
        
        # Check if the model is likely to fit within the character limit
        _, estimated_size = self.calculate_model_size()
        if estimated_size > 100000:
            result = wx.MessageBox(
                f"The estimated model size ({estimated_size/1000:.2f} KB) exceeds the 100,000 character limit for CodinGame. "
                "The model may not export correctly. Do you want to continue?",
                "Size Warning",
                wx.YES_NO | wx.ICON_WARNING
            )
            if result != wx.YES:
                return
        
        wx.MessageBox("Network configuration updated", "Configuration Updated", wx.OK | wx.ICON_INFORMATION)


class VisualizationPanel(wx.Panel):
    """Panel for visualizing the network architecture"""
    
    def __init__(self, parent):
        super(VisualizationPanel, self).__init__(parent)
        self.network_config = None
        
        # Bind the paint event
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
    
    def set_network_config(self, config):
        """Set the network configuration to visualize"""
        self.network_config = config
    
    def on_size(self, event):
        """Handle resize events"""
        self.Refresh()  # Redraw when resized
        event.Skip()
    
    def on_paint(self, event):
        """Paint event handler to draw the network visualization"""
        if not self.network_config:
            return
        
        # Create a paint DC
        dc = wx.PaintDC(self)
        dc.Clear()
        
        # Get panel dimensions
        width, height = self.GetSize()
        
        # Draw network layers
        hidden_layers = self.network_config['hidden_layers']
        
        # Calculate positions
        num_layers = len(hidden_layers) + 2  # Input + hidden + output
        layer_width = width / num_layers
        
        # Draw input layer
        input_dim = self.network_config['observation_dim']
        self._draw_layer(dc, 0, layer_width, height, "Input", input_dim)
        
        # Draw hidden layers
        for i, layer in enumerate(hidden_layers):
            layer_type = layer['type'].split('+')[0]
            layer_size = layer['size']
            self._draw_layer(dc, i+1, layer_width, height, layer_type, layer_size)
            
            # Draw connection to previous layer
            prev_x = (i * layer_width) + (layer_width / 2)
            curr_x = ((i+1) * layer_width) + (layer_width / 2)
            dc.SetPen(wx.Pen(wx.Colour(100, 100, 100), 1))
            dc.DrawLine(int(prev_x), int(height/2), int(curr_x), int(height/2))
        
        # Draw output layer
        self._draw_layer(dc, num_layers-1, layer_width, height, "Output", 2)
        
        # Draw connection to last hidden layer
        prev_x = ((num_layers-2) * layer_width) + (layer_width / 2)
        curr_x = ((num_layers-1) * layer_width) + (layer_width / 2)
        dc.SetPen(wx.Pen(wx.Colour(100, 100, 100), 1))
        dc.DrawLine(int(prev_x), int(height/2), int(curr_x), int(height/2))
        
        # Draw heads
        head_y = height * 0.8
        dc.SetTextForeground(wx.Colour(0, 0, 0))
        dc.DrawText("Policy Head", 10, int(head_y))
        dc.DrawText("Value Head", 10, int(head_y + 20))
        dc.DrawText("Special Action Head", 10, int(head_y + 40))
        
        # Draw head sizes
        policy_size = self.network_config.get('policy_hidden_size', 32)
        value_size = self.network_config.get('value_hidden_size', 32)
        special_size = self.network_config.get('special_hidden_size', 32)
        
        text_width = dc.GetTextExtent("Special Action Head")[0]
        dc.DrawText(f"({policy_size})", text_width + 20, int(head_y))
        dc.DrawText(f"({value_size})", text_width + 20, int(head_y + 20))
        dc.DrawText(f"({special_size})", text_width + 20, int(head_y + 40))
        
        # Draw parameter count
        total_params = self._calculate_parameter_count()
        dc.DrawText(f"Total parameters: {total_params:,}", 10, int(head_y + 70))
    
    def _calculate_parameter_count(self):
        """Calculate the total number of parameters in the network"""
        if not self.network_config:
            return 0
            
        observation_dim = self.network_config.get('observation_dim', 56)
        hidden_layers = self.network_config.get('hidden_layers', [])
        policy_hidden_size = self.network_config.get('policy_hidden_size', 12)
        value_hidden_size = self.network_config.get('value_hidden_size', 12)
        special_hidden_size = self.network_config.get('special_hidden_size', 12)
        
        # Ensure we have at least one hidden layer for calculation
        if len(hidden_layers) == 0:
            hidden_layers = [{'size': 12}]
        
        # Calculate parameters
        total_params = 0
        
        # Encoder layers
        input_size = observation_dim
        for layer in hidden_layers:
            layer_size = layer.get('size', 12)
            # Weights + biases
            total_params += (input_size * layer_size) + layer_size
            input_size = layer_size
        
        # Last hidden layer size
        last_hidden_size = hidden_layers[-1].get('size', 12) if hidden_layers else observation_dim
        
        # Policy head
        total_params += (last_hidden_size * policy_hidden_size) + policy_hidden_size  # First layer
        total_params += (policy_hidden_size * 2) + 2  # Output layer (2 outputs)
        
        # Value head
        total_params += (last_hidden_size * value_hidden_size) + value_hidden_size  # First layer
        total_params += (value_hidden_size * 1) + 1  # Output layer (1 output)
        
        # Special action head
        total_params += (last_hidden_size * special_hidden_size) + special_hidden_size  # First layer
        total_params += (special_hidden_size * 2) + 2  # Output layer (2 outputs)
        
        return total_params
    
    def _draw_layer(self, dc, layer_idx, layer_width, panel_height, layer_type, layer_size):
        """Helper method to draw a network layer"""
        x = layer_idx * layer_width
        y = panel_height / 2
        
        # Draw layer box
        box_width = layer_width * 0.8
        box_height = min(panel_height * 0.6, layer_size * 2 + 30)  # Scale height with layer size but with a cap
        
        dc.SetBrush(wx.Brush(wx.Colour(200, 220, 240)))
        dc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 1))
        
        box_x = x + (layer_width - box_width) / 2
        box_y = y - (box_height / 2)
        
        dc.DrawRectangle(int(box_x), int(box_y), int(box_width), int(box_height))
        
        # Draw layer label
        dc.SetTextForeground(wx.Colour(0, 0, 0))
        label = f"{layer_type} ({layer_size})"
        text_width, text_height = dc.GetTextExtent(label)
        
        text_x = x + (layer_width - text_width) / 2
        text_y = box_y + box_height + 5
        
        dc.DrawText(label, int(text_x), int(text_y))
        
        # Draw neurons (circles) inside the layer box
        num_neurons_to_draw = min(layer_size, 10)  # Limit the number of neurons to draw
        
        if num_neurons_to_draw > 0:
            neuron_spacing = box_height / (num_neurons_to_draw + 1)
            neuron_radius = min(5, neuron_spacing / 2 - 1)
            
            for i in range(num_neurons_to_draw):
                neuron_y = box_y + (i + 1) * neuron_spacing
                neuron_x = box_x + box_width / 2
                
                dc.SetBrush(wx.Brush(wx.Colour(100, 150, 200)))
                dc.DrawCircle(int(neuron_x), int(neuron_y), int(neuron_radius))
            
            # If there are more neurons than we're drawing, add an ellipsis
            if layer_size > num_neurons_to_draw:
                dc.SetTextForeground(wx.Colour(0, 0, 0))
                ellipsis = "..."
                ellipsis_width = dc.GetTextExtent(ellipsis)[0]
                dc.DrawText(ellipsis, int(box_x + (box_width - ellipsis_width) / 2), 
                           int(box_y + box_height - 20))

    def create_network_from_config(self):
        """Create a PodNetwork instance from the current configuration"""
        # Update config from UI first
        self.update_config()
        
        # Get network configuration
        network_config = self.config.get('network_config', {})
        
        # Create network
        from ..models.neural_pod import PodNetwork
        
        network = PodNetwork(
            observation_dim=network_config.get('observation_dim', 56),
            hidden_layers=network_config.get('hidden_layers', []),
            policy_hidden_size=network_config.get('policy_hidden_size', 12),
            value_hidden_size=network_config.get('value_hidden_size', 12),
            special_hidden_size=network_config.get('special_hidden_size', 12)
        )
        
        return network
