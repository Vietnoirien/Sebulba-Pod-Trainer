import sys
import os
import math
import json
import torch
import numpy as np
from pathlib import Path

from ..models.neural_pod import PodNetwork

class ModelExporter:
    """
    Exports a trained PyTorch model to pure Python code for CodinGame submission.
    The exported code will only use math and sys libraries.
    """
    def __init__(self, model_path: str, output_path: str = None):
        """
        Initialize the exporter with paths to the model and output file.
        
        Args:
            model_path: Path to the directory containing trained pod models
            output_path: Path where the output Python file will be saved
        """
        self.model_path = Path(model_path)
        self.output_path = Path(output_path) if output_path else Path("codingame_submission.py")
        
        # Load models
        self.runner_model = PodNetwork()
        self.blocker_model = PodNetwork()
        
        # Find the best model files
        runner_path = self._find_best_model_file("player0_pod0")
        if runner_path is None:
            # Try alternative naming patterns
            runner_path = self._find_best_model_file("pod0")
        
        if runner_path is not None:
            self.runner_model.load_state_dict(torch.load(runner_path, map_location='cpu'))
            print(f"Loaded runner model from {runner_path}")
        else:
            raise FileNotFoundError(f"Runner model not found in {self.model_path}")
        
        # Load blocker pod model if available
        blocker_path = self._find_best_model_file("player0_pod1")
        if blocker_path is None:
            # Try alternative naming patterns
            blocker_path = self._find_best_model_file("pod1")
        
        if blocker_path is not None:
            self.blocker_model.load_state_dict(torch.load(blocker_path, map_location='cpu'))
            print(f"Loaded blocker model from {blocker_path}")
        else:
            print(f"Blocker model not found in {self.model_path}, using runner model for both pods")
            self.blocker_model = self.runner_model
        
        # Set models to evaluation mode
        self.runner_model.eval()
        self.blocker_model.eval()
    
    def _find_best_model_file(self, pod_name):
        """Find the best model file for a pod using various naming patterns"""
        # First try standard filename
        standard_file = self.model_path / f"{pod_name}.pt"
        if standard_file.exists():
            return standard_file
            
        # Next try latest model
        latest_file = self.model_path / f"{pod_name}_latest.pt"
        if latest_file.exists():
            return latest_file
            
        # Try GPU-specific filename without iteration (latest)
        gpu_files = list(self.model_path.glob(f"{pod_name}_gpu*.pt"))
        if gpu_files:
            return gpu_files[0]  # Use the first one found
            
        # Try GPU-specific filename with iteration
        iter_files = list(self.model_path.glob(f"{pod_name}_gpu*_iter*.pt"))
        if iter_files:
            # Sort by iteration number to get the latest
            iter_files.sort(key=lambda x: int(str(x).split('_iter')[-1].split('.')[0]), reverse=True)
            return iter_files[0]
            
        # Try iteration-specific files
        iter_files = list(self.model_path.glob(f"{pod_name}_iter*.pt"))
        if iter_files:
            # Sort by iteration number to get the latest
            iter_files.sort(key=lambda x: int(str(x).split('_iter')[-1].split('.')[0]), reverse=True)
            return iter_files[0]
            
        return None
    
    def _extract_weights(self, model, quantize=True, precision=3):
        """
        Extract weights from PyTorch model as plain Python lists/dicts
        with optional quantization to reduce file size
        
        Args:
            model: The PyTorch model
            quantize: Whether to quantize weights
            precision: Number of decimal places to keep when quantizing
        """
        weights = {}
        
        # Extract weights from each layer
        for name, param in model.state_dict().items():
            param_numpy = param.detach().cpu().numpy()
            
            if quantize:
                # Quantize by rounding to specified precision
                param_numpy = np.round(param_numpy, precision)
                
            weights[name] = param_numpy.tolist()
        
        return weights
    
    def _compress_weights(self, weights_list, precision=3):
        """
        Compress weights to reduce file size using a more efficient encoding
        
        Args:
            weights_list: List of weights to compress
            precision: Number of decimal places to keep
        """
        if isinstance(weights_list, list):
            if all(isinstance(x, (int, float)) for x in weights_list):
                # For 1D arrays, use a more compact representation
                scale = 10**precision
                return [round(x * scale) / scale for x in weights_list]
            else:
                # Recursively compress nested lists
                return [self._compress_weights(x, precision) for x in weights_list]
        return weights_list
    
    def _encode_weights_efficiently(self, weights, precision=3):
        """
        Encode weights more efficiently to reduce file size
        
        Args:
            weights: Dictionary of weight tensors
            precision: Number of decimal places to keep
        """
        encoded_weights = {}
        
        for name, param in weights.items():
            if isinstance(param, list):
                # Determine if we can use a more efficient encoding
                if len(param) > 0 and isinstance(param[0], list):
                    # For 2D arrays (matrices)
                    flattened = []
                    shape = [len(param), len(param[0])]
                    
                    for row in param:
                        flattened.extend(row)
                    
                    # Quantize the flattened array
                    quantized = [round(x * 10**precision) / 10**precision for x in flattened]
                    
                    encoded_weights[name] = {
                        'shape': shape,
                        'data': quantized
                    }
                else:
                    # For 1D arrays (vectors)
                    encoded_weights[name] = [round(x * 10**precision) / 10**precision for x in param]
        
        return encoded_weights
    
    def _generate_layer_code(self, layer_name, weights, biases, precision=3):
        """Generate Python code for a linear layer with compressed weights"""
        code = []
        
        # Compress weights and biases
        compressed_weights = self._compress_weights(weights, precision)
        compressed_biases = self._compress_weights(biases, precision)
        
        # Format weights and biases as Python lists
        weights_str = json.dumps(compressed_weights)
        biases_str = json.dumps(compressed_biases)
        
        # Generate function for this layer
        code.append(f"def {layer_name}(x):")
        code.append(f"    weights = {weights_str}")
        code.append(f"    biases = {biases_str}")
        code.append("    result = [0.0] * len(biases)")
        code.append("    for i in range(len(result)):")
        code.append("        for j in range(len(x)):")
        code.append("            result[i] += x[j] * weights[i][j]")
        code.append("        result[i] += biases[i]")
        code.append("    return result")
        code.append("")
        
        return code
    
    def _generate_efficient_layer_code(self, layer_name, weights, biases, precision=3):
        """Generate Python code for a linear layer with efficiently encoded weights"""
        code = []
        
        # Encode weights and biases efficiently
        encoded_weights = {
            'shape': [len(weights), len(weights[0])],
            'data': [round(val * 10**precision) / 10**precision for row in weights for val in row]
        }
        encoded_biases = [round(val * 10**precision) / 10**precision for val in biases]
        
        # Generate function for this layer
        code.append(f"def {layer_name}(x):")
        code.append(f"    w_shape = {encoded_weights['shape']}")
        code.append(f"    w_data = {json.dumps(encoded_weights['data'])}")
        code.append(f"    biases = {json.dumps(encoded_biases)}")
        code.append("    result = [0.0] * len(biases)")
        code.append("    idx = 0")
        code.append("    for i in range(len(result)):")
        code.append("        for j in range(len(x)):")
        code.append("            result[i] += x[j] * w_data[idx]")
        code.append("            idx += 1")
        code.append("        result[i] += biases[i]")
        code.append("    return result")
        code.append("")
        
        return code
    
    def _generate_activation_functions(self):
        """Generate activation function code"""
        code = []
        
        # ReLU activation
        code.append("def relu(x):")
        code.append("    return [max(0.0, val) for val in x]")
        code.append("")
        
        # Tanh activation
        code.append("def tanh(x):")
        code.append("    return [math.tanh(val) for val in x]")
        code.append("")
        
        # Sigmoid activation
        code.append("def sigmoid(x):")
        code.append("    return [1.0 / (1.0 + math.exp(-val)) for val in x]")
        code.append("")
        
        return code
    
    def _generate_model_code(self, model_name, weights, precision=3):
        """Generate Python code for the full model with compressed weights"""
        code = []
        
        # Extract layer weights
        encoder_layers = [
            ("encoder.0", weights["encoder.0.weight"], weights["encoder.0.bias"]),
            ("encoder.2", weights["encoder.2.weight"], weights["encoder.2.bias"]),
        ]
        
        policy_layers = [
            ("policy_head.0", weights["policy_head.0.weight"], weights["policy_head.0.bias"]),
            ("policy_head.2", weights["policy_head.2.weight"], weights["policy_head.2.bias"]),
        ]
        
        special_layers = [
            ("special_action_head.0", weights["special_action_head.0.weight"], weights["special_action_head.0.bias"]),
            ("special_action_head.2", weights["special_action_head.2.weight"], weights["special_action_head.2.bias"]),
        ]
        
        # Generate layer functions with efficient encoding
        for layer_name, layer_weights, layer_biases in encoder_layers + policy_layers + special_layers:
            layer_code = self._generate_efficient_layer_code(
                f"{model_name}_{layer_name.replace('.', '_')}", 
                layer_weights, 
                layer_biases,
                precision
            )
            code.extend(layer_code)
        
        # Generate forward function
        code.append(f"def {model_name}_forward(x):")
        
        # Encoder
        code.append(f"    x1 = {model_name}_encoder_0(x)")
        code.append("    x1 = relu(x1)")
        code.append(f"    x1 = {model_name}_encoder_2(x1)")
        code.append("    x1 = relu(x1)")
        
        # Policy head
        code.append(f"    policy = {model_name}_policy_head_0(x1)")
        code.append("    policy = relu(policy)")
        code.append(f"    policy = {model_name}_policy_head_2(policy)")
        
        # Process policy outputs
        code.append("    angle = math.tanh(policy[0])")
        code.append("    thrust = sigmoid([policy[1]])[0]")
        
        # Special actions
        code.append(f"    special = {model_name}_special_action_head_0(x1)")
        code.append("    special = relu(special)")
        code.append(f"    special = {model_name}_special_action_head_2(special)")
        code.append("    shield_prob = sigmoid([special[0]])[0]")
        code.append("    boost_prob = sigmoid([special[1]])[0]")
        
        # Return processed outputs
        code.append("    return angle, thrust, shield_prob, boost_prob")
        code.append("")
        
        return code
    
    def _generate_observation_code(self):
        """Generate code for processing observations"""
        code = []
        
        code.append("def get_observations(x, y, vx, vy, angle, next_checkpoint_x, next_checkpoint_y, next_next_checkpoint_x, next_next_checkpoint_y):")
        code.append("    # Constants for normalization")
        code.append("    distance_upper_bound = 16000.0")
        code.append("    speed_upper_bound = 2000.0")
        
        code.append("    # Convert angle to radians")
        code.append("    angle_rad = angle * math.pi / 180.0")
        
        code.append("    # Calculate normalized observations")
        code.append("    obs = []")
        
        # Next checkpoint
        code.append("    # Next checkpoint relative position")
        code.append("    dx1 = (next_checkpoint_x - x) / distance_upper_bound")
        code.append("    dy1 = (next_checkpoint_y - y) / distance_upper_bound")
        code.append("    obs.append(dx1)")
        code.append("    obs.append(dy1)")
        
        code.append("    # Angle to next checkpoint")
        code.append("    target_angle = math.atan2(next_checkpoint_y - y, next_checkpoint_x - x)")
        code.append("    angle_diff = (target_angle - angle_rad + math.pi) % (2 * math.pi) - math.pi")
        code.append("    obs.append(angle_diff / math.pi)")

        # Next-next checkpoint        
        code.append("    # Next-next checkpoint relative position")
        code.append("    dx2 = (next_next_checkpoint_x - x) / distance_upper_bound")
        code.append("    dy2 = (next_next_checkpoint_y - y) / distance_upper_bound")
        code.append("    obs.append(dx2)")
        code.append("    obs.append(dy2)")
        
        code.append("    # Angle to next-next checkpoint")
        code.append("    target_angle2 = math.atan2(next_next_checkpoint_y - y, next_next_checkpoint_x - x)")
        code.append("    angle_diff2 = (target_angle2 - angle_rad + math.pi) % (2 * math.pi) - math.pi")
        code.append("    obs.append(angle_diff2 / math.pi)")
        
        # Speed
        code.append("    # Normalized speed")
        code.append("    obs.append(vx / speed_upper_bound)")
        code.append("    obs.append(vy / speed_upper_bound)")
        
        code.append("    return obs")
        code.append("")
        
        return code
    
    def _generate_main_code(self):
        """Generate the main game loop code"""
        code = []
        
        # Imports
        code.append("import sys")
        code.append("import math")
        code.append("")
        
        # Game initialization
        code.append("# Read game input")
        code.append("laps = int(input())")
        code.append("checkpoint_count = int(input())")
        code.append("checkpoints = []")
        code.append("for i in range(checkpoint_count):")
        code.append("    checkpoint_x, checkpoint_y = [int(j) for j in input().split()]")
        code.append("    checkpoints.append((checkpoint_x, checkpoint_y))")
        code.append("")
        
        # Game state tracking
        code.append("# Game state")
        code.append("boost_used = [False, False]  # Track boost usage for each pod")
        code.append("turn_count = 0")
        code.append("")
        
        # Main game loop
        code.append("# Game loop")
        code.append("while True:")
        code.append("    turn_count += 1")
        code.append("    pods = []")
        code.append("    # Read pod data")
        code.append("    for i in range(2):")
        code.append("        x, y, vx, vy, angle, next_checkpoint_id = [int(j) for j in input().split()]")
        code.append("        pods.append((x, y, vx, vy, angle, next_checkpoint_id))")
        code.append("    # Read opponent data")
        code.append("    opponents = []")
        code.append("    for i in range(2):")
        code.append("        x, y, vx, vy, angle, next_checkpoint_id = [int(j) for j in input().split()]")
        code.append("        opponents.append((x, y, vx, vy, angle, next_checkpoint_id))")
        code.append("")
        
        # Process each pod and store commands
        code.append("    # Process each pod and store commands")
        code.append("    commands = []")
        code.append("    for pod_id in range(2):")
        code.append("        # Get pod data")
        code.append("        x, y, vx, vy, angle, next_checkpoint_id = pods[pod_id]")
        code.append("        next_checkpoint_x, next_checkpoint_y = checkpoints[next_checkpoint_id]")
        code.append("        next_next_id = (next_checkpoint_id + 1) % len(checkpoints)")
        code.append("        next_next_checkpoint_x, next_next_checkpoint_y = checkpoints[next_next_id]")
        code.append("")
        
        # Get observations and predictions
        code.append("        # Get observations")
        code.append("        obs = get_observations(")
        code.append("            x, y, vx, vy, angle, ")
        code.append("            next_checkpoint_x, next_checkpoint_y,")
        code.append("            next_next_checkpoint_x, next_next_checkpoint_y")
        code.append("        )")
        code.append("")
        
        # Use appropriate model based on pod ID
        code.append("        # Get model predictions")
        code.append("        if pod_id == 0:")
        code.append("            angle, thrust, shield_prob, boost_prob = runner_forward(obs)")
        code.append("        else:")
        code.append("            angle, thrust, shield_prob, boost_prob = blocker_forward(obs)")
        code.append("")
        
        # Convert model outputs to game actions
        code.append("        # Convert angle to target coordinates")
        code.append("        target_angle = angle * math.pi  # Convert from [-1,1] to [-π,π]")
        code.append("        angle_rad = angle * math.pi / 180.0")
        code.append("        target_x = x + math.cos(angle_rad + target_angle) * 10000")
        code.append("        target_y = y + math.sin(angle_rad + target_angle) * 10000")
        code.append("")
        
        # Determine thrust value
        code.append("        # Determine thrust")
        code.append("        if shield_prob > 0.5:")
        code.append("            thrust_command = 'SHIELD'")
        code.append("        elif boost_prob > 0.5 and not boost_used[pod_id]:")
        code.append("            thrust_command = 'BOOST'")
        code.append("            boost_used[pod_id] = True")
        code.append("        else:")
        code.append("            thrust_value = int(thrust * 100)")
        code.append("            thrust_command = str(thrust_value)")
        code.append("")
        
        # Store command for this pod
        code.append("        # Store command for this pod")
        code.append("        commands.append(f\"{int(target_x)} {int(target_y)} {thrust_command}\")")
        code.append("")
        
        # Output commands for both pods
        code.append("    # Output commands for both pods")
        code.append("    for command in commands:")
        code.append("        print(command)")
        code.append("")
        
        return code
    
    def export(self, quantize=True, precision=3):
        """
        Export the model to a Python file for CodinGame submission
        
        Args:
            quantize: Whether to quantize weights
            precision: Number of decimal places to keep when quantizing
        """
        # Extract weights from models with quantization
        runner_weights = self._extract_weights(self.runner_model, quantize=quantize, precision=precision)
        blocker_weights = self._extract_weights(self.blocker_model, quantize=quantize, precision=precision)
        
        # Generate code sections
        code = []
        
        # Add activation functions
        code.extend(self._generate_activation_functions())
        
        # Add observation processing
        code.extend(self._generate_observation_code())
        
        # Add model code with quantized weights
        code.extend(self._generate_model_code("runner", runner_weights, precision))
        code.extend(self._generate_model_code("blocker", blocker_weights, precision))
        
        # Add main game loop
        code.extend(self._generate_main_code())
        
        # Write code to file
        with open(self.output_path, 'w') as f:
            f.write('\n'.join(code))
        
        file_size_kb = os.path.getsize(self.output_path) / 1024
        print(f"Model exported to {self.output_path}")
        print(f"File size: {file_size_kb:.2f} KB with quantization={quantize}, precision={precision}")
