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
    def __init__(self, model_path: str, output_path: str = None, network_config: dict = None):
        """
        Initialize the exporter with paths to the model and output file.
        
        Args:
            model_path: Path to the directory containing trained pod models
            output_path: Path where the output Python file will be saved
            network_config: Network configuration dict (if None, will try to infer from model)
        """
        self.model_path = Path(model_path)
        self.output_path = Path(output_path) if output_path else Path("codingame_submission.py")
        self.network_config = network_config
        
        # Load models with correct architecture
        self.runner_model = None
        self.blocker_model = None
        
        # Load default models (can be overridden later with set_model_files)
        self._load_default_models()
    
    def _infer_network_config(self, model_path):
        """Infer network configuration from a saved model with improved detection"""
        try:
            # Load the state dict to examine layer shapes
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Infer observation dimension from first layer
            first_layer_weight = state_dict.get('encoder.0.weight')
            if first_layer_weight is not None:
                observation_dim = first_layer_weight.shape[1]
            else:
                observation_dim = 56  # Default fallback
            
            # Infer hidden layers from encoder - improved detection
            hidden_layers = []
            layer_idx = 0
            
            while f'encoder.{layer_idx}.weight' in state_dict:
                weight = state_dict[f'encoder.{layer_idx}.weight']
                hidden_size = weight.shape[0]
                
                # Check what comes after this layer to determine activation
                activation_type = 'ReLU'  # Default
                if f'encoder.{layer_idx + 1}.weight' in state_dict:
                    # This means no activation between layers (rare)
                    activation_type = 'Linear'
                    layer_idx += 1
                else:
                    # Assume activation layer, skip it
                    layer_idx += 2
                
                hidden_layers.append({
                    'type': f'Linear+{activation_type}',
                    'size': hidden_size
                })
            
            # If no encoder layers found, try to infer from policy head input
            if not hidden_layers and 'policy_head.0.weight' in state_dict:
                # The policy head input size tells us the encoder output size
                encoder_output_size = state_dict['policy_head.0.weight'].shape[1]
                # Create a default single layer
                hidden_layers = [{
                    'type': 'Linear+ReLU',
                    'size': encoder_output_size
                }]
            
            # Infer head sizes with better error handling
            policy_hidden_size = 12  # Default
            value_hidden_size = 12   # Default
            special_hidden_size = 12 # Default
            
            if 'policy_head.0.weight' in state_dict:
                policy_hidden_size = state_dict['policy_head.0.weight'].shape[0]
            
            if 'value_head.0.weight' in state_dict:
                value_hidden_size = state_dict['value_head.0.weight'].shape[0]
                
            if 'special_action_head.0.weight' in state_dict:
                special_hidden_size = state_dict['special_action_head.0.weight'].shape[0]
            
            config = {
                'observation_dim': observation_dim,
                'hidden_layers': hidden_layers,
                'policy_hidden_size': policy_hidden_size,
                'value_hidden_size': value_hidden_size,
                'special_hidden_size': special_hidden_size
            }
            
            print(f"Inferred network config from {model_path}: {config}")
            return config
            
        except Exception as e:
            print(f"Error inferring network config from {model_path}: {e}")
            # Return a more conservative default config
            return {
                'observation_dim': 56,  # Updated default
                'hidden_layers': [
                    {'type': 'Linear+ReLU', 'size': 24},
                    {'type': 'Linear+ReLU', 'size': 16}
                ],
                'policy_hidden_size': 12,
                'value_hidden_size': 12,
                'special_hidden_size': 12
            }
    
    def _adapt_config_to_state_dict(self, state_dict, base_config):
        """Adapt configuration to match the actual state dict structure"""
        adapted_config = base_config.copy() if base_config else {}
        
        # Check for old parameter names and map them to new ones
        if 'special_hidden_size' in adapted_config and 'action_hidden_size' not in adapted_config:
            adapted_config['action_hidden_size'] = adapted_config['special_hidden_size']
        
        # Infer sizes from state dict if not in config
        for key, tensor in state_dict.items():
            if 'action_head' in key and 'weight' in key:
                if key.endswith('0.weight'):  # First layer of action head
                    adapted_config['action_hidden_size'] = tensor.shape[0]
                elif key.endswith('2.weight'):  # Output layer of action head
                    # Should be 4 for the new format
                    expected_output_size = tensor.shape[0]
                    if expected_output_size != 4:
                        print(f"Warning: Action head output size is {expected_output_size}, expected 4")
            
            elif 'policy_head' in key and 'weight' in key:
                if key.endswith('0.weight'):  # First layer of policy head (legacy)
                    adapted_config['policy_hidden_size'] = tensor.shape[0]
            
            elif 'special_action_head' in key and 'weight' in key:
                if key.endswith('0.weight'):  # First layer of special action head (legacy)
                    adapted_config['special_hidden_size'] = tensor.shape[0]
            
            elif 'value_head' in key and 'weight' in key:
                if key.endswith('0.weight'):  # First layer of value head
                    adapted_config['value_hidden_size'] = tensor.shape[0]
        
        return adapted_config
    
    def _load_model_with_config(self, model_path, config=None):
        """Load a model with the given or inferred configuration"""
        if config is None:
            config = self._infer_network_config(model_path)
        
        try:
            # Create model with inferred config
            model = PodNetwork(
                observation_dim=config.get('observation_dim', 56),
                hidden_layers=config.get('hidden_layers', []),
                policy_hidden_size=config.get('policy_hidden_size', 12),
                value_hidden_size=config.get('value_hidden_size', 12),
                action_hidden_size=config.get('action_hidden_size', 12)  # Updated parameter name
            )
            
            # Load state dict
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            
            print(f"Successfully loaded model from {model_path} with config: {config}")
            return model
            
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print(f"Attempted config: {config}")
            
            # Try with a more flexible approach - load state dict first and adapt
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                
                # Create a model that matches the state dict exactly
                adapted_config = self._adapt_config_to_state_dict(state_dict, config)
                
                model = PodNetwork(
                    observation_dim=adapted_config.get('observation_dim', 56),
                    hidden_layers=adapted_config.get('hidden_layers', []),
                    policy_hidden_size=adapted_config.get('policy_hidden_size', 12),
                    value_hidden_size=adapted_config.get('value_hidden_size', 12),
                    action_hidden_size=adapted_config.get('action_hidden_size', 12)  # Updated parameter name
                )
                
                model.load_state_dict(state_dict)
                model.eval()
                
                print(f"Successfully loaded model with adapted config: {adapted_config}")
                return model
                
            except Exception as e2:
                print(f"Failed to load model even with adaptation: {e2}")
                raise e2
    
    def _find_best_model_file_for_role(self, role):
        """Find the best model file for a specific role (runner or blocker)"""
        # Look for role-based files first (new naming convention)
        role_files = list(self.model_path.glob(f"player*_{role}*.pt"))
        
        # Fallback to legacy pod-based naming if no role-based files found
        if not role_files:
            if role == "runner":
                role_files = list(self.model_path.glob("player*_pod0*.pt"))
            elif role == "blocker":
                role_files = list(self.model_path.glob("player*_pod1*.pt"))
        
        if not role_files:
            return None
        
        # First try standard filename (usually the final/best model)
        standard_files = [f for f in role_files if '_gpu' not in f.name and '_iter' not in f.name]
        if standard_files:
            return standard_files[0]
        
        # Next try latest model
        latest_files = [f for f in role_files if '_latest' in f.name]
        if latest_files:
            return latest_files[0]
        
        # Try GPU-specific filename without iteration (latest)
        gpu_files = [f for f in role_files if '_gpu' in f.name and '_iter' not in f.name]
        if gpu_files:
            return gpu_files[0]
        
        # Try GPU-specific filename with iteration (get latest iteration)
        iter_files = [f for f in role_files if '_iter' in f.name]
        if iter_files:
            # Sort by iteration number to get the latest
            iter_files_with_numbers = []
            for file in iter_files:
                try:
                    iter_num = int(str(file).split('_iter')[-1].split('.')[0])
                    iter_files_with_numbers.append((iter_num, file))
                except (ValueError, IndexError):
                    continue
            
            if iter_files_with_numbers:
                iter_files_with_numbers.sort(key=lambda x: x[0], reverse=True)
                return iter_files_with_numbers[0][1]
        
        # Final fallback: return the first file
        return role_files[0]
    
    def _load_default_models(self):
        """Load default models using role-based auto-discovery"""
        # Find the best model files for each role
        runner_path = self._find_best_model_file_for_role("runner")
        
        if runner_path is not None:
            runner_config = self._infer_network_config(runner_path)
            self.runner_model = self._load_model_with_config(runner_path, runner_config)
            self.network_config = runner_config  # Store for reference
            print(f"Loaded runner model: {runner_path}")
        else:
            raise FileNotFoundError(f"Runner model not found in {self.model_path}")
        
        # Load blocker pod model if available
        blocker_path = self._find_best_model_file_for_role("blocker")
        
        if blocker_path is not None:
            blocker_config = self._infer_network_config(blocker_path)
            self.blocker_model = self._load_model_with_config(blocker_path, blocker_config)
            print(f"Loaded blocker model: {blocker_path}")
        else:
            print(f"Blocker model not found in {self.model_path}, using runner model for both roles")
            self.blocker_model = self.runner_model
    
    def set_model_files(self, model_files):
        """
        Set specific model files to use instead of auto-discovery
        
        Args:
            model_files: Dict with keys 'runner' and 'blocker' or legacy 'player0_pod0' and 'player0_pod1'
        """
        # Handle both new role-based and legacy pod-based keys
        runner_path = None
        blocker_path = None
        
        # Check for new role-based keys first
        if 'runner' in model_files:
            runner_path = Path(model_files['runner'])
        elif 'player0_pod0' in model_files:
            runner_path = Path(model_files['player0_pod0'])
        
        if 'blocker' in model_files:
            blocker_path = Path(model_files['blocker'])
        elif 'player0_pod1' in model_files:
            blocker_path = Path(model_files['player0_pod1'])
        
        # Load runner model
        if runner_path and runner_path.exists():
            runner_config = self._infer_network_config(runner_path)
            self.runner_model = self._load_model_with_config(runner_path, runner_config)
            self.network_config = runner_config  # Store for reference
            print(f"Set runner model to {runner_path}")
        
        # Load blocker model
        if blocker_path and blocker_path.exists():
            blocker_config = self._infer_network_config(blocker_path)
            self.blocker_model = self._load_model_with_config(blocker_path, blocker_config)
            print(f"Set blocker model to {blocker_path}")
        
        # If only one model was specified, use it for both roles if the other wasn't loaded
        if len(model_files) == 1:
            if runner_path and self.blocker_model is None:
                print("Using runner model for blocker role as well")
                self.blocker_model = self.runner_model
            elif blocker_path and self.runner_model is None:
                print("Using blocker model for runner role as well")
                self.runner_model = self.blocker_model
    
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
        code.append("    return [1.0 / (1.0 + math.exp(-max(-500, min(500, val)))) for val in x]")  # Clamp to prevent overflow
        code.append("")
        
        return code
    
    def _generate_model_code(self, model_name, weights, precision=3):
        """Generate Python code for the full model with dynamic layer handling"""
        code = []
        
        # Extract encoder layers dynamically
        encoder_layers = []
        layer_idx = 0
        
        # Find all encoder layers
        while f"encoder.{layer_idx}.weight" in weights:
            layer_weights = weights[f"encoder.{layer_idx}.weight"]
            layer_biases = weights[f"encoder.{layer_idx}.bias"]
            encoder_layers.append((f"encoder.{layer_idx}", layer_weights, layer_biases))
            layer_idx += 2  # Skip activation layers
        
        # Extract head layers - Updated for new model structure
        head_layers = []
        
        # Action head layers (new structure)
        action_layers = []
        action_layer_idx = 0
        while f"action_head.{action_layer_idx}.weight" in weights:
            layer_weights = weights[f"action_head.{action_layer_idx}.weight"]
            layer_biases = weights[f"action_head.{action_layer_idx}.bias"]
            action_layers.append((f"action_head.{action_layer_idx}", layer_weights, layer_biases))
            head_layers.append((f"action_head.{action_layer_idx}", layer_weights, layer_biases))
            action_layer_idx += 2  # Skip activation layers
        
        # Value head layers
        value_layers = []
        value_layer_idx = 0
        while f"value_head.{value_layer_idx}.weight" in weights:
            layer_weights = weights[f"value_head.{value_layer_idx}.weight"]
            layer_biases = weights[f"value_head.{value_layer_idx}.bias"]
            value_layers.append((f"value_head.{value_layer_idx}", layer_weights, layer_biases))
            head_layers.append((f"value_head.{value_layer_idx}", layer_weights, layer_biases))
            value_layer_idx += 2  # Skip activation layers
        
        # Legacy support: Check for old policy_head and special_action_head
        policy_layers = []
        special_layers = []
        
        # Policy head layers (legacy)
        policy_layer_idx = 0
        while f"policy_head.{policy_layer_idx}.weight" in weights:
            layer_weights = weights[f"policy_head.{policy_layer_idx}.weight"]
            layer_biases = weights[f"policy_head.{policy_layer_idx}.bias"]
            policy_layers.append((f"policy_head.{policy_layer_idx}", layer_weights, layer_biases))
            head_layers.append((f"policy_head.{policy_layer_idx}", layer_weights, layer_biases))
            policy_layer_idx += 2
        
        # Special action head layers (legacy)
        special_layer_idx = 0
        while f"special_action_head.{special_layer_idx}.weight" in weights:
            layer_weights = weights[f"special_action_head.{special_layer_idx}.weight"]
            layer_biases = weights[f"special_action_head.{special_layer_idx}.bias"]
            special_layers.append((f"special_action_head.{special_layer_idx}", layer_weights, layer_biases))
            head_layers.append((f"special_action_head.{special_layer_idx}", layer_weights, layer_biases))
            special_layer_idx += 2
        
        # Generate layer functions with efficient encoding
        all_layers = encoder_layers + head_layers
        for layer_name, layer_weights, layer_biases in all_layers:
            layer_code = self._generate_efficient_layer_code(
                f"{model_name}_{layer_name.replace('.', '_')}", 
                layer_weights, 
                layer_biases,
                precision
            )
            code.extend(layer_code)
        
        # Generate forward function - Updated for new model structure
        code.append(f"def {model_name}_forward(x):")

        # Encoder layers
        current_var = "x"
        for i, (layer_name, _, _) in enumerate(encoder_layers):
            next_var = f"x{i+1}"
            code.append(f"    {next_var} = {model_name}_{layer_name.replace('.', '_')}({current_var})")
            code.append(f"    {next_var} = relu({next_var})")
            current_var = next_var
        
        # Check if we have new action_head structure or legacy policy/special heads
        if action_layers:
            # New model structure with action_head
            code.append(f"    # Action head (new structure)")
            if len(action_layers) >= 2:
                code.append(f"    action = {model_name}_action_head_0({current_var})")
                code.append("    action = relu(action)")
                code.append(f"    action = {model_name}_action_head_2(action)")
            elif len(action_layers) == 1:
                code.append(f"    action = {model_name}_action_head_0({current_var})")
            
            # Extract and process action components
            code.append("    # Extract action components")
            code.append("    angle = math.tanh(action[0])")  # Range: [-1, 1]
            code.append("    thrust = sigmoid([action[1]])[0]")  # Range: [0, 1]
            code.append("    shield_prob = sigmoid([action[2]])[0]")  # Range: [0, 1]
            code.append("    boost_prob = sigmoid([action[3]])[0]")  # Range: [0, 1]
            
        else:
            # Legacy model structure with separate policy and special heads
            code.append(f"    # Policy head (legacy structure)")
            if len(policy_layers) >= 2:
                code.append(f"    policy = {model_name}_policy_head_0({current_var})")
                code.append("    policy = relu(policy)")
                code.append(f"    policy = {model_name}_policy_head_2(policy)")
            elif len(policy_layers) == 1:
                code.append(f"    policy = {model_name}_policy_head_0({current_var})")
            
            # Apply proper activations to policy outputs
            code.append("    angle = math.tanh(policy[0])")  # Normalize to [-1,1]
            code.append("    thrust = sigmoid([policy[1]])[0]")  # Normalize to [0,1]
            
            # Special actions head - handle variable number of layers
            code.append(f"    # Special action head (legacy structure)")
            if len(special_layers) >= 2:
                code.append(f"    special = {model_name}_special_action_head_0({current_var})")
                code.append("    special = relu(special)")
                code.append(f"    special = {model_name}_special_action_head_2(special)")
            elif len(special_layers) == 1:
                code.append(f"    special = {model_name}_special_action_head_0({current_var})")
            
            # Apply proper activations to special action outputs
            code.append("    shield_prob = sigmoid([special[0]])[0]")  # Normalize to [0,1]
            code.append("    boost_prob = sigmoid([special[1]])[0]")  # Normalize to [0,1]
        
        # Return processed outputs (same format for both structures)
        code.append("    return angle, thrust, shield_prob, boost_prob")
        code.append("")
        
        return code
    
    def _generate_observation_code(self):
        """Generate code for processing observations to match training environment (56 dimensions)"""
        code = []
        
        code.append("def get_observations(pod_data, opponent_data, checkpoints, boost_used, pod_id):")
        code.append("    # Constants for normalization")
        code.append("    WIDTH = 16000.0")
        code.append("    HEIGHT = 9000.0")
        code.append("    MAX_SPEED = 800.0")
        code.append("    CHECKPOINT_RADIUS = 600.0")
        code.append("")
        
        code.append("    # Extract pod data")
        code.append("    x, y, vx, vy, angle, next_checkpoint_id = pod_data")
        code.append("    next_checkpoint_x, next_checkpoint_y = checkpoints[next_checkpoint_id]")
        code.append("    next_next_id = (next_checkpoint_id + 1) % len(checkpoints)")
        code.append("    next_next_checkpoint_x, next_next_checkpoint_y = checkpoints[next_next_id]")
        code.append("")

        code.append("    # Start building observation vector")
        code.append("    obs = []")
        code.append("")
        
        # Base observations (48 dimensions)
        code.append("    # === BASE OBSERVATIONS (48 dimensions) ===")
        code.append("    # 1-2: Normalized position")
        code.append("    norm_x = (x * 2.0 / WIDTH) - 1.0")
        code.append("    norm_y = (y * 2.0 / HEIGHT) - 1.0")
        code.append("    obs.extend([norm_x, norm_y])")
        code.append("")
        
        code.append("    # 3-4: Normalized velocity") 
        code.append("    norm_vx = vx / 1000.0")
        code.append("    norm_vy = vy / 1000.0")
        code.append("    obs.extend([norm_vx, norm_vy])")
        code.append("")

        code.append("    # 5-6: Relative position to next checkpoint (normalized)")
        code.append("    next_cp_norm_x = (next_checkpoint_x * 2.0 / WIDTH) - 1.0")
        code.append("    next_cp_norm_y = (next_checkpoint_y * 2.0 / HEIGHT) - 1.0")
        code.append("    rel_next_x = next_cp_norm_x - norm_x")
        code.append("    rel_next_y = next_cp_norm_y - norm_y")
        code.append("    obs.extend([rel_next_x, rel_next_y])")
        code.append("")

        code.append("    # 7-8: Relative position to next-next checkpoint (normalized)")        
        code.append("    next_next_cp_norm_x = (next_next_checkpoint_x * 2.0 / WIDTH) - 1.0")
        code.append("    next_next_cp_norm_y = (next_next_checkpoint_y * 2.0 / HEIGHT) - 1.0")
        code.append("    rel_next_next_x = next_next_cp_norm_x - norm_x")
        code.append("    rel_next_next_y = next_next_cp_norm_y - norm_y")
        code.append("    obs.extend([rel_next_next_x, rel_next_next_y])")
        code.append("")
        
        code.append("    # 9-10: Pod's absolute angle as sin/cos")
        code.append("    pod_angle_rad = math.radians(angle)")
        code.append("    pod_angle_sin = math.sin(pod_angle_rad)")
        code.append("    pod_angle_cos = math.cos(pod_angle_rad)")
        code.append("    obs.extend([pod_angle_sin, pod_angle_cos])")
        code.append("")
        
        code.append("    # 11-12: Angle to next checkpoint as sin/cos")
        code.append("    angle_to_next = math.atan2(next_checkpoint_y - y, next_checkpoint_x - x)")
        code.append("    angle_to_next_sin = math.sin(angle_to_next)")
        code.append("    angle_to_next_cos = math.cos(angle_to_next)")
        code.append("    obs.extend([angle_to_next_sin, angle_to_next_cos])")
        code.append("")
        
        code.append("    # 13: Relative angle (how much to turn) normalized to [-1, 1]")
        code.append("    angle_to_next_deg = math.degrees(angle_to_next)")
        code.append("    relative_angle = angle_to_next_deg - angle")
        code.append("    # Normalize to [-180, 180] range")
        code.append("    while relative_angle > 180:")
        code.append("        relative_angle -= 360")
        code.append("    while relative_angle < -180:")
        code.append("        relative_angle += 360")
        code.append("    relative_angle_normalized = relative_angle / 180.0")
        code.append("    obs.append(relative_angle_normalized)")
        code.append("")
        
        code.append("    # 14: Speed magnitude")
        code.append("    speed_magnitude = math.sqrt(vx * vx + vy * vy) / 800.0")
        code.append("    obs.append(speed_magnitude)")
        code.append("")
        
        code.append("    # 15: Distance to next checkpoint")
        code.append("    distance_to_next_cp = math.sqrt((next_checkpoint_x - x) ** 2 + (next_checkpoint_y - y) ** 2)")
        code.append("    distance_normalized = distance_to_next_cp / 600.0")
        code.append("    obs.append(distance_normalized)")
        code.append("")
        
        code.append("    # 16: Progress through race")
        code.append("    total_checkpoints = len(checkpoints) * 3  # Assuming 3 laps")
        code.append("    progress = next_checkpoint_id / total_checkpoints")
        code.append("    obs.append(progress)")
        code.append("")
        
        code.append("    # 17: Shield cooldown (simplified - always 0 in CodinGame)")
        code.append("    shield_cooldown = 0.0")
        code.append("    obs.append(shield_cooldown)")
        code.append("")
        
        code.append("    # 18: Boost availability")
        code.append("    boost_available = 0.0 if boost_used[pod_id] else 1.0")
        code.append("    obs.append(boost_available)")
        code.append("")
        
        # Opponent information (30 dimensions)
        code.append("    # Add opponent information (30 dimensions total)")
        code.append("    # Build list of all other pods: teammate + 2 opponents")
        code.append("    all_other_pods = []")
        code.append("    ")
        code.append("    # Add teammate data (if pod_id == 0, teammate is pod 1, vice versa)")
        code.append("    teammate_id = 1 - pod_id")
        code.append("    if teammate_id < len(pod_data_all):")
        code.append("        all_other_pods.append(pod_data_all[teammate_id])")
        code.append("    else:")
        code.append("        # Fallback: duplicate current pod data")
        code.append("        all_other_pods.append(pod_data)")
        code.append("    ")
        code.append("    # Add opponent data (2 opponent pods)")
        code.append("    all_other_pods.extend(opponent_data)")
        code.append("    ")
        code.append("    # Ensure we have exactly 3 opponents (teammate + 2 opponents)")
        code.append("    while len(all_other_pods) < 3:")
        code.append("        all_other_pods.append(all_other_pods[-1])  # Duplicate last if needed")
        code.append("    all_other_pods = all_other_pods[:3]  # Take only first 3")
        code.append("")
        
        code.append("    # Process each opponent (10 dimensions each)")
        code.append("    for opp_data in all_other_pods:")
        code.append("        opp_x, opp_y, opp_vx, opp_vy, opp_angle, opp_next_cp_id = opp_data")
        code.append("        ")
        code.append("        # 1-2: Relative position (normalized)")
        code.append("        opp_norm_x = (opp_x * 2.0 / WIDTH) - 1.0")
        code.append("        opp_norm_y = (opp_y * 2.0 / HEIGHT) - 1.0")
        code.append("        rel_opp_x = opp_norm_x - norm_x")
        code.append("        rel_opp_y = opp_norm_y - norm_y")
        code.append("        obs.extend([rel_opp_x, rel_opp_y])")
        code.append("        ")
        code.append("        # 3-4: Relative velocity")
        code.append("        rel_opp_vx = (opp_vx - vx) / 1000.0")
        code.append("        rel_opp_vy = (opp_vy - vy) / 1000.0")
        code.append("        obs.extend([rel_opp_vx, rel_opp_vy])")
        code.append("        ")
        code.append("        # 5: Distance to opponent (normalized)")
        code.append("        opp_distance = math.sqrt((opp_x - x)**2 + (opp_y - y)**2) / (WIDTH/2)")
        code.append("        obs.append(opp_distance)")
        code.append("        ")
        code.append("        # 6: Opponent's absolute angle (normalized)")
        code.append("        opp_normalized_angle = (opp_angle % 360) / 180.0 - 1.0")
        code.append("        obs.append(opp_normalized_angle)")
        code.append("        ")
        code.append("        # 7: Opponent's progress")
        code.append("        opp_progress = opp_next_cp_id / total_checkpoints")
        code.append("        obs.append(opp_progress)")
        code.append("        ")
        code.append("        # 8: Opponent's next checkpoint (normalized)")
        code.append("        opp_next_cp_normalized = (opp_next_cp_id % len(checkpoints)) / len(checkpoints)")
        code.append("        obs.append(opp_next_cp_normalized)")
        code.append("        ")
        code.append("        # 9: Opponent's shield cooldown (always 0 in CodinGame)")
        code.append("        obs.append(0.0)")
        code.append("        ")
        code.append("        # 10: Opponent's boost availability (assume available)")
        code.append("        obs.append(1.0)")
        code.append("")
        
        # Role-specific observations (8 dimensions)
        code.append("    # === ROLE-SPECIFIC OBSERVATIONS (8 dimensions) ===")
        code.append("    # Role identifier")
        code.append("    role_id = float(pod_id)  # 0 for runner (pod0), 1 for blocker (pod1)")
        code.append("    obs.append(role_id)")
        code.append("")
        
        code.append("    if pod_id == 1:  # Blocker-specific observations")
        code.append("        # Get teammate (runner) data")
        code.append("        runner_x, runner_y, runner_vx, runner_vy, runner_angle, runner_next_cp = pod_data_all[0]")
        code.append("        ")
        code.append("        # Distance to teammate runner (normalized)")
        code.append("        runner_distance = math.sqrt((x - runner_x)**2 + (y - runner_y)**2) / 8000.0")
        code.append("        obs.append(runner_distance)")
        code.append("        ")
        code.append("        # Runner's progress relative to blocker")
        code.append("        progress_diff = (runner_next_cp - next_checkpoint_id) / 5.0")
        code.append("        progress_diff = max(-1.0, min(1.0, progress_diff))  # Clamp to [-1, 1]")
        code.append("        obs.append(progress_diff)")
        code.append("        ")
        code.append("        # Find closest opponent")
        code.append("        min_opp_distance = float('inf')")
        code.append("        closest_opp_x, closest_opp_y = 0, 0")
        code.append("        for opp_data in opponent_data:")
        code.append("            opp_x_temp, opp_y_temp = opp_data[0], opp_data[1]")
        code.append("            opp_distance = math.sqrt((x - opp_x_temp)**2 + (y - opp_y_temp)**2)")
        code.append("            if opp_distance < min_opp_distance:")
        code.append("                min_opp_distance = opp_distance")
        code.append("                closest_opp_x, closest_opp_y = opp_x_temp, opp_y_temp")
        code.append("        ")
        code.append("        # Normalized distance to closest opponent")
        code.append("        closest_opp_distance_norm = min_opp_distance / 8000.0")
        code.append("        obs.append(closest_opp_distance_norm)")
        code.append("        ")
        code.append("        # Relative position to closest opponent (normalized)")
        code.append("        rel_closest_opp_x = (closest_opp_x - x) / 8000.0")
        code.append("        rel_closest_opp_y = (closest_opp_y - y) / 8000.0")
        code.append("        obs.extend([rel_closest_opp_x, rel_closest_opp_y])")
        code.append("        ")
        code.append("        # Blocking opportunity score")
        code.append("        blocking_opportunity = 0.0")
        code.append("        for opp_data in opponent_data:")
        code.append("            opp_x_temp, opp_y_temp, _, _, _, opp_next_cp = opp_data")
        code.append("            # Get opponent's next checkpoint")
        code.append("            opp_cp_x, opp_cp_y = checkpoints[opp_next_cp]")
        code.append("            ")
        code.append("            # Distance from opponent to their checkpoint")
        code.append("            opp_to_cp_dist = math.sqrt((opp_cp_x - opp_x_temp)**2 + (opp_cp_y - opp_y_temp)**2)")
        code.append("            # Distance from blocker to opponent's checkpoint")
        code.append("            blocker_to_cp_dist = math.sqrt((opp_cp_x - x)**2 + (opp_cp_y - y)**2)")
        code.append("            ")
        code.append("            # Simple interception opportunity")
        code.append("            if opp_to_cp_dist > 0:")
        code.append("                intercept_score = max(0.0, 1.0 - blocker_to_cp_dist / (opp_to_cp_dist + 1e-6))")
        code.append("                blocking_opportunity += intercept_score")
        code.append("        ")
        code.append("        blocking_opportunity = min(1.0, blocking_opportunity / 2.0)  # Normalize")
        code.append("        obs.append(blocking_opportunity)")
        code.append("        ")
        code.append("        # Reserved dimension for future use")
        code.append("        obs.append(0.0)")
        code.append("    else:")
        code.append("        # Runner pod - fill role-specific observations with zeros/defaults")
        code.append("        obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 7 dimensions (role_id already added)")
        code.append("")
        
        code.append("    # Verify we have exactly 56 dimensions")
        code.append("    # 48 (base) + 8 (role-specific) = 56")
        code.append("    assert len(obs) == 56, f'Expected 56 dimensions, got {len(obs)}'")
        code.append("    return obs")
        code.append("")
        
        return code

    def _generate_main_code(self):
        """Generate the main game loop code with corrected observation handling for 56 dimensions"""
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
        code.append("    ")
        code.append("    # Read all pod data")
        code.append("    pod_data_all = []")
        code.append("    for i in range(2):")
        code.append("        x, y, vx, vy, angle, next_checkpoint_id = [int(j) for j in input().split()]")
        code.append("        pod_data_all.append((x, y, vx, vy, angle, next_checkpoint_id))")
        code.append("    ")
        code.append("    # Read opponent data")
        code.append("    opponent_data = []")
        code.append("    for i in range(2):")
        code.append("        x, y, vx, vy, angle, next_checkpoint_id = [int(j) for j in input().split()]")
        code.append("        opponent_data.append((x, y, vx, vy, angle, next_checkpoint_id))")
        code.append("")
        
        # Process each pod
        code.append("    # Process each pod and store commands")
        code.append("    commands = []")
        code.append("    for pod_id in range(2):")
        code.append("        # Get observations (56 dimensions matching training)")
        code.append("        obs = get_observations(")
        code.append("            pod_data_all[pod_id], ")
        code.append("            opponent_data, ")
        code.append("            checkpoints, ")
        code.append("            boost_used, ")
        code.append("            pod_id")
        code.append("        )")
        code.append("")

        # Get model predictions based on role
        code.append("        # Get model predictions based on role")
        code.append("        if pod_id == 0:  # Runner")
        code.append("            angle_output, thrust, shield_prob, boost_prob = runner_forward(obs)")
        code.append("        else:  # Blocker")
        code.append("            angle_output, thrust, shield_prob, boost_prob = blocker_forward(obs)")
        code.append("")
        
        # Convert outputs to game commands - FIXED VERSION
        code.append("        # Convert model outputs to action format")
        code.append("        x, y, vx, vy, angle, next_checkpoint_id = pod_data_all[pod_id]")
        code.append("        next_checkpoint_x, next_checkpoint_y = checkpoints[next_checkpoint_id]")
        code.append("        ")
        code.append("        # The model outputs an angle adjustment in range [-1, 1]")
        code.append("        # Convert to degree adjustment [-18, 18] as in training")
        code.append("        angle_adjustment = angle_output * 18.0")
        code.append("        ")
        code.append("        # Calculate target angle")
        code.append("        base_angle_to_checkpoint = math.degrees(math.atan2(next_checkpoint_y - y, next_checkpoint_x - x))")
        code.append("        target_angle = base_angle_to_checkpoint + angle_adjustment")
        code.append("        ")
        code.append("        # Convert to target coordinates")
        code.append("        target_angle_rad = math.radians(target_angle)")
        code.append("        target_distance = 10000  # Fixed distance for target point")
        code.append("        target_x = x + math.cos(target_angle_rad) * target_distance")
        code.append("        target_y = y + math.sin(target_angle_rad) * target_distance")
        code.append("")

        # Determine thrust value - CORRECTED VERSION using special action probabilities
        code.append("        # Determine action based on model outputs")
        code.append("        # Priority: Shield > Boost > Normal Thrust")
        code.append("        if shield_prob > 0.5:")
        code.append("            thrust_command = 'SHIELD'")
        code.append("        elif boost_prob > 0.5 and not boost_used[pod_id]:")
        code.append("            thrust_command = 'BOOST'")
        code.append("            boost_used[pod_id] = True")
        code.append("        else:")
        code.append("            # Convert normalized thrust [0, 1] to game thrust [0, 100]")
        code.append("            thrust_value = max(0, min(100, int(thrust * 100)))")
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
    def validate_models(self):
        """Validate that models are properly loaded and compatible for export"""
        issues = []
        
        if self.runner_model is None:
            issues.append("Runner model is not loaded")
        
        if self.blocker_model is None:
            issues.append("Blocker model is not loaded")
        
        if self.runner_model is not None:
            try:
                # Test forward pass with dummy input (56 dimensions)
                dummy_input = torch.randn(1, self.network_config.get('observation_dim', 56))
                with torch.no_grad():
                    self.runner_model(dummy_input)
            except Exception as e:
                issues.append(f"Runner model forward pass failed: {e}")
        
        if self.blocker_model is not None and self.blocker_model != self.runner_model:
            try:
                # Test forward pass with dummy input (56 dimensions)
                dummy_input = torch.randn(1, self.network_config.get('observation_dim', 56))
                with torch.no_grad():
                    self.blocker_model(dummy_input)
            except Exception as e:
                issues.append(f"Blocker model forward pass failed: {e}")
        
        return issues

    def export(self, quantize=True, precision=3):
        """
        Export the model to a Python file for CodinGame submission
        
        Args:
            quantize: Whether to quantize weights
            precision: Number of decimal places to keep when quantizing
        """
        # Validate models before export
        validation_issues = self.validate_models()
        if validation_issues:
            raise ValueError(f"Model validation failed: {'; '.join(validation_issues)}")
        
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
        
        return {
            'output_path': str(self.output_path),
            'file_size_kb': file_size_kb,
            'quantized': quantize,
            'precision': precision,
            'runner_model_path': self._find_best_model_file_for_role("runner"),
            'blocker_model_path': self._find_best_model_file_for_role("blocker")
        }
