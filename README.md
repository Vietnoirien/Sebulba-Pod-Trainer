# Sebulba Pod Trainer

A reinforcement learning training application for CodinGame's "Mad Pod Racing" challenge using PyTorch and Proximal Policy Optimization (PPO). This project trains neural network agents to compete in pod racing through deep reinforcement learning with specialized role-based training.

![Sebulba Pod Trainer - Training Interface](screenshots/training.png)
*Main training interface with real-time parameter monitoring and control*

## 🏎️ Project Overview

This CS50X final project implements a sophisticated AI training system for CodinGame's Mad Pod Racing challenge. While I've achieved Gold rank using a heuristic approach ([`pods/heuristic_pod.py`](pods/heuristic_pod.py)), this project explores deep reinforcement learning to potentially achieve even higher performance through neural network-based pod control.

The trainer features a full GUI application built with wxPython, real-time training visualization, multi-GPU support, role-based training system, and model export capabilities for CodinGame submission.

## 🎯 Features

### Training & AI
- **Proximal Policy Optimization (PPO)** for stable reinforcement learning
- **Role-based training system** with specialized runner and blocker strategies
- **Advanced parallel training** with parameter server and gradient aggregation
- **Shared experience buffers** across multiple training environments
- **Continued training support** with flexible model loading strategies
- **Anti-stall timeout prevention** system to discourage exploitation
- **Multi-GPU support** for faster training with parallel environments
- **Mixed precision training** with NaN detection and recovery
- **Optimized race environment** with vectorized PyTorch operations
- **Dynamic track generation** with randomizable checkpoint layouts
- **League training system** for self-play and opponent diversity
- **Configurable neural network architectures**

![Neural Network Architecture Designer](screenshots/layers.png)
*Neural network architecture configuration with layer-by-layer design*

### Visualization & Monitoring
- **Real-time training metrics** (rewards, losses, convergence)
- **Interactive race visualization** with pod trajectories and collision detection
- **Multi-environment monitoring** for parallel training sessions
- **Training progress tracking** with detailed logging
- **Timeout exploitation detection** and success rate monitoring

![Real-time Training Visualization](screenshots/visualizations.png)
*Comprehensive training metrics and race visualization dashboard*

### User Interface
- **Full GUI application** with tabbed interface
- **Model architecture designer** with visual configuration
- **Training parameter tuning** interface
- **Race simulation and analysis** tools
- **Model export system** for CodinGame integration

## 🚀 Installation

### Prerequisites
- Python 3.11.2+ (tested on Debian stable)
- NVIDIA GPU with CUDA support (recommended)
- GTK 3.0 development libraries (for wxPython)


### Install the Package
```bash
git clone https://github.com/Vietnoirien/Sebulba-Pod-Trainer
cd sebulba_pod_trainer
pip install -e .
```

### Dependencies
- PyTorch (with CUDA support recommended)
- wxPython (requires GTK 3.0 dev libraries)
- NumPy
- Matplotlib

## 🎮 Usage

### Launch the GUI Application
```bash
sebulba
```

### Quick Start Training
1. **Configure Model Architecture**: Use the "Model Architecture" tab to design your neural network
2. **Set Training Parameters**: Configure batch sizes, learning rates, and PPO parameters in "Training Configuration"
3. **Start Training**: Click "Training" → "Start Training" or use Ctrl+T
4. **Monitor Progress**: Watch real-time metrics in the "Visualization" tab
5. **Export Model**: Use the "Export" tab to generate CodinGame-compatible code

### Continued Training Workflow
1. **Load Previous Model**: Use the model loader to continue from latest, best, or specific iteration
2. **Resume Training**: Maintain training progress and hyperparameters from previous sessions
3. **Iterative Improvement**: Build upon previous training results for enhanced performance

### Command Line Training (Advanced)
```bash
python -m sebulba_pod_trainer.training.trainer --config config.json
```

## 🏆 League Training System

![League Training Management](screenshots/leagues.png)
*Advanced league system for self-play training and opponent diversity*

The trainer includes a sophisticated league system that enables:
- **Self-play training** against previous model versions
- **Opponent diversity** through multiple skill levels
- **Progressive difficulty** as models improve
- **Tournament-style evaluation** of different strategies

## 🧠 Technical Architecture

### Reinforcement Learning Approach
This project uses **Proximal Policy Optimization (PPO)**, a state-of-the-art policy gradient method that balances exploration and exploitation while maintaining training stability. PPO works by:

1. **Policy Network**: Outputs actions (steering angle, thrust) given game state observations
2. **Value Network**: Estimates expected future rewards for current state
3. **Advantage Estimation**: Uses Generalized Advantage Estimation (GAE) for stable learning
4. **Clipped Objectives**: Prevents destructive policy updates through clipping

### Advanced Parallel Training System
The trainer implements sophisticated multi-GPU coordination:

#### Parameter Server Architecture
- **Centralized parameter management** for consistent model updates across workers
- **Gradient aggregation** from multiple training environments
- **Synchronized model updates** to prevent training divergence

#### Shared Experience System
- **Cross-environment experience sharing** for improved sample efficiency
- **Diverse training scenarios** with different track layouts per worker
- **Enhanced exploration** through varied environmental conditions

#### Training Stability Features
- **NaN detection and recovery** mechanisms for robust training
- **Mixed precision training** with automatic fallback to FP32
- **Entropy monitoring** to detect and handle training instabilities

### Role-Based Training System
The environment supports specialized training for different pod roles:

#### Runner Pods (Pod 0 in each team)
- **Primary objective**: Complete the race as quickly as possible
- **Reward focus**: Speed maintenance, checkpoint completion, efficient thrust usage
- **Penalties**: Low thrust usage, stalling, timeout without checkpoint progress

#### Blocker Pods (Pod 1 in each team)
- **Primary objective**: Interfere with opponent progress while supporting teammate
- **Reward focus**: Blocking opponent paths, strategic positioning, team coordination
- **Penalties**: Excessive shielding, remaining stationary, poor team support

### Anti-Stall System
- **Progressive timeout penalties** to prevent stalling exploitation
- **Success rate monitoring** to distinguish active racing from timeout abuse
- **Dynamic penalty adjustment** based on racing behavior patterns

### Model Loading & Continuation
- **Flexible loading strategies**: Latest checkpoint, best performance, or specific iteration
- **Training state preservation**: Maintains optimizer states and training progress
- **Seamless workflow**: Easy continuation of interrupted training sessions

### Neural Network Architecture
- **Configurable hidden layers** with various activation functions (ReLU, Tanh, Sigmoid)
- **Separate policy and value heads** for actor-critic learning
- **Role-aware processing** for specialized pod behaviors
- **Special action handling** for SHIELD and BOOST mechanics
- **Observation normalization** for stable training

### Environment Features
- **Vectorized PyTorch operations** for efficient batch processing
- **Realistic physics simulation** including collisions and momentum
- **Dynamic track generation** with randomizable checkpoint counts (3-8 checkpoints)
- **Checkpoint-based reward system** encouraging race completion
- **Multi-agent support** with 4 pods (2 per player) and role specialization

### Observation Space (56 dimensions)
#### Base Observations (48 dimensions)
- **Pod-specific data** (18 dims): Position, velocity, angle, speed, progress, abilities
- **Opponent data** (30 dims): 3 opponents × 10 dimensions each (relative position, velocity, progress, abilities)

#### Role-Specific Observations (8 dimensions)
- **Role identifier**: Runner (0) or Blocker (1)
- **Timeout progress**: Normalized turns since last checkpoint
- **Runner-specific**: Opponent distances, progress comparison, speed metrics
- **Blocker-specific**: Teammate coordination, blocking opportunities, strategic positioning

### Action Space
- **Steering angle**: Continuous value [-1, 1] (±18° adjustment)
- **Thrust control**: Continuous [0, 1] for thrust, or special values for SHIELD/BOOST

## 📊 Performance

### Hardware Testing
- **RTX 3060**: ~1 second per batch size (32 batch size, 32 mini-batch size)
- **GTX 1060**: ~1 second per batch size (32 batch size, 32 mini-batch size)
- **Multi-GPU scaling**: Linear improvement with additional GPUs

### Training Recommendations
- **Batch size**: 32-64 for single GPU, scale with available VRAM
- **Mini-batch size**: 16-32 for stable gradients
- **Learning rate**: 3e-4 (Adam optimizer default)
- **Training time**: 1000+ iterations for convergent policies

### Known Training Challenges
- **Early NaN issues**: Shared experience across diverse track layouts can cause entropy instabilities
- **Training stability**: Requires careful monitoring and automatic recovery mechanisms
- **Memory management**: Large shared experience buffers require significant VRAM

## 🏁 CodinGame Integration

![Model Export System](screenshots/export.png)
*Streamlined model export interface for CodinGame submission*

The trainer includes an export system that generates self-contained Python code for CodinGame submission:

```bash
# Export trained model
python -m sebulba_pod_trainer.export.model_exporter models/trained_model/ output.py
```

**Note**: Early models were too large for CodinGame's editor constraints. Current focus is on compact architectures suitable for the platform.

## 🛠️ Development Status

This project is actively in development as a CS50X final project. Recent major improvements include:

- **Advanced parallel training architecture** with parameter server and shared experience
- **Robust model loading system** for continued training workflows
- **Enhanced anti-stall mechanisms** to prevent timeout exploitation
- **Role-based training system** for specialized pod behaviors
- **Expanded observation space** (56 dimensions) with role-specific information
- **Enhanced reward systems** tailored for runner vs blocker strategies
- **Dynamic track generation** for improved training diversity
- **Advanced GUI interface** with real-time visualization
- **Training stability improvements** with NaN detection and recovery

### Known Limitations
- UI not yet tested on Windows
- Model size optimization ongoing for CodinGame constraints
- Training hyperparameter tuning in progress for role-based system
- Early entropy instabilities with shared experience across diverse environments

## 📚 Acknowledgments

- **CS50X Course**: Harvard's Introduction to Computer Science
- **External References**: Portions inspired by [gymnasium-search-race](https://github.com/Quentin18/gymnasium-search-race)
- **AI Assistance**: This project was developed with extensive assistance from AI tools for:
  - Code optimization and architectural design
  - Debugging complex multi-GPU training issues
  - PyTorch tensor operation optimization
  - GUI event handling and threading problems
  - Memory management and performance tuning
  - Role-based reward system design
  - Parameter server and gradient aggregation implementation
  - README.md writing and formatting
- **CodinGame**: For providing the Mad Pod Racing challenge environment

## 🎥 Demo

[Video demonstration will be included as required by Harvard's CS50X submission guidelines]

## 📖 Learning Resources

### For Reinforcement Learning Beginners
- **Policy Gradient Methods**: The neural network learns by trying actions and reinforcing successful ones
- **Actor-Critic**: Combines policy learning (actor) with value estimation (critic) for stable training
- **PPO**: Prevents destructive updates through "clipped" policy changes
- **Multi-Agent Training**: Coordinating multiple agents with different roles and objectives
- **Distributed Training**: Parameter servers and gradient aggregation for scalable learning
- **Experience Replay**: Learns from batches of past experiences for sample efficiency

### Useful Papers
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

## 🤝 Contributing

This is a CS50X final project, but suggestions and feedback are welcome! The codebase is designed to be modular and extensible.

---

*This project demonstrates the application of modern deep reinforcement learning techniques to competitive programming challenges, showcasing both the potential and current limitations of AI in strategic game environments with multi-agent coordination and distributed training.*
