# Sebulba Pod Trainer

A reinforcement learning training application for CodinGame's "Mad Pod Racing" challenge using PyTorch and Proximal Policy Optimization (PPO). This project trains neural network agents to compete in pod racing through deep reinforcement learning.

![Sebulba Pod Trainer UI](screenshots/main_interface.png)
*Main training interface with real-time visualization*

## 🏎️ Project Overview

This CS50X final project implements a sophisticated AI training system for CodinGame's Mad Pod Racing challenge. While I've achieved Gold rank using a heuristic approach ([`pods/heuristic_pod.py`](pods/heuristic_pod.py)), this project explores deep reinforcement learning to potentially achieve even higher performance through neural network-based pod control.

The trainer features a full GUI application built with wxPython, real-time training visualization, multi-GPU support, and model export capabilities for CodinGame submission.

## 🎯 Features

### Training & AI
- **Proximal Policy Optimization (PPO)** for stable reinforcement learning
- **Multi-GPU support** for faster training with parallel environments
- **Mixed precision training** for improved performance
- **Optimized race environment** with vectorized PyTorch operations
- **League training system** for self-play and opponent diversity
- **Configurable neural network architectures**

### Visualization & Monitoring
- **Real-time training metrics** (rewards, losses, convergence)
- **Interactive race visualization** with pod trajectories and collision detection
- **Multi-environment monitoring** for parallel training sessions
- **Training progress tracking** with detailed logging

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

### Command Line Training (Advanced)
```bash
python -m sebulba_pod_trainer.training.trainer --config config.json
```

## 🧠 Technical Architecture

### Reinforcement Learning Approach
This project uses **Proximal Policy Optimization (PPO)**, a state-of-the-art policy gradient method that balances exploration and exploitation while maintaining training stability. PPO works by:

1. **Policy Network**: Outputs actions (steering angle, thrust) given game state observations
2. **Value Network**: Estimates expected future rewards for current state
3. **Advantage Estimation**: Uses Generalized Advantage Estimation (GAE) for stable learning
4. **Clipped Objectives**: Prevents destructive policy updates through clipping

### Neural Network Architecture
- **Configurable hidden layers** with various activation functions (ReLU, Tanh, Sigmoid)
- **Separate policy and value heads** for actor-critic learning
- **Special action handling** for SHIELD and BOOST mechanics
- **Observation normalization** for stable training

### Environment Features
- **Vectorized PyTorch operations** for efficient batch processing
- **Realistic physics simulation** including collisions and momentum
- **Checkpoint-based reward system** encouraging race completion
- **Multi-agent support** with 4 pods (2 per player)

### Observation Space (27 dimensions)
- Pod position and velocity (normalized)
- Relative positions to next/future checkpoints
- Opponent pod positions and velocities
- Game state (checkpoint progress, shield cooldown, boost availability)

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

## 🏁 CodinGame Integration

The trainer includes an export system that generates self-contained Python code for CodinGame submission:

```bash
# Export trained model
python -m sebulba_pod_trainer.export.model_exporter models/trained_model/ output.py
```

**Note**: Early models were too large for CodinGame's editor constraints. Current focus is on compact architectures suitable for the platform.

## 🛠️ Development Status

This project is actively in development as a CS50X final project. It represents a significant evolution from an earlier basic training script, incorporating:

- Modular neural network architecture
- Advanced GUI interface
- Multi-GPU training capabilities
- Real-time visualization systems

### Known Limitations
- UI not yet tested on Windows
- Model size optimization ongoing for CodinGame constraints
- Training hyperparameter tuning in progress

## 📚 Acknowledgments

- **CS50X Course**: Harvard's Introduction to Computer Science
- **External References**: Portions inspired by [gymnasium-search-race](https://github.com/Quentin18/gymnasium-search-race)
- **AI Assistance**: This project was developed with extensive assistance from AI tools for:
  - Code optimization and architectural design
  - Debugging complex multi-GPU training issues
  - PyTorch tensor operation optimization
  - GUI event handling and threading problems
  - Memory management and performance tuning
- **CodinGame**: For providing the Mad Pod Racing challenge environment

## 🎥 Demo

[Video demonstration will be included as required by Harvard's CS50X submission guidelines]

## 📖 Learning Resources

### For Reinforcement Learning Beginners
- **Policy Gradient Methods**: The neural network learns by trying actions and reinforcing successful ones
- **Actor-Critic**: Combines policy learning (actor) with value estimation (critic) for stable training
- **PPO**: Prevents destructive updates through "clipped" policy changes
- **Experience Replay**: Learns from batches of past experiences for sample efficiency

### Useful Papers
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

## 🤝 Contributing

This is a CS50X final project, but suggestions and feedback are welcome! The codebase is designed to be modular and extensible.

---

*This project demonstrates the application of modern deep reinforcement learning techniques to competitive programming challenges, showcasing both the potential and current limitations of AI in strategic game environments.*
