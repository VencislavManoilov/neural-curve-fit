# Neural Network Curve Fitting

A flexible implementation of neural networks to fit various mathematical functions using PyTorch.

## Description

This project demonstrates how a basic neural network can be trained to approximate different mathematical functions. It allows users to select from multiple function types, customize training parameters, and visualizes both the training process in real-time and the final result with a loss history graph.

## Requirements

- Python 3.6 or higher
- PyTorch
- NumPy
- Matplotlib

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd neural-curve-fit
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install torch numpy matplotlib
   ```

## Usage

Run the script with:

```
python main.py
```

This will:
1. Check if CUDA is available and use it if possible
2. Present a menu of mathematical functions to approximate:
   - Sine Wave
   - Gaussian Curve
   - Absolute Value
   - Polynomial (0.5x³ - 4x² + 2x + 5)
   - Step Function
   - Sigmoid
3. Ask for training parameters (number of epochs and display frequency)
4. Create and train a neural network to approximate the selected function
5. Display real-time training progress visualization
6. Show a final comparison plot with the loss history when training is complete

## Customizing the Code

You can modify the following in `main.py`:

- **Add new functions**: Extend the `function_options` dictionary with additional mathematical functions
- **Network architecture**: Adjust the `hidden_size` parameter or modify the `Net` class structure
- **Learning rate**: Change the `lr` parameter in the optimizer (default: 0.01)
- **Data range**: Modify the range in `torch.linspace(-5, 5, 1000)` to change the domain for function approximation

## Requirements for GPU Acceleration

To use GPU acceleration (if available):

- NVIDIA GPU with CUDA support
- Compatible NVIDIA drivers
- PyTorch built with CUDA support

The script will automatically use GPU if available; otherwise, it will fall back to CPU.
