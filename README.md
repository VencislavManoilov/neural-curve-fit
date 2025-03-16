# Neural Network Curve Fitting

A simple implementation of neural networks to fit a sine wave using PyTorch.

## Description

This project demonstrates how a basic neural network can be trained to approximate a sine wave function. It visualizes the training process in real-time and shows the final result as a comparison between the true sine wave and the neural network's approximation.

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
2. Generate a sine wave dataset
3. Create and train a neural network to approximate the sine function
4. Display real-time training progress visualization
5. Show a final comparison plot when training is complete

## Customization

You can modify the following parameters in `main.py` to experiment:

- Network architecture: Change the layer sizes or add more layers in the `Net` class
- Training duration: Adjust the `epochs` variable (default: 5000)
- Visualization frequency: Change `display_freq` to update the plot more or less frequently
- Learning rate: Modify the `lr` parameter in the optimizer (default: 0.01)
- Data range: Change the range in `torch.linspace(-2 * np.pi, 2 * np.pi, 1000)` to fit different ranges of the sine function

## Requirements for GPU Acceleration

To use GPU acceleration (if available):

- NVIDIA GPU with CUDA support
- Compatible NVIDIA drivers
- PyTorch built with CUDA support

The script will automatically use GPU if available; otherwise, it will fall back to CPU.
