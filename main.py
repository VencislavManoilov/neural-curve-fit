import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import platform

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
else:
    print(platform.processor())

# Function definitions
def sine_wave(x):
    return torch.sin(x)

def gaussian_curve(x):
    return torch.exp(-(x**2) / 2) / torch.sqrt(torch.tensor(2 * np.pi))

def absolute_value(x):
    return torch.abs(x)

def polynomial(x):
    return 0.5 * x**3 - 4 * x**2 + 2 * x + 5

def step_function(x):
    return (x > 0).float()

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Dictionary of available functions
function_options = {
    '1': {'name': 'Sine Wave', 'function': sine_wave},
    '2': {'name': 'Gaussian Curve', 'function': gaussian_curve},
    '3': {'name': 'Absolute Value', 'function': absolute_value},
    '4': {'name': 'Polynomial (0.5x³ - 4x² + 2x + 5)', 'function': polynomial},
    '5': {'name': 'Step Function', 'function': step_function},
    '6': {'name': 'Sigmoid', 'function': sigmoid},
}

# Display options to the user
print("\nSelect a function to fit:")
for key, value in function_options.items():
    print(f"{key}: {value['name']}")

# Get user selection
selection = input("Enter your choice (1-6): ")
while selection not in function_options:
    print("Invalid selection. Please try again.")
    selection = input("Enter your choice (1-6): ")

selected_function = function_options[selection]
print(f"\nFitting neural network to {selected_function['name']}...")

# Generate data for the selected function
x = torch.linspace(-5, 5, 1000).view(-1, 1).to(device)
y = selected_function['function'](x).to(device)

# Define a simple neural network
class Net(nn.Module):
    def __init__(self, hidden_size=64):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss, and optimizer
hidden_size = 64 if selection in ['4', '5'] else 32  # More neurons for complex functions
model = Net(hidden_size=hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Set up the plot
plt.figure(figsize=(10, 6))
plt.ion()  # Turn on interactive mode

# Track loss history for plotting
loss_history = []

# Train the model
epochs = 5000
display_freq = 10  # How often to update the plot

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if epoch % display_freq == 0:
        # Clear previous plot
        plt.clf()
        
        # Plot the current function approximation
        plt.subplot(1, 1, 1)
        plt.title(f'Training Progress - {selected_function["name"]} - Epoch {epoch}')
        x_cpu = x.cpu().detach().numpy()
        y_cpu = y.cpu().detach().numpy()
        pred_cpu = model(x).cpu().detach().numpy()
        
        plt.plot(x_cpu, y_cpu, label=f"True {selected_function['name']}")
        plt.plot(x_cpu, pred_cpu, label="Neural Network Approximation", linestyle="dashed")
        plt.legend()
        plt.ylabel('y = sin(x)')
        plt.xlabel('x')

        plt.tight_layout()

        plt.pause(0.01)  # Small pause to update the figure

# Turn off interactive mode
plt.ioff()

# Final plot
plt.figure(figsize=(10, 8))
x_cpu = x.cpu().detach().numpy()
y_cpu = y.cpu().detach().numpy()
pred_cpu = model(x).cpu().detach().numpy()

# Function plot
plt.subplot(2, 1, 1)
plt.plot(x_cpu, y_cpu, label=f"True {selected_function['name']}")
plt.plot(x_cpu, pred_cpu, label="Neural Network Approximation", linestyle="dashed")
plt.legend()
plt.title(f'Final Result - {selected_function["name"]}')

# Loss plot
plt.subplot(2, 1, 2)
plt.plot(loss_history)
plt.yscale('log')
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss (Log Scale)')

plt.tight_layout()
plt.show()