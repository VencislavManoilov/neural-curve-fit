import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
# from IPython.display import clear_output

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Generate sine wave data
x = torch.linspace(-2 * np.pi, 2 * np.pi, 1000).view(-1, 1).to(device)
y = torch.sin(x).to(device)

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss, and optimizer
model = Net().to(device)
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
        plt.title(f'Training Progress - Epoch {epoch}')
        x_cpu = x.cpu().detach().numpy()
        y_cpu = y.cpu().detach().numpy()
        pred_cpu = model(x).cpu().detach().numpy()
        
        plt.plot(x_cpu, y_cpu, label="True Sine Wave")
        plt.plot(x_cpu, pred_cpu, label="Neural Network Approximation", linestyle="dashed")
        plt.legend()
        plt.ylabel('y = sin(x)')
        plt.xlabel('x')
        
        plt.tight_layout()
        
        plt.pause(0.01)  # Small pause to update the figure

# Turn off interactive mode
plt.ioff()

# Final plot
plt.figure(figsize=(10, 5))
x_cpu = x.cpu().detach().numpy()
y_cpu = y.cpu().detach().numpy()
pred_cpu = model(x).cpu().detach().numpy()

plt.plot(x_cpu, y_cpu, label="True Sine Wave")
plt.plot(x_cpu, pred_cpu, label="Neural Network Approximation", linestyle="dashed")
plt.legend()
plt.title('Final Result')
plt.show()