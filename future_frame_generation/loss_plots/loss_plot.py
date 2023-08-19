import pickle
import matplotlib.pyplot as plt


# Loadtraining and validation losses

with open("losses.pkl", 'rb') as f:
    train_losses = pickle.load(f)

with open("losses_val.pkl", 'rb') as f:
    val_losses = pickle.load(f)


# Create x-axis values (epochs or iterations)
epochs = range(1, len(train_losses) + 1)

# Plot training and validation loss curves
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()  # Show legend with labels

# Show the plot
plt.savefig("loss_plot.png")
