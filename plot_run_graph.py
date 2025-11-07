import matplotlib.pyplot as plt

# Assuming 'history' is the output of model.fit()
def plot_training_history(history, save_path="training_history.png"):
    """Plot and save the accuracy and loss graphs."""
    
    # Extract values
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Create figure
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label="Training Accuracy")
    plt.plot(epochs, val_acc, 'r*-', label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label="Training Loss")
    plt.plot(epochs, val_loss, 'r*-', label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path)  # Save the graph as an image
    plt.show()

def plot_training_history_torch(train_losses, train_accuracies, save_path="training_results.png"):
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy Over Epochs")
    plt.legend()

    plt.savefig(save_path)
    #plt.show()