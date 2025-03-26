import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def plot_learning_curve(num_epochs, train_losses, train_accuracies, test_accuracies, save_path='./learning_curve.png'):
    """Plot and save learning curves for training and evaluation
    
    Args:
        num_epochs: Number of epochs completed
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        test_accuracies: List of test accuracies
        save_path: Path to save the plot
    """
    def smooth_curve(values, window=5, poly=2):
        if len(values) < window:
            return values  # Not enough points to smooth
        return savgol_filter(values, window, poly)

    epochs = range(1, num_epochs + 1)
    
    # Apply smoothing if enough data points
    if len(train_losses) >= 5:
        smoothed_train_losses = smooth_curve(train_losses)
        smoothed_train_accuracies = smooth_curve(train_accuracies)
        smoothed_test_accuracies = smooth_curve(test_accuracies)
    else:
        smoothed_train_losses = train_losses
        smoothed_train_accuracies = train_accuracies
        smoothed_test_accuracies = test_accuracies

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, smoothed_train_losses, label='Training Loss', linestyle='-', linewidth=2, color='tab:red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, smoothed_train_accuracies, label='Training Accuracy', linestyle='-', linewidth=2, color='tab:blue')
    plt.plot(epochs, smoothed_test_accuracies, label='Test Accuracy', linestyle='-', linewidth=2, color='tab:green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory