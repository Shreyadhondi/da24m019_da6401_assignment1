# da6401_assignment1/data.py

import numpy as np
import matplotlib.pyplot as plt
import wandb
from keras.datasets import fashion_mnist

def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    # Normalize images to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return (X_train, y_train), (X_test, y_test)

def plot_sample_images(X, y):
    num_classes = 10
    samples = []
    
    # Pick one image per class
    for label in range(num_classes):
        idx = np.where(y == label)[0][0]
        samples.append(X[idx])
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, img in enumerate(samples):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Class {i}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Log the figure to wandb
    wandb.log({'sample_images': wandb.Image(fig)})
    
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    wandb.init(project="da24m019_shreya_da6401_assignment1", entity="shreyadhondi-indian-institute-of-technology-madras")
    (X_train, y_train), _ = load_fashion_mnist()
    plot_sample_images(X_train, y_train)
