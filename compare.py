"""
This code is to compare cross-entropy loss with the mean_squared_error.
"""

import wandb
from Model1 import *
from Utils import data_download, preprocess, plot_images

def run_experiment(loss_type):
    wandb.init(
        project="da24m019_shreya_da6401_assignment1",
        name=f"loss_compare_{loss_type}",
        config={
            "epochs": 5,
            "batch_size": 64,
            "loss": loss_type,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "num_layers": 2,
            "hidden_size": 64,
            "activation": "ReLu",
            "weight_init": "Xavier",
            "weight_decay": 0,
        }
    )
    config = wandb.config

    # Load and preprocess data
    X_train, Y_train, X_test, Y_test = data_download("fashion_mnist")
    x_val, y_ohv_val, x_train, y_ohv_train, X_test, Y_ohv_test = preprocess(X_train, Y_train, X_test, Y_test)

    # Initialize model
    model = FFNN(
        dataset="fashion_mnist",
        weight_init=config.weight_init,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        class_size=10,
        activation=config.activation,
        loss=config.loss,
        optimizer=config.optimizer,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        eta=config.learning_rate,
        momentum=0.9,
        batch_size=config.batch_size,
        eps=1e-8,
        beta=0.9,
        beta1=0.9,
        beta2=0.9
    )

    # Train model
    model.fit(x_val, y_ohv_val, x_train, y_ohv_train, X_test, Y_ohv_test)

if __name__ == "__main__":
    run_experiment("cross_entropy")
    wandb.finish()
    run_experiment("mean_squared_error")
