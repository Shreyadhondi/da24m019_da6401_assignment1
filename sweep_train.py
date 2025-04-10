import argparse
import wandb
from Model1 import *

def sweep_train():
    wandb.init()
    config = wandb.config

    wandb.run.name = f"lr_{config.eta}_opt_{config.optimizer}_epoch_{config.epochs}_bs_{config.batch_size}_act_{config.activation}"

    X_train, Y_train, X_test, Y_test = data_download('fashion_mnist')
    plot_images(X_train, Y_train)
    x_val, y_ohv_val, x_train, y_ohv_train, X_test, Y_ohv_test = preprocess(X_train, Y_train, X_test, Y_test)

    model = FFNN(
        dataset='fashion_mnist',
        weight_init=config.weight_init,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        class_size=10,
        activation=config.activation,
        loss='cross_entropy',
        optimizer=config.optimizer,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        eta=config.eta,
        momentum=0.9,
        batch_size=config.batch_size,
        eps=1e-8,
        beta=0.9,
        beta1=0.9,
        beta2=0.9
    )

    model.fit(x_val, y_ohv_val, x_train, y_ohv_train, X_test, Y_ohv_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default="shreyadhondi-indian-institute-of-technology-madras")
    parser.add_argument("--project", type=str, default="da24m019_shreya_da6401_assignment1")
    args = parser.parse_args()

    wandb.login()
    sweep_train()
