import argparse
import wandb
import yaml
from Model1 import *
from Utils import data_download, preprocess, plot_images

def sweep_train():
    run = wandb.init()
    config = run.config
    dataset = "fashion_mnist"
    run_name = (
        f"hl_{config.num_layers}_bs_{config.batch_size}_ac_{config.activation}"
        f"_opt_{config.optimizer}_lr_{config.eta}"
    )
    wandb.run.name = run_name
    wandb.log({"run_name": run_name})

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
    parser.add_argument("--count", type=int, default=10, help="Number of sweep runs")
    args = parser.parse_args()

    with open("sweep_config.yaml") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, entity=args.entity, project=args.project)
    wandb.agent(sweep_id, function=sweep_train, count=args.count)
