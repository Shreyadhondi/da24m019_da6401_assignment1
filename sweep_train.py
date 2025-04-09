from Model1 import *
#from Utils import train, test
import argparse
import wandb
import yaml

def sweep_train(sweep_config=None):
    user = "shreyadhondi-indian-institute-of-technology-madras"
    project = "da24m019_shreya_da6401_assignment1"
    display_name = "da24m019"

    wandb.init(entity=user, project=project, name=display_name, config = sweep_config)



    config = wandb.config
    wandb.run.name = "lr_" + str(config.eta) + "_opt_" + str(config.optimizer) + "_epoch_" + str(config.epochs) + "_bs_" + str(config.batch_size) + "_act_" + str(config.activation)



    X_train, Y_train, X_test, Y_test = data_download('fashion_mnist')
    plot_images(X_train,Y_train)
    x_val,y_ohv_val,x_train,y_ohv_train,X_test,Y_ohv_test = preprocess(X_train, Y_train, X_test, Y_test)
    print(x_val.shape,y_ohv_val.shape,x_train.shape,y_ohv_train.shape,X_test.shape,Y_ohv_test.shape)

    model = FFNN('fashion_mnist',config.weight_init,config.num_layers,config.hidden_size,10,config.activation,'cross_entropy',config.optimizer,config.weight_decay,config.epochs,config.eta,0.9,config.batch_size,1e-8,0.9,0.9,0.9)
    model.fit(x_val,y_ohv_val,x_train,y_ohv_train,X_test,Y_ohv_test)



if __name__=="__main__":
    with open("sweep_config.yaml") as f:
        sweep_config = yaml.safe_load(f)
    print(sweep_config)
    sweep_id = wandb.sweep(sweep_config,project="da24m019_shreya_da6401_assignment1")
    wandb.agent(sweep_id,function=sweep_train, count = 10)
