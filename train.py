from Model1 import FFNN
from Utils import data_download,preprocess,plot_images
import argparse
import wandb

def main(args):
  """
  runs all the functions
  """
  # Construct the run name first
  run_name = (
      f"hl_{args.num_layers}_bs_{args.batch_size}_ac_{args.activation}"
      f"_opt_{args.optimizer}_lr_{args.learning_rate}"
  )

  # Now initialize wandb with the name
  wandb.init(
      entity=args.wandb_entity,
      project=args.wandb_project,
      name=run_name
  )
  
  #we will now log the run name 
  wandb.log({"run_name": wandb.run.name})
 
  X_train, Y_train, X_test, Y_test = data_download(args.dataset)
  plot_images(X_train,Y_train)
  x_val,y_ohv_val,x_train,y_ohv_train,X_test,Y_ohv_test = preprocess(X_train, Y_train, X_test, Y_test)
  model = FFNN(args.dataset,args.weight_init,args.num_layers,args.hidden_size,10,args.activation,args.loss,args.optimizer,args.weight_decay,args.epochs,args.learning_rate,args.momentum,args.batch_size,args.epsilon,args.beta,args.beta1,args.beta2)
  model.fit(x_val,y_ohv_val,x_train,y_ohv_train,X_test,Y_ohv_test)



if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--wandb_project",default="da24m019_shreya_da6401_assignment1", help="Project name used to track experiments in Weights & Biases dashboard",type=str)
  parser.add_argument("--wandb_entity",default="shreyadhondi-indian-institute-of-technology-madras", help="Wandb Entity used to track experiments in the Weights & Biases dashboard",type=str)
  parser.add_argument("--dataset",default="fashion_mnist",choices = ["mnist", "fashion_mnist"], help="dataset to use for experiment",type=str)
  parser.add_argument("--epochs",default=1,help="Number of epochs to train neural network",type=int)
  parser.add_argument("--batch_size",default=64,help="Batch size used to train neural network",type=int)
  parser.add_argument("--loss",default="cross_entropy",choices=["mean_squared_error", "cross_entropy"], help="Loss Function to train neural network",type=str)
  parser.add_argument("--optimizer",default="nag",choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="optimizer to train neural network",type=str)
  parser.add_argument("--learning_rate",default=0.02853927530143583, help="Learning rate used to optimize model parameters",type=float)
  parser.add_argument("--momentum",default=0.9, help="Momentum used by momentum and nag optimizers",type=float)
  parser.add_argument("--beta",default=0.9, help="Beta used by rmsprop optimizer",type=float)
  parser.add_argument("--beta1",default=0.9, help="Beta1 used by adam and nadam optimizers",type=float)
  parser.add_argument("--beta2",default=0.9, help="Beta2 used by adam and nadam optimizers",type=float)
  parser.add_argument("--epsilon",default=0.000001, help="Epsilon used by optimizers",type=float)
  parser.add_argument("--weight_decay",default=0.0004801100588819423, help="Weight decay used by optimizers",type=float)
  parser.add_argument("--weight_init",default='Xavier',choices = ["random", "Xavier"], help="Weight initialization techniques",type=str)
  parser.add_argument("--num_layers",default=2,help="Number of hidden layers used in feedforward neural network",type=int)
  parser.add_argument("--hidden_size",default=64,help="Number of hidden neurons in a feedforward layer",type=int)
  parser.add_argument("--Num_Neurons_per_Layer",default=[64,64],help="Number of hidden neurons in a feedforward layer",type=list)
  parser.add_argument("--activation",default='ReLu',choices = ["identity", "sigmoid", "tanh", "ReLu"], help="Activation functions",type=str)
  args = parser.parse_args()
  main(args)
