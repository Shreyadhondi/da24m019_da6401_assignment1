import argparse

def args_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--wandb_project",default="da24m019_shreya_da6401_assignment1", help="Project name used to track experiments in Weights & Biases dashboard",type=str)
  parser.add_argument("--wandb_entity",default="shreyadhondi-indian-institute-of-technology-madras", help="Wandb Entity used to track experiments in the Weights & Biases dashboard",type=str)
  parser.add_argument("--dataset",default="fashion_mnist",choices = ["mnist", "fashion_mnist"], help="dataset to use for experiment",type=str)
  parser.add_argument("--epochs",default=20,help="Number of epochs to train neural network",type=int)
  parser.add_argument("--batch_size",default=64,help="Batch size used to train neural network",type=int)
  parser.add_argument("--loss",default="cross_entropy",choices=["mean_squared_error", "cross_entropy"], help="Loss Function to train neural network",type=str)
  parser.add_argument("--optimizer",default="sgd",choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="optimizer to train neural network",type=str)
  parser.add_argument("--learning_rate",default=0.1, help="Learning rate used to optimize model parameters",type=float)
  parser.add_argument("--momentum",default=0.9, help="Momentum used by momentum and nag optimizers",type=float)
  parser.add_argument("--beta",default=0.9, help="Beta used by rmsprop optimizer",type=float)
  parser.add_argument("--beta1",default=0.9, help="Beta1 used by adam and nadam optimizers",type=float)
  parser.add_argument("--beta2",default=0.9, help="Beta2 used by adam and nadam optimizers",type=float)
  parser.add_argument("--epsilon",default=0.000001, help="Epsilon used by optimizers",type=float)
  parser.add_argument("--weight_decay",default=.0, help="Weight decay used by optimizers",type=float)
  parser.add_argument("--weight_init",default='random',choices = ["random", "Xavier"], help="Weight initialization techniques",type=str)
  parser.add_argument("--num_layers",default=2,help="Number of hidden layers used in feedforward neural network",type=int)
  parser.add_argument("--hidden_size",default=32,help="Number of hidden neurons in a feedforward layer",type=int)
  parser.add_argument("--activation",default='ReLu',choices = ["identity", "sigmoid", "tanh", "ReLu"], help="Activation functions",type=str)
  args = parser.parse_args()
  return args
