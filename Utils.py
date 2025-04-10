from tensorflow.keras.datasets import mnist, fashion_mnist
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import wandb

def plot_images(features, labels):
    """
    Creates plot of one sample from each class

    Parameters:
    ----------
        features (numpy.ndarray): Image samples in the form of numpy array
        labels (numpy.ndarray): Image labels in the form of numpy array

    Returns:
    -------
        None
    """


    label_dict = { 0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9: 'Ankle boot'}
    label_values = list(label_dict.keys())
    pixels = []
    lbls = []
    for i in label_values:
        pixels.append(features[np.random.choice(np.where(labels==i)[0])])
        lbls.append(label_dict[i])
    wandb.log({"examples": [wandb.Image(img,caption=lbl) for img,lbl in zip(pixels,lbls)]}) 

# def plot_images(features,labels):
#     """
#     Creates plot of one sample from each class

#     Parameters:
#     ----------
#         features (numpy.ndarray): Image samples in the form of numpy array
#         labels (numpy.ndarray): Image labels in the form of numpy array

#     Returns:
#     -------
#         None
#     """

#     label_dict = { 0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9: 'Ankle boot'}
#     label_values = list(label_dict.keys())
#     plt.figure(figsize = (8,8))
#     plt.tight_layout()
#     for i in label_values:
#         j = np.random.choice(np.where(labels==i)[0])
#         plt.subplot(2, 5, i+1)
#         plt.imshow(features[j])
#         plt.title(label_dict[i])
#         plt.axis("off")

def create_confusion_matrix(y_actual, y_pred):
    """
    Creates confusion matrix plot

    Parameters:
    ----------
        y_actual (numpy.ndarray): Actual output
        y_pred (numpy.ndarray): predicted output

    Returns:
    -------
        None
    """
    # conf_matrix = confusion_matrix(y_actual, y_pred)
    # conf_matrix_df = pd.DataFrame(conf_matrix,
    #               index = ['T-shirt/top','Trouser','Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    #               columns = ['T-shirt/top','Trouser','Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])

    all_labels = ['T-shirt/top','Trouser','Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    wandb.log({"Confusion_matrix_fashion_mnist" : wandb.plot.confusion_matrix(preds=y_pred, y_true=y_actual,class_names=all_labels)})

    # plt.figure(figsize=(10,10))
    # sns.heatmap(conf_matrix_df, annot=True, fmt="d")
    # plt.title('Confusion Matrix for fashion_mnist dataset')
    # plt.ylabel('Actual Values')
    # plt.xlabel('Predicted Values')
    # plt.savefig("confusion_matrix.png")

def data_download(dataset):
    """
    Downloads data and return test and trains data samples

    Parameters:
    ----------
        dataset (str): Dataset name. fashion_mnist or mnist.

    Returns:
    -------
        numpy.ndarray : Train features
        numpy.ndarray : Train Labels
        numpy.ndarray : Test features
        numpy.ndarray : Test Labels
    """

    if dataset == "fashion_mnist":
        (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    elif dataset == "mnist":
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    return X_train, Y_train, X_test, Y_test

def preprocess(X_train, Y_train, X_test, Y_test):
    """
    Flattens the pixels and Normalizes them. Splits training data to training and validation data(10%). Converts the labels to one-hot vectors

    Parameters:
    ----------
        X_train(numpy.ndarray) : Train features
        Y_train(numpy.ndarray) : Train Labels
        X_test(numpy.ndarray) : Test features
        Y_test(numpy.ndarray) : Test Labels

    Returns:
    -------
        x_val(numpy.ndarray) : Flattened and Normalized Validation features
        y_ohv_val(numpy.ndarray): Validation Labels in 1-hot vector form
        x_train(numpy.ndarray) : Flattened and Normalized Train features
        y_ohv_train(numpy.ndarray) : Train Labels in 1-hot vector form
        X_test(numpy.ndarray) : Flattened and Normalized Test features
        Y_ohv_test(numpy.ndarray) : Test Labels n 1-hot vector form

    """

    X_train = X_train.reshape(X_train.shape[0],-1,1)/255
    X_test = X_test.reshape(X_test.shape[0],-1,1)/255
    Y_ohv_train = np.zeros((Y_train.size, Y_train.max() + 1))
    Y_ohv_train[np.arange(Y_train.size), Y_train] = 1
    Y_ohv_train = Y_ohv_train.reshape((Y_ohv_train.shape[0],Y_ohv_train.shape[1],1))

    Y_ohv_test = np.zeros((Y_test.size, Y_test.max() + 1))
    Y_ohv_test[np.arange(Y_test.size), Y_test] = 1
    Y_ohv_test = Y_ohv_test.reshape((Y_ohv_test.shape[0],Y_ohv_test.shape[1],1))

    num_train_data_points = X_train.shape[0]
    len_validation = int(0.1*num_train_data_points)
    x_val = X_train[:len_validation]
    x_train = X_train[len_validation:]
    y_ohv_val =  Y_ohv_train[:len_validation]
    y_ohv_train = Y_ohv_train[len_validation:]
    return x_val,y_ohv_val,x_train,y_ohv_train,X_test,Y_ohv_test

def plot_metrics(epochs,Training_Loss,Training_Accuracy,Validation_Loss,Validation_Accuracy):
    """
    Plots performance metrics

    Parameters:
    ----------
        epochs(int) : Number of epochs
        Training_Loss(float) : Training Loss of the epoch
        Training_Accuracy(float) : Training Accuracy of the epoch
        Validation_Loss(float) : Validation Loss of the epoch
        Validation_Accuracy(float) : Validation Accuracy of the epoch

    Returns:
    -------
        None
    """

    plt.figure(figsize = (8,8))
    e = range(1,epochs+1)
    plt.subplot(1, 2, 1)
    plt.plot(e,Training_Loss.values(),label='Training_Loss')
    plt.plot(e,Validation_Loss.values(),label='Validation_Loss')
    plt.xlabel('Number Of Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Vs Number of epochs')
    plt.legend()
    plt.show()
    plt.subplot(1, 2, 2)
    plt.plot(e,Training_Accuracy.values(),label='Training_Accuracy')
    plt.plot(e,Validation_Accuracy.values(),label='Validation_Accuracy')
    plt.xlabel('Number Of Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Vs Number of epochs')
    plt.legend()
    plt.show()

def create_HL_list(num_layers,hidden_size):
    """
    Creates a list of number of neurons per layer

    Parameters:
    ----------
        num_layers(int) : Number of hidden layers in the neural network
        hidden_size(int) : Number of neurons in per hidden layer
     Returns:
    -------
        Num_Neurons_per_Layer(List) : List containing Number of neurons in all hidden layers
    """
    Num_Neurons_per_Layer = []
    for i in range(num_layers):
        Num_Neurons_per_Layer.append(hidden_size)
    return Num_Neurons_per_Layer

"""#**Weight Initialization**"""

def Xavier_weight_init(F_in,F_out):
    """
    Performs Xavier Weight initialization

    Parameters:
    ----------
        F_in(int) : “fan in,” or the number of inputs to the layer
        F_out(int) : “fan out,” or number of outputs from the layer
     Returns:
    -------
        W(numpy.ndarray) : Weight matrix
    """
    limit = np.sqrt(2 / float(F_in + F_out))
    W = np.random.normal(0.0, limit, size=(F_in, F_out))
    return W

def initialize_Weights_biases(Num_hidden_layers,Num_Neurons_per_Layer,class_size,weight_init,in_shape,grad_wandb=False):
    """
    Performs Weight initialization

    Parameters:
    ----------
        Num_hidden_layers(int) : Number of hidden layers in the neural network
        Num_Neurons_per_Layer(List) : List containing Number of neurons in all hidden layers
        weight_init(str): Type of weight initialization. random or Xavier
        in_shape(int): Shape of the imput vector
        grad_wandb(bool): To initialize gradients to zero

     Returns:
    -------
        W_dict(dict) : contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
    """
    W_dict = {}
    B_dict = {}
    in_shape = in_shape
    if Num_hidden_layers != len(Num_Neurons_per_Layer):
        raise Exception("Length of the list 'Num_Neurons_per_Layer' should be equal to Number of hidden layers")

    if grad_wandb:
        for i,j in zip(range(1,Num_hidden_layers+1), Num_Neurons_per_Layer):
          W_dict[f'W_layer_{i}'] = np.zeros((j, in_shape))
          in_shape = j
          B_dict[f'B_layer_{i}'] = np.zeros((j, 1))
        W_dict[f'W_layer_{i+1}'] = np.zeros((class_size, in_shape))
        B_dict[f'B_layer_{i+1}'] = np.zeros((class_size, 1))
        return W_dict, B_dict

    else:
        if weight_init == "Xavier":
            for i,j in zip(range(1,Num_hidden_layers+1), Num_Neurons_per_Layer):
                W_dict[f'W_layer_{i}'] = Xavier_weight_init(j,in_shape)
                in_shape = j
                B_dict[f'B_layer_{i}'] = Xavier_weight_init(j,1)
            W_dict[f'W_layer_{i+1}'] = Xavier_weight_init(class_size,in_shape)
            B_dict[f'B_layer_{i+1}'] = Xavier_weight_init(class_size,1)
            return W_dict, B_dict


        elif weight_init == "random":
            for i,j in zip(range(1,Num_hidden_layers+1), Num_Neurons_per_Layer):
                W_dict[f'W_layer_{i}'] = np.random.randn(j, in_shape)*0.01
                in_shape = j
                B_dict[f'B_layer_{i}'] = np.random.randn(j, 1)*0.01
            W_dict[f'W_layer_{i+1}'] = np.random.randn(class_size, in_shape)*0.01
            B_dict[f'B_layer_{i+1}'] = np.random.randn(class_size, 1)*0.01
            return W_dict, B_dict
          # for i,j in zip(range(1,Num_hidden_layers+1), Num_Neurons_per_Layer):
          #   W_dict[f'W_layer_{i}'] = np.random.rand(j, in_shape) - 0.5
          #   in_shape = j
          #   B_dict[f'B_layer_{i}'] = np.random.rand(j, 1) - 0.5
          # W_dict[f'W_layer_{i+1}'] = np.random.rand(class_size, in_shape) - 0.5
          # B_dict[f'B_layer_{i+1}'] = np.random.rand(class_size, 1) - 0.5
          # return W_dict, B_dict

# def L2_Loss(W_dict, weight_decay):
#     if weight_decay == 0:
#         return 0.
#     else:
#         cost = 0.
#         for i in W_dict.keys():
#             cost += float(np.sum(np.square(W_dict[i])))
        # return 0.5*cost*weight_decay

"""#**Weight update functions**"""

def sort_keys(input_dict):
    """
    Sorts keys in weight dictionaries after backward propogation.

    Parameters:
    ----------
        input_dict(dict) : Weight dictionary whose keys are to be sorted

     Returns:
    -------
        d1(dict) : Sorted Weight dictionary
    """
    l = sorted(input_dict.keys())
    s = list()
    for ele in l:
        s.append(input_dict[ele])
    d1 = dict(zip(l,s))
    return d1


def pre_activation_func(Weights, Biases,Input):
    """
    Performs pre-activation i.e., Wx + b.

    Parameters:
    ----------
        Weights(numpy.ndarray) : Weights
        Input(numpy.ndarray): Inputs (or activations of previous layers)
        Biases(numpy.ndarray) : Biases

     Returns:
    -------
        a_i(numpy.ndarray) : pre-activation
    """
    a_i = np.matmul(Weights, Input) + Biases
    return a_i

"""#**Activation functions**"""

def sigmoid(x):
    """
    Performs sigmoid activation i.e., 1.0/(1.0 + exp(-x)).

    Parameters:
    ----------
        x(numpy.ndarray) : input or Pre-activation
     Returns:
    -------
        (numpy.ndarray) : activation
    """
    return 1.0/(1.0 + np.exp(-x))
def ReLu(x):
    """
    Performs ReLu activation i.e., max(0,x)

    Parameters:
    ----------
        x(numpy.ndarray) : input or Pre-activation
     Returns:
    -------
        (numpy.ndarray) : activation
    """
    return max(0,x)

def activation_func(pre_activation,activation,derivative=False):
    """
    Performs different activations i.e., Relu, tanh, sigmoid.

    Parameters:
    ----------
        pre_activation(numpy.ndarray) : input or Pre-activation
        derivative(bool): True when derivative of activation is needed.

     Returns:
    -------
        h_i(numpy.ndarray) : when derivative = False, activation
        (numpy.ndarray): when derivative = True, gradient of activation
    """
    if activation == 'identity':
        h_i = pre_activation
        if derivative:
            return np.ones((h_i.shape[0],1))
        else:
            return h_i
    elif activation == 'sigmoid':
        sigmoid_v = np.vectorize(sigmoid)
        h_i = sigmoid_v(pre_activation)
        if derivative:
            return h_i*(np.ones((pre_activation.shape[0],1))-h_i)
        else:
            return h_i
    elif activation == 'tanh':
        h_i = np.tanh(pre_activation)
        if derivative:
            return 1.0 - (h_i)**2
        else:
            return h_i
    elif activation == 'ReLu':
        h_i = pre_activation * (pre_activation > 0)

        if derivative:
            return 1.0 * (pre_activation>0)
        else:
            return h_i

def output_func(A_final):
    e_x = np.exp(A_final - np.max(A_final))
    return e_x / e_x.sum()


"""#**Loss functions**"""

def loss_func_CEL(actual_value,prediction):
    """
    Calculates cross entropy loss i.e., -sum(y*log(y_hat)).

    Parameters:
    ----------
        actual_value(numpy.ndarray) : Output label as 1-hot vector
        prediction(numpy.ndarray): model prediction

     Returns:
    -------
        (float) : cross entropy loss
    """
    CrossEntropyLoss = -np.dot(actual_value.T,np.log(prediction))
    return float(CrossEntropyLoss)

def L2_Loss(W_dict, weight_decay):
    """
    Calculates L2 Regularization loss i.e., (lambda/2)*(|W|^2).

    Parameters:
    ----------
        W_dict(dict) : contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        weight_decay(float): lambda value

     Returns:
    -------
        (float) : L2 Regularization loss
    """
    if weight_decay == 0.0:
        return 0.
    else:
        cost = 0.
        for i in W_dict.keys():
            cost += float(np.sum(np.square(W_dict[i])))
        return 0.5*cost*weight_decay

def loss_func(actual_value,prediction,loss_type):
    """
    Calculates loss i.e., (lambda/2)*(|W|^2).

    Parameters:
    ----------
        actual_value(numpy.ndarray) : Output label as 1-hot vector
        prediction(numpy.ndarray): model prediction
        loss_type(str): 'mean_squared_error' or 'cross_entropy'

     Returns:
    -------
        (float) : Loss
    """
    if loss_type == 'mean_squared_error':
        a = actual_value - prediction
        SE = np.sum(np.square(a))
        return 0.5*float(SE)
    elif loss_type == 'cross_entropy':
        CrossEntropyLoss = -np.dot(actual_value.T,np.log(prediction))
        return float(CrossEntropyLoss)


def unisonShuffleDataset(a, b):
    """
    Helper function to shuffle data

    Parameters:
    ----------
        a(numpy.ndarray) : input
        b(numpy.ndarray): output

     Returns:
    -------
        a(numpy.ndarray) : shuffled input
        b(numpy.ndarray): shufled output
    """

    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

