from Optimizer import *
from Utils import *



class FFNN():
  """
    Creates Feed Forward Neural Network model

    Methods:
    --------
        __init__: create layers and initialize weight
        forward_pass: forward pass of the network
        backward_prop: backward pass of the network
        evaulate: created to run a forward pass on validation and test data and return metrics
        fit: runs a training loop


  """

  def __init__(self,dataset,weight_init,num_layers,hidden_size,class_size,activation,loss,optimizer,weight_decay,epochs,eta,momentum,batch_size,eps,beta,beta1,beta2):
    """

    Parameters:
    ----------
        dataset (str): Dataset name. fashion_mnist or mnist.
        weight_init(str): Type of weight initialization. random or Xavier
        num_layers(int) : Number of hidden layers in the neural network
        hidden_size(int) : Number of neurons in per hidden layer
        class_size(int): 10 for mnist and fashion_mnist datasets
        activation(str): Type of activation function
        loss(str): Type of activation function
        optimizer(str): Name of optimization algorithm
        weight_decay(float): parameter in L2 regularization
        epochs(int): Number os epochs
        eta(float): Learning rate
        momentum(float): Momentum used by momentum and nag optimizers.
        batch_size(int): batch size
        eps(float): Epsilon used by optimizers.
        beta(float): Beta used by rmsprop optimizer
        beta1(float): Beta1 used by adam and nadam optimizers.
        beta2(float): Beta2 used by adam and nadam optimizers.
  """
    self.dataset = dataset
    self.weight_init = weight_init
    self.activation = activation
    self.loss = loss
    self.num_layers = num_layers
    self.class_size = class_size
    self.hidden_size = hidden_size
    self.optimizer = optimizer
    self.Num_Neurons_per_Layer = create_HL_list(num_layers,hidden_size)
    self.weight_decay = weight_decay
    self.epochs = epochs
    self.eta = eta
    self.momentum = momentum
    self.batch_size = batch_size
    self.eps = eps
    self.beta = beta
    self.beta1 = beta1
    self.beta2 = beta2

  def forward_pass(self,input_vec, W_dict, B_dict):
    """
    Performs Forward propogation in the neural network, i.e., calculates output

    Parameters:
    ----------
        input_vec(numpy.ndarray): flattened input vector(1 sample of feature)
        W_dict(dict) : contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays


     Returns:
    -------
        Pre_act_dict(dict) : contains pre-activations whose keys are layer names and values are numpy.ndarrays of pre-activations
        act_dict(dict) : contains activations whose keys are layer names and values are numpy.ndarrays of activations
        Y_hat(numpy.ndarray): output from forward pass
    """
    Pre_act_dict = {}
    act_dict = {}
    act_dict[f'H_layer_{0}'] = input_vec
    for i in range(1,self.num_layers+1):
      Pre_act_dict[f'A_layer_{i}'] = pre_activation_func(W_dict[f'W_layer_{i}'],B_dict[f'B_layer_{i}'],act_dict[f'H_layer_{i-1}'])
      act_dict[f'H_layer_{i}'] = activation_func(Pre_act_dict[f'A_layer_{i}'],self.activation)
    Pre_act_dict[f'A_layer_{i+1}'] = pre_activation_func(W_dict[f'W_layer_{i+1}'],B_dict[f'B_layer_{i+1}'],act_dict[f'H_layer_{i}'])
    Y_hat = output_func(Pre_act_dict[f'A_layer_{i+1}'])
    return Pre_act_dict, act_dict, Y_hat


  def backward_prop(self, Pre_act_dict, W_dict, B_dict, act_dict,Y_hat,Y_ohv):
    """
    Performs Backward propogation in the neural network, i.e., calculates gradients of loss function w.r.t to weights and biases

    Parameters:
    ----------
        Pre_act_dict(dict) : contains pre-activations whose keys are layer names and values are numpy.ndarrays of pre-activations
        W_dict(dict) : contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
        act_dict(dict) : contains activations whose keys are layer names and values are numpy.ndarrays of activations
        Y_hat(numpy.ndarray): output from forward pass
        Y_ohv(numpy.ndarray): actual label in 1-hot vector form

     Returns:
    -------
        grad_W_dict(dict) : contains gradients of weights whose keys are layer names and values are corresponding gradients of weights in the form of numpy.ndarrays
        grad_B_dict(dict): contains gradients of biases whose keys are layer names and values are corresponding gradients of biases in the form of numpy.ndarrays
    """
    grad_W_dict = {}
    grad_B_dict = {}
    grad_act_dict = {}
    grad_pre_act_dict = {}

    a = (Y_ohv-Y_hat)
    if self.loss == 'cross_entropy':
      grad_pre_act_dict[f'A_layer_{self.num_layers+1}'] = -a
    elif self.loss == 'mean_squared_error':
      grad_pre_act_dict[f'A_layer_{self.num_layers+1}'] = float(np.dot(a.T,a))*Y_hat

    # if self.loss == 'cross_entropy':
    #   grad_pre_act_dict[f'A_layer_{self.num_layers+1}'] = -(Y_ohv-Y_hat)
    # elif self.loss == 'mean_squared_error':
    #   grad_pre_act_dict[f'A_layer_{self.num_layers+1}'] = (Y_hat - Y_ohv) * (np.ones((10,1))-Y_hat) * Y_hat

    # grad_pre_act_dict[f'A_layer_{self.num_layers+1}'] = -(Y_ohv-Y_hat)
    for i in range(self.num_layers+1,0,-1):
      grad_W_dict[f'W_layer_{i}'] = np.dot(grad_pre_act_dict[f'A_layer_{i}'],act_dict[f'H_layer_{i-1}'].T)
      grad_B_dict[f'B_layer_{i}'] = grad_pre_act_dict[f'A_layer_{i}']
      if i == 1:
        break
      grad_act_dict[f'H_layer_{i-1}'] = np.dot(np.transpose(W_dict[f'W_layer_{i}']),grad_pre_act_dict[f'A_layer_{i}'])
      grad_pre_act_dict[f'A_layer_{i-1}'] = grad_act_dict[f'H_layer_{i-1}'] * activation_func(Pre_act_dict[f'A_layer_{i-1}'],self.activation,derivative=True)

    grad_W_dict = sort_keys(grad_W_dict)
    grad_B_dict = sort_keys(grad_B_dict)

    return grad_W_dict, grad_B_dict

  def evaluate(self,X_val,Y_val,W_dict,B_dict,weight_decay,confusion=False):
    """
    Performs testing on validation and test data

    Parameters:
    ----------
        x_val(numpy.ndarray) : Flattened and Normalized Validation features
        y_ohv_val(numpy.ndarray): Validation Labels in 1-hot vector form
        W_dict(dict) : contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
        weight_decay(float): parameter in L2 regularization
        confusion(bool): To create confusion matrix when test data is passed

     Returns:
    -------
        validation_loss(float)
        validation_accuracy(float)
    """
    loss = 0.
    correct = 0
    validation_loss = 0.
    validation_accuracy = 0.
    Y_hat_list = []
    Y_val_list = []
    # print('eval:',B_dict['B_layer_1'][0])
    for x,y_val in zip(X_val,Y_val):
      _, _,Y_hat = self.forward_pass(x, W_dict, B_dict)
      loss += (loss_func(y_val,Y_hat,self.loss) + L2_Loss(W_dict,weight_decay))
      if np.argmax(y_val) == np.argmax(Y_hat):
        correct += 1
      Y_hat_list.append(np.argmax(y_val))
      Y_val_list.append(np.argmax(Y_hat))
    if confusion:
      create_confusion_matrix(np.array(Y_val_list), np.array(Y_hat_list))
    validation_loss = loss/X_val.shape[0]
    validation_accuracy = correct/X_val.shape[0]
    return validation_loss, validation_accuracy

  def fit(self,x_val,y_ohv_val,x_train,y_ohv_train,X_test,Y_ohv_test):
    """

    Runs a training loop

    Parameters:
    ----------
        x_val(numpy.ndarray) : Flattened and Normalized Validation features
        y_ohv_val(numpy.ndarray): Validation Labels in 1-hot vector form
        x_train(numpy.ndarray) : Flattened and Normalized Train features
        y_ohv_train(numpy.ndarray) : Train Labels in 1-hot vector form
        X_test(numpy.ndarray) : Flattened and Normalized Test features
        Y_ohv_test(numpy.ndarray) : Test Labels n 1-hot vector form

    Returns:
    -------
        None

    """

    W_dict,B_dict = initialize_Weights_biases(self.num_layers,self.Num_Neurons_per_Layer,self.class_size,self.weight_init,in_shape = x_train[0].shape[0],grad_wandb=False)
    v_w, v_b = initialize_Weights_biases(self.num_layers,self.Num_Neurons_per_Layer,self.class_size,self.weight_init,in_shape = x_train[0].shape[0],grad_wandb=True)
    m_w, m_b = initialize_Weights_biases(self.num_layers,self.Num_Neurons_per_Layer,self.class_size,self.weight_init,in_shape = x_train[0].shape[0],grad_wandb=True)
    validation_loss = 0.
    validation_accuracy = 0.
    training_loss = 0.
    training_accuracy = 0.
    Training_Loss = {}
    Training_Accuracy = {}
    Validation_Loss = {}
    Validation_Accuracy = {}
    update = 0
    print(B_dict['B_layer_1'].shape)
    
    """
    Adaptive early stoppage:
    -------------------------
    I am using this strategy as a technique to stop the sweep early if it seems like
    the model is not going to give a good accuracy or cannot make potentaial improvement. 
    I am using this strategy in order to not waste the compute and save time.

    Strategy:
    ---------
    1. if the validation loss dosen't reduce for a long time then it should stop running.
    2. if the sweep runs for 10 eopcs, after 5 epocs, the logic will check if the validation accuracy < 50%
    if it is less then the training is stopped.
    similarlly, if the sweep runs for 20/30 eopcs, after 10 epocs, the logic will check if the validation accuracy < 50%
    if it is less then the training is stopped.
    similarlly, if the sweep runs for more then 30 eopcs, after 25 epocs, the logic will check if the validation accuracy < 50%
    if it is less then the training is stopped.

    """
    prev_val_loss = float('inf')
    no_improvement_epochs = 0
    if self.epochs <= 10:
        check_epoch = 5
    elif self.epochs <= 30:
        check_epoch = 10
    else:
        check_epoch = 25

    for t in range(1,self.epochs+1):
      print(f"EPOCH: {t}")
      # print('before',B_dict['B_layer_1'].shape)
      dw, db = initialize_Weights_biases(self.num_layers,self.Num_Neurons_per_Layer,self.class_size,self.weight_init,in_shape = x_train[0].shape[0],grad_wandb=True)
      # print('before derivative',db['B_layer_1'].shape)
      loss_per_epoch = 0.
      Num_points_seen = 0
      correct = 0
      training_loss = 0.
      X,Y = unisonShuffleDataset(x_train,y_ohv_train)
      for x,y_ohv in zip(X,Y):
        temp_W_dict = {}
        temp_B_dict = {}
        Pre_act_dict, act_dict,Y_hat = self.forward_pass(x, W_dict, B_dict)
        temp_W_dict, temp_B_dict = self.backward_prop(Pre_act_dict, W_dict, B_dict, act_dict,Y_hat,y_ohv)
        Num_points_seen += 1
        dw, db = update_grad(dw,temp_W_dict,db,temp_B_dict)
        if Num_points_seen % self.batch_size == 0:
          update += 1
          if self.optimizer == "sgd":
            W_dict, B_dict = Optimizer_step(self.optimizer,update, W_dict, B_dict, dw, db, self.eta, self.batch_size,self.weight_decay)
          elif self.optimizer == "momentum":
            v_w, v_b, W_dict, B_dict = Optimizer_step(self.optimizer,update, W_dict, B_dict, dw, db, self.eta, self.batch_size, self.weight_decay, self.momentum, self.beta, self.beta1, self.beta2,self.eps, v_w, v_b)
          elif self.optimizer == 'nag':
            v_w, v_b, W_dict, B_dict = Optimizer_step(self.optimizer,update, W_dict, B_dict, dw, db, self.eta, self.batch_size, self.weight_decay, self.momentum, self.beta,self.beta1, self.beta2,self.eps, v_w, v_b)
          elif self.optimizer == 'rmsprop':
            v_w, v_b, W_dict, B_dict = Optimizer_step(self.optimizer,update, W_dict, B_dict, dw, db, self.eta, self.batch_size, self.weight_decay, self.momentum, self.beta, self.beta1, self.beta2,self.eps, v_w, v_b)
          elif self.optimizer == 'adam':
            m_w, m_b, v_w, v_b, W_dict, B_dict = Optimizer_step(self.optimizer,update, W_dict, B_dict, dw, db, self.eta, self.batch_size, self.weight_decay, self.momentum, self.beta, self.beta1, self.beta2,self.eps, v_w, v_b, m_w, m_b)
          elif self.optimizer == 'nadam':
            m_w, m_b, v_w, v_b, W_dict, B_dict = Optimizer_step(self.optimizer,update, W_dict, B_dict, dw, db, self.eta, self.batch_size, self.weight_decay, self.momentum, self.beta, self.beta1, self.beta2,self.eps, v_w, v_b, m_w, m_b)
          dw, db = initialize_Weights_biases(self.num_layers,self.Num_Neurons_per_Layer,self.class_size,self.weight_init,in_shape = x_train[0].shape[0],grad_wandb=True)
        loss_per_epoch += (loss_func(y_ohv,Y_hat,self.loss) + L2_Loss(W_dict,self.weight_decay))
        if np.argmax(y_ohv) == np.argmax(Y_hat):
            correct += 1
      training_loss  = loss_per_epoch/X.shape[0]
      training_accuracy = correct/X.shape[0]
      # print('after',B_dict['B_layer_1'][0])
      # print('Num_updates:',update)
      validation_loss, validation_accuracy = self.evaluate(x_val,y_ohv_val,W_dict,B_dict,self.weight_decay)
      Training_Loss[f'Epoch_{t}'] = training_loss
      Training_Accuracy[f'Epoch_{t}'] = training_accuracy
      Validation_Loss[f'Epoch_{t}'] = validation_loss
      Validation_Accuracy[f'Epoch_{t}'] = validation_accuracy
      wandb.log({"validation_accuracy" : validation_accuracy, "validation_loss" : validation_loss, "training_accuracy" : training_accuracy, "training_loss" : training_loss, "epochs": t})
      print(f"TRAINING LOSS :{Training_Loss[f'Epoch_{t}']}")
      print(f"TRAINING ACCURACY :{Training_Accuracy[f'Epoch_{t}']}")
      print(f"VALIDATION LOSS :{Validation_Loss[f'Epoch_{t}']}")
      print(f"VALIDATION ACCURACY :{Validation_Accuracy[f'Epoch_{t}']}")

      """ Early Stoppage """
      if t >= check_epoch:
          if validation_accuracy < 0.5:
              print(f"I do not see signs of further improvement. Therefore, Ending training..\nStopping early at epoch {t}: Accuracy too low ({validation_accuracy:.2f})")
              break

          if validation_loss >= prev_val_loss - 0.001:
              no_improvement_epochs += 1
          else:
              no_improvement_epochs = 0
          prev_val_loss = validation_loss

          if no_improvement_epochs >= 3:
              print(f"I do not see signs of further improvement. Therefore, Ending training. \nStopping early at epoch {t}: Validation loss not improving for 3 consecutive epochs.")
              break
            
    print("\n Testing on unseen data")
    Test_loss, Test_Accuracy = self.evaluate(X_test,Y_ohv_test,W_dict,B_dict,self.weight_decay,confusion=True)
    print(f"TEST LOSS :{Test_loss}")
    print(f"TEST ACCURACY :{Test_Accuracy}")
    # if plot:
    #   plot_metrics(self.epochs, Training_Loss,Training_Accuracy,Validation_Loss,Validation_Accuracy)
    MyDicts = [W_dict, B_dict]
    pkl.dump(MyDicts, open("model_final.p", "wb" ))

    return validation_accuracy
