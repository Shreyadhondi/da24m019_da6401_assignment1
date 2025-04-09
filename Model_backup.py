from Optimizer import *
from Utils import *
import numpy as np

class FFNN():
  def __init__(self,dataset="fashion_mnist",weight_init = 'random',num_layers=2,hidden_size=[32,32],class_size=10,activation='ReLu',loss='cross_entropy',optimizer='sgd',device="cpu"):
    self.dataset = dataset
    self.weight_init = weight_init
    self.activation = activation
    self.loss = loss
    self.num_layers = num_layers
    self.class_size = class_size
    self.hidden_size = hidden_size
    self.optimizer = optimizer
    self.device = device


  def forward_pass(self,input_vec, W_dict, B_dict):
    Pre_act_dict = {}
    act_dict = {}
    act_dict[f'H_layer_{0}'] = input_vec
    for i in range(1,self.num_layers+1):
      Pre_act_dict[f'A_layer_{i}'] = pre_activation_func(W_dict[f'W_layer_{i}'],B_dict[f'B_layer_{i}'],act_dict[f'H_layer_{i-1}'])
      act_dict[f'H_layer_{i}'] = activation_func(Pre_act_dict[f'A_layer_{i}'],self.activation)
    Pre_act_dict[f'A_layer_{i+1}'] = pre_activation_func(W_dict[f'W_layer_{i+1}'],B_dict[f'B_layer_{i+1}'],act_dict[f'H_layer_{i}'])
    Y_hat = output_func(Pre_act_dict[f'A_layer_{i+1}'],self.loss)
    return Pre_act_dict, act_dict, Y_hat


  def backward_prop(self, Pre_act_dict, W_dict, B_dict, act_dict,Y_hat,Y_ohv):
    grad_W_dict = {}
    grad_B_dict = {}
    grad_act_dict = {}
    grad_pre_act_dict = {}

    a = (Y_ohv-Y_hat)
    if self.loss == 'cross_entropy':
      grad_pre_act_dict[f'A_layer_{self.num_layers+1}'] = -a
    elif self.loss == 'mean_squared_error':
      grad_pre_act_dict[f'A_layer_{self.num_layers+1}'] = float(np.dot(a.T,a))*Y_hat
    if self.loss == 'cross_entropy':
      grad_pre_act_dict[f'A_layer_{self.num_layers+1}'] = -(Y_ohv-Y_hat)
    elif self.loss == 'mse':
      grad_pre_act_dict[f'A_layer_{self.num_layers+1}'] = (Y_hat - Y_ohv) * (np.ones((10,1))-Y_hat) * Y_hat

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

  def evaluate(self,X_val,Y_val,W_dict,B_dict,weight_decay):
    loss = 0.
    correct = 0
    validation_loss = 0.
    validation_accuracy = 0.
    # print('eval:',B_dict['B_layer_1'][0])
    for x,y_val in zip(X_val,Y_val):
      _, _,Y_hat = self.forward_pass(x, W_dict, B_dict)
      loss += (loss_func(y_val,Y_hat,self.loss) + L2_Loss(W_dict,weight_decay))
      if np.argmax(y_val) == np.argmax(Y_hat):
        correct += 1
    validation_loss = loss/X_val.shape[0]
    validation_accuracy = correct/X_val.shape[0]
    return validation_loss, validation_accuracy

  def fit(self,x_val,y_ohv_val,x_train,y_ohv_train,X_test,Y_ohv_test,weight_decay=0,epochs=1, eta=0.1,momentum = 0.9,batch_size=16, eps=1e-8,beta=0.5, beta1=0.5, beta2=0.5,plot=True):
    W_dict,B_dict = initialize_Weights_biases(self.num_layers,self.hidden_size,self.class_size,self.weight_init,in_shape = x_train[0].shape[0],grad_wandb=False)
    v_w, v_b = initialize_Weights_biases(self.num_layers,self.hidden_size,self.class_size,self.weight_init,in_shape = x_train[0].shape[0],grad_wandb=True)
    m_w, m_b = initialize_Weights_biases(self.num_layers,self.hidden_size,self.class_size,self.weight_init,in_shape = x_train[0].shape[0],grad_wandb=True)
    Training_Loss = {}
    Training_Accuracy = {}
    Validation_Loss = {}
    Validation_Accuracy = {}
    for t in range(1,epochs+1):
      print(f"EPOCH: {t}")
      print('before',B_dict['B_layer_1'][0])
      dw, db = initialize_Weights_biases(self.num_layers,self.hidden_size,self.class_size,self.weight_init,in_shape = x_train[0].shape[0],grad_wandb=True)
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
        if Num_points_seen % batch_size == 0:
          if self.optimizer == "sgd":
            W_dict, B_dict = step(self.optimizer,t, W_dict, B_dict, dw, db, eta, batch_size,weight_decay)
          elif self.optimizer == "momentum":
            v_w, v_b, W_dict, B_dict = step(self.optimizer,t, W_dict, B_dict, dw, db, eta, batch_size, weight_decay, momentum, beta, beta1, beta2,eps, v_w, v_b)
          elif self.optimizer == 'nag':
            v_w, v_b, W_dict, B_dict = step(self.optimizer,t, W_dict, B_dict, dw, db, eta, batch_size, weight_decay, momentum, beta, beta1, beta2,eps, v_w, v_b)
          elif self.optimizer == 'rmsprop':
            v_w, v_b, W_dict, B_dict = step(self.optimizer,t, W_dict, B_dict, dw, db, eta, batch_size, weight_decay, momentum, beta, beta1, beta2,eps, v_w, v_b)
          elif self.optimizer == 'adam':
            m_w, m_b, v_w, v_b, W_dict, B_dict = step(self.optimizer,t, W_dict, B_dict, dw, db, eta, batch_size, weight_decay, momentum, beta, beta1, beta2,eps, v_w, v_b, m_w, m_b)
          elif self.optimizer == 'nadam':
            m_w, m_b, v_w, v_b, W_dict, B_dict = step(self.optimizer,t, W_dict, B_dict, dw, db, eta, batch_size, weight_decay, momentum, beta, beta1, beta2,eps, v_w, v_b, m_w, m_b)
          dw, db = initialize_Weights_biases(self.num_layers,self.hidden_size,self.class_size,self.weight_init,in_shape = x_train[0].shape[0],grad_wandb=True)
        loss_per_epoch += (loss_func(y_ohv,Y_hat,self.loss) + L2_Loss(W_dict,weight_decay))
        if np.argmax(y_ohv) == np.argmax(Y_hat):
            correct += 1
      Training_Loss[f'Epoch_{t}'] = loss_per_epoch/X.shape[0]
      Training_Accuracy[f'Epoch_{t}'] = correct/X.shape[0]
      print('after',B_dict['B_layer_1'][0])
      validation_loss, validation_accuracy = self.evaluate(x_val,y_ohv_val,W_dict,B_dict,weight_decay)
      Validation_Loss[f'Epoch_{t}'] = validation_loss
      Validation_Accuracy[f'Epoch_{t}'] = validation_accuracy
      wandb.log({"validation_accuracy" : validation_accuracy, "validation_loss" : validation_loss, "training_accuracy" : Training_Accuracy[f'Epoch_{t}'], "training_loss" : Training_Loss[f'Epoch_{t}'], "epochs": t})
