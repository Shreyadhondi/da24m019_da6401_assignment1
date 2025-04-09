import numpy as np

def update_grad(dw,grad_W_dict,db,grad_B_dict):
  """
    Accumulates gradients till 1 batch is complete.

    Parameters:
    ----------

        grad_W_dict(dict) : contains gradients of weights after training on one point. whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        grad_B_dict(dict): contains gradients of biases after training on one point. contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
        dw(dict) : contains gradients of weights whose keys are layer names and values are corresponding gradients of weights in the form of numpy.ndarrays
        db(dict): contains gradients of biases whose keys are layer names and values are corresponding gradients of biases in the form of numpy.ndarrays

     Returns:
    -------
        dw(dict) : After Update...contains gradients of weights whose keys are layer names and values are corresponding gradients of weights in the form of numpy.ndarrays
        db(dict): After Update...contains gradients of biases whose keys are layer names and values are corresponding gradients of biases in the form of numpy.ndarrays
  """

  for i,j in zip(dw.keys(),db.keys()):
    dw[i] = dw[i] + grad_W_dict[i]
    db[j] = db[j] + grad_B_dict[j]
  return dw,db

def SGD_step(W_dict,B_dict,dw,db,eta,batch_size, weight_decay):
  """
    Performs Optimizer step i.e., W = W - (eta*dw) - (eta*w*weight_decay) for SGD
    Parameters:
    ----------

        W_dict(dict) : contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
        dw(dict) : contains gradients of weights whose keys are layer names and values are corresponding gradients of weights in the form of numpy.ndarrays
        db(dict): contains gradients of biases whose keys are layer names and values are corresponding gradients of biases in the form of numpy.ndarrays
        eta(float): Learning rate
        batch_size(int): batch size
        weight_decay(float): parameter in L2 regularization
     Returns:
    -------
        W_dict(dict) : after update...contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): after update...contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
  """
  for i,j in zip(W_dict.keys(),B_dict.keys()):
    W_dict[i] = W_dict[i] - (eta*(dw[i]/batch_size)) - (eta*weight_decay*W_dict[i])
    B_dict[j] = B_dict[j] - (eta*(db[j]/batch_size))
  return W_dict,B_dict


def momentum_velocity_update(v_w,v_b,momentum,dw,db,eta,batch_size):
  """
    Performs Velocity update i.e., v_w =  momentum*v_w + (eta*dw) for momentum based SGD
    Parameters:
    ----------
        v_w(dict): contains velocity weights whose keys are layer names and values are corresponding  velocity weights in the form of numpy.ndarrays
        v_b(dict): contains velocity biases whose keys are layer names and values are corresponding  velocity biases in the form of numpy.ndarrays
        momentum(float) : Momentum used by momentum and nag optimizers.
        dw(dict) : contains gradients of weights whose keys are layer names and values are corresponding gradients of weights in the form of numpy.ndarrays
        db(dict): contains gradients of biases whose keys are layer names and values are corresponding gradients of biases in the form of numpy.ndarrays
        eta(float): Learning rate
        batch_size(int): batch size
        weight_decay(float): parameter in L2 regularization
     Returns:
    -------
        v_w(dict): after update...contains velocity weights whose keys are layer names and values are corresponding  velocity weights in the form of numpy.ndarrays
        v_b(dict): after update...contains velocity biases whose keys are layer names and values are corresponding  velocity biases in the form of numpy.ndarrays
  """
  for i,j in zip(dw.keys(),db.keys()):
    v_w[i] = momentum*v_w[i] + eta*(dw[i]/batch_size)
    v_b[j] = momentum*v_b[j] + eta*(db[j]/batch_size)
  return v_w,v_b

def momentum_step(W_dict,B_dict,v_w,v_b,eta,batch_size, weight_decay):
  """
    Performs optimizer step in momentum i.e., w =  w - v_w - (eta*w*weight_decay)
    Parameters:
    ----------
        v_w(dict): contains velocity weights whose keys are layer names and values are corresponding  velocity weights in the form of numpy.ndarrays
        v_b(dict): contains velocity biases whose keys are layer names and values are corresponding  velocity biases in the form of numpy.ndarrays
        W_dict(dict) : contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
        eta(float): Learning rate
        batch_size(int): batch size
        weight_decay(float): parameter in L2 regularization
     Returns:
    -------
        W_dict(dict) : after update...contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): after update...contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
  """
  for i,j in zip(W_dict.keys(),B_dict.keys()):
    W_dict[i] = W_dict[i] - v_w[i] - (eta*weight_decay*W_dict[i])
    B_dict[j] = B_dict[j] - v_b[j]
  return W_dict,B_dict



def NAG_step(W_dict,B_dict,momentum,v_w,v_b,dw,db,eta,batch_size,weight_decay):
  """
    Performs optimizer step in nag i.e., w =  w - eta*(momentum*v_w + dw) - (eta*w*weight_decay)
    Parameters:
    ----------
        v_w(dict): contains velocity weights whose keys are layer names and values are corresponding  velocity weights in the form of numpy.ndarrays
        v_b(dict): contains velocity biases whose keys are layer names and values are corresponding  velocity biases in the form of numpy.ndarrays
        W_dict(dict) : contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
        momentum(float) : Momentum used by momentum and nag optimizers.
        dw(dict) : contains gradients of weights whose keys are layer names and values are corresponding gradients of weights in the form of numpy.ndarrays
        db(dict): contains gradients of biases whose keys are layer names and values are corresponding gradients of biases in the form of numpy.ndarrays
        eta(float): Learning rate
        batch_size(int): batch size
        weight_decay(float): parameter in L2 regularization
     Returns:
    -------
        W_dict(dict) : after update...contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): after update...contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
  """
  for i,j in zip(W_dict.keys(),B_dict.keys()):
    W_dict[i] = W_dict[i] - eta*(momentum*v_w[i] + (dw[i]/batch_size)) - (eta*weight_decay*W_dict[i])
    B_dict[j] = B_dict[j] - eta*(momentum*v_b[j] + (db[j]/batch_size))
  return W_dict,B_dict

def rmsprop_update_grad(v_w,v_b,beta,dw,db,batch_size):
  """
    Performs rmsprop gradient update i.e., v_w = (beta*v_w) + (1-beta)*dw^2

    Parameters:
    ----------

        v_w(dict): contains velocity weights whose keys are layer names and values are corresponding  velocity weights in the form of numpy.ndarrays
        v_b(dict): contains velocity biases whose keys are layer names and values are corresponding  velocity biases in the form of numpy.ndarrays
        dw(dict) : contains gradients of weights whose keys are layer names and values are corresponding gradients of weights in the form of numpy.ndarrays
        db(dict): contains gradients of biases whose keys are layer names and values are corresponding gradients of biases in the form of numpy.ndarrays
        batch_size(int): batch size
        beta(float): Beta used by rmsprop optimizer

     Returns:
    -------
        v_w(dict): after update...contains velocity weights whose keys are layer names and values are corresponding  velocity weights in the form of numpy.ndarrays
        v_b(dict): after update...contains velocity biases whose keys are layer names and values are corresponding  velocity biases in the form of numpy.ndarrays
  """
  for i,j in zip(dw.keys(),db.keys()):
    v_w[i] = beta*v_w[i] + (1-beta)*((dw[i]/batch_size)**2)
    v_b[j] = beta*v_b[j] + (1-beta)*((db[j]/batch_size)**2)
  return v_w,v_b

def rmsprop_step(W_dict,B_dict,v_w,v_b,eps,dw,db,eta,batch_size,weight_decay):
  """
    Performs rmsprop step i.e., w = w - ((eta* dw)/sqrt(v_w+eps)) - (eta*w*weight_decay)

    Parameters:
    ----------
        W_dict(dict) : contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
        v_w(dict): contains velocity weights whose keys are layer names and values are corresponding  velocity weights in the form of numpy.ndarrays
        v_b(dict): contains velocity biases whose keys are layer names and values are corresponding  velocity biases in the form of numpy.ndarrays
        eps(float): Epsilon used by optimizers.
        dw(dict) : contains gradients of weights whose keys are layer names and values are corresponding gradients of weights in the form of numpy.ndarrays
        db(dict): contains gradients of biases whose keys are layer names and values are corresponding gradients of biases in the form of numpy.ndarrays
        eta(float): Learning rate
        batch_size(int): batch size
        weight_decay(float): parameter in L2 regularization

    Returns:
    -------
        W_dict(dict) : after update...contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): after update...contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
  """
  for i,j in zip(W_dict.keys(),B_dict.keys()):
    W_dict[i] = W_dict[i] - ((eta / np.sqrt(v_w[i] + eps))*(dw[i]/batch_size)) - (eta*weight_decay*W_dict[i])
    B_dict[j] = B_dict[j] - ((eta / np.sqrt(v_b[j] + eps))*(db[j]/batch_size))
  return W_dict,B_dict

def adam_update_momentum(m_w,m_b,beta1,dw,db,batch_size):
  """
    Performs adam momentum update i.e.,  m_w =  beta1*m_w + (1-beta1)*dw
    Parameters:
    ----------
        m_w(dict): contains adam velocity weights whose keys are layer names and values are corresponding  velocity weights in the form of numpy.ndarrays
        m_b(dict): contains adam velocity biases whose keys are layer names and values are corresponding  velocity biases in the form of numpy.ndarrays
        eps(float): Epsilon used by optimizers.
        dw(dict) : contains gradients of weights whose keys are layer names and values are corresponding gradients of weights in the form of numpy.ndarrays
        db(dict): contains gradients of biases whose keys are layer names and values are corresponding gradients of biases in the form of numpy.ndarrays
        beta1(float): Beta1 used by adam and nadam optimizers.
        batch_size(int): batch size


    Returns:
    -------
        m_w(dict): after update...contains adam velocity weights whose keys are layer names and values are corresponding  velocity weights in the form of numpy.ndarrays
        m_b(dict): after update...contains adam velocity biases whose keys are layer names and values are corresponding  velocity biases in the form of numpy.ndarrays
  """
  for i,j in zip(dw.keys(),db.keys()):
    m_w[i] = beta1*m_w[i] + (1-beta1)*(dw[i]/batch_size)
    m_b[j] = beta1*m_b[j] + (1-beta1)*(db[j]/batch_size)
  return m_w,m_b

def adam_bias_correction(w,b,t,beta):
  """
    Performs adam bias correction i.e.,  m_w =  m_w/(1-beta1**t)
    Parameters:
    ----------
        w(dict): contains weights whose keys are layer names and values are corresponding  weights in the form of numpy.ndarrays
        b(dict): contains biases whose keys are layer names and values are corresponding  biases in the form of numpy.ndarrays
        t(float): update
        beta(float): parameter

    Returns:
    -------
        w_hat(dict): after bias correction...contains weights whose keys are layer names and values are corresponding  weights in the form of numpy.ndarrays
        b_hat(dict): after bias correction...contains biases whose keys are layer names and values are corresponding  biases in the form of numpy.ndarrays
  """
  w_hat = {}
  b_hat = {}
  for i,j in zip(w.keys(),b.keys()):
    w_hat[i] = w[i]/(1-beta**t)
    b_hat[j] = b[j]/(1-beta**t)
  return w_hat,b_hat



def adam_step(W_dict,B_dict,v_w_hat,v_b_hat,m_w_hat,m_b_hat,eps,eta,batch_size,weight_decay):
  """
    Performs adam step i.e.,  W = W - ((eta *m_w_hat)/ (np.sqrt(v_w_hat) + eps)) - (eta*weight_decay*W)
    Parameters:
    ----------
        W_dict(dict) : contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
        v_w_hat(dict):  after bias correction...contains rmsprop weight terms whose keys are layer names and values are corresponding  velocity weights in the form of numpy.ndarrays
        v_b_hat(dict):  after bias correction...contains rmsprop bias terms biases whose keys are layer names and values are corresponding  velocity biases in the form of numpy.ndarrays
        eps(float): Epsilon used by optimizers.
        m_w_hat(dict):  after bias correction...contains adam velocity weights whose keys are layer names and values are corresponding  velocity weights in the form of numpy.ndarrays
        m_b_hat(dict):  after bias correction...contains adam velocity biases whose keys are layer names and values are corresponding  velocity biases in the form of numpy.ndarrays
        eps(float): Epsilon used by optimizers.
        dw(dict) : contains gradients of weights whose keys are layer names and values are corresponding gradients of weights in the form of numpy.ndarrays
        db(dict): contains gradients of biases whose keys are layer names and values are corresponding gradients of biases in the form of numpy.ndarrays
        beta1(float): Beta1 used by adam and nadam optimizers.
        batch_size(int): batch size
        weight_decay(float): parameter in L2 regularization


    Returns:
    -------
        W_dict(dict) : after update...contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): after update...contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
  """
  for i,j in zip(W_dict.keys(),B_dict.keys()):
    W_dict[i] = W_dict[i] - ((eta *m_w_hat[i])/ (np.sqrt(v_w_hat[i]) + eps)) - (eta*weight_decay*W_dict[i])
  return W_dict,B_dict

def Nadam_step(t,W_dict,B_dict,dw,db,eta,batch_size,v_w_hat,v_b_hat,m_w_hat,m_b_hat,eps, beta1,weight_decay):
  """
    Performs Nadam step i.e.,  W - ((eta * ((beta1*m_w_hat) + (((1-beta1)*(dw)/(1-beta1**t))))/ (np.sqrt(v_w_hat) + eps)) - (eta*weight_decay*W)
    Parameters:
    ----------
        t(int): update
        W_dict(dict) : contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
        v_w_hat(dict):  after bias correction...contains rmsprop weight terms whose keys are layer names and values are corresponding  velocity weights in the form of numpy.ndarrays
        v_b_hat(dict):  after bias correction...contains rmsprop bias terms biases whose keys are layer names and values are corresponding  velocity biases in the form of numpy.ndarrays
        eps(float): Epsilon used by optimizers.
        m_w_hat(dict):  after bias correction...contains adam velocity weights whose keys are layer names and values are corresponding  velocity weights in the form of numpy.ndarrays
        m_b_hat(dict):  after bias correction...contains adam velocity biases whose keys are layer names and values are corresponding  velocity biases in the form of numpy.ndarrays
        eps(float): Epsilon used by optimizers.
        dw(dict) : contains gradients of weights whose keys are layer names and values are corresponding gradients of weights in the form of numpy.ndarrays
        db(dict): contains gradients of biases whose keys are layer names and values are corresponding gradients of biases in the form of numpy.ndarrays
        beta1(float): Beta1 used by adam and nadam optimizers.
        batch_size(int): batch size
        weight_decay(float): parameter in L2 regularization

    Returns:
    -------
        W_dict(dict) : after update...contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): after update...contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
  """
  for i,j in zip(W_dict.keys(),B_dict.keys()):
    W_dict[i] = W_dict[i] - ((eta * ((beta1*m_w_hat[i]) + (((1-beta1)*(dw[i]/batch_size))/(1-beta1**t))))/ (np.sqrt(v_w_hat[i]) + eps)) - (eta*weight_decay*W_dict[i])
    B_dict[j] = B_dict[j] - ((eta * ((beta1*m_b_hat[j]) + (((1-beta1)*(db[j]/batch_size))/(1-beta1**t))))/ (np.sqrt(v_b_hat[j]) + eps))
  return W_dict,B_dict
