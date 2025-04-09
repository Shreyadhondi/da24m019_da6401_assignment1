from Helper_Functions import *

def Optimizer_step(optimizer,t, W_dict, B_dict, dw, db, eta, batch_size, weight_decay = 0, momentum = 0.9, beta = 0.9, beta1 = 0.9, beta2 = 0.9,eps = 1e-8, v_w={}, v_b={},m_w={}, m_b={}):
    """
    Performs Optimizer step i.e., W = W - (eta*dw) depending on the algorithm that we are using.

    Parameters:
    ----------
        optimizer(str): Name of optimization algorithm
        t(int): update
        W_dict(dict) : contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
        dw(dict) : contains gradients of weights whose keys are layer names and values are corresponding gradients of weights in the form of numpy.ndarrays
        db(dict): contains gradients of biases whose keys are layer names and values are corresponding gradients of biases in the form of numpy.ndarrays
        eta(float): Learning rate
        batch_size(int): batch size
        weight_decay(float): parameter in L2 regularization
        momentum(float): Momentum used by momentum and nag optimizers.
        beta(float): Beta used by rmsprop optimizer
        beta1(float): Beta1 used by adam and nadam optimizers.
        beta2(float): Beta2 used by adam and nadam optimizers.
        eps(float): Epsilon used by optimizers.
        v_w(dict): contains velocity weights whose keys are layer names and values are corresponding  velocity weights in the form of numpy.ndarrays
        v_b(dict): contains velocity biases whose keys are layer names and values are corresponding  velocity biases in the form of numpy.ndarrays
        m_w(dict):
        m_b(dict):



     Returns:
    -------
        v_w(dict): velocity weights after update (all algorithms except SGD)
        v_b(dict): velocity biases after update (all algorithms except SGD)
        m_w(dict):
        m_b(dict):
        W_dict(dict) : after update...contains weights whose keys are layer names and values are corresponding weights in the form of numpy.ndarrays
        B_dict(dict): after update...contains biases whose keys are layer names and values are corresponding biases in the form of numpy.ndarrays
    """

    if optimizer == 'sgd':
        W_dict,B_dict = SGD_step(W_dict,B_dict,dw,db,eta,batch_size, weight_decay)
        return W_dict,B_dict

    elif optimizer == 'momentum':
        v_w,v_b = momentum_velocity_update(v_w,v_b,momentum,dw,db,eta,batch_size)
        W_dict, B_dict = momentum_step(W_dict,B_dict,v_w,v_b,eta,batch_size, weight_decay)
        return v_w, v_b, W_dict, B_dict

    elif optimizer == 'nag':
        v_w,v_b = momentum_velocity_update(v_w,v_b,momentum,dw,db,eta,batch_size)
        W_dict, B_dict = NAG_step(W_dict,B_dict,momentum,v_w,v_b,dw,db,eta,batch_size,weight_decay)
        return v_w, v_b, W_dict, B_dict

    elif optimizer == 'rmsprop':
        v_w,v_b = rmsprop_update_grad(v_w,v_b,beta,dw,db,batch_size)
        W_dict, B_dict = rmsprop_step(W_dict,B_dict,v_w,v_b,eps,dw,db,eta,batch_size,weight_decay)
        return v_w, v_b, W_dict, B_dict

    elif optimizer == 'adam':
        m_w_hat ={}
        m_b_hat = {}
        v_w_hat = {}
        v_b_hat = {}
        m_w, m_b = adam_update_momentum(m_w,m_b,beta1,dw,db,batch_size)
        m_w_hat, m_b_hat = adam_bias_correction(m_w, m_b,t,beta1)
        v_w,v_b = rmsprop_update_grad(v_w,v_b,beta2,dw,db,batch_size)
        v_w_hat,v_b_hat = adam_bias_correction(v_w,v_b,t,beta2)
        W_dict, B_dict = adam_step(W_dict,B_dict,v_w_hat,v_b_hat,m_w_hat,m_b_hat,eps,eta,batch_size,weight_decay)
        return m_w, m_b, v_w, v_b, W_dict, B_dict

    elif optimizer == 'nadam':
        m_w_hat ={}
        m_b_hat = {}
        v_w_hat = {}
        v_b_hat = {}
        m_w, m_b = adam_update_momentum(m_w,m_b,beta1,dw,db,batch_size)
        m_w_hat, m_b_hat = adam_bias_correction(m_w, m_b,t,beta1)
        v_w,v_b = rmsprop_update_grad(v_w,v_b,beta2,dw,db,batch_size)
        v_w_hat,v_b_hat = adam_bias_correction(v_w,v_b,t,beta2)
        W_dict, B_dict = Nadam_step(t,W_dict,B_dict,dw,db,eta,batch_size,v_w_hat,v_b_hat,m_w_hat,m_b_hat,eps, beta1,weight_decay)
        return m_w, m_b, v_w, v_b, W_dict, B_dict
