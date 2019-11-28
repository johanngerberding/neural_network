import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# In this file I will implement a neural network from scratch only using numpy
# I also will test it on different datasets, maybe from scikit-learn

#1 Initialize the neural net parameters

def init_neural_net(neural_net_architecture, seed=42):
    # set fixed random seed to get same results 
    np.random.seed(seed)
    # dictionary for weight matrices and bias units
    parameter_values = {}
    # random initialization of weights

    for idx, layer in enumerate(neural_net_architecture):
        layer_num = idx + 1
        layer_input_dim = layer["input_dim"]
        layer_output_dim = layer["output_dim"]

        # Random initialization of the weights
        # shape of W(l) = (n(l), n(l-1))  --> n(l) = output_dims, n(l-1) = input_dims
        parameter_values["W{0}".format(layer_num)] = np.random.randn(layer_output_dim,
        layer_input_dim)
        # Random initialization of the bias unit
        # b(l) = (n(l), 1)
        parameter_values["b{0}".format(layer_num)] = np.random.randn(layer_output_dim, 1)
    
    return parameter_values

# Implementations of activation functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

#2 Forward Propagation
def forward_prop(X, nn_params, neural_net_architecture):
    # values you need to memorize for backprop
    cache = {}
    # Initialize A with input values
    A_curr = X

    for idx, layer in enumerate(neural_net_architecture):
        # load W and b from params dictionary
        layer_num = idx + 1
        A_prev = A_curr

        W_curr = nn_params["W{0}".format(layer_num)]
        b_curr = nn_params["b{0}".format(layer_num)]
        activation = layer["activation"]
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation is "relu":
            A_curr = relu(Z_curr)
        elif activation is "sigmoid":
            A_curr = sigmoid(Z_curr)
        else:
            raise Exception('Activation function not supported')

        cache["A{0}".format(idx)] = A_prev
        cache["Z{0}".format(layer_num)] = Z_curr
    
    return A_curr, cache

# Implement cost function (for classification --> cross entropy)
def calc_cost(y, y_pred):
    m = y_pred.shape[1]
    cost = -1 / m * (np.dot(y, np.log(y_pred).T) + np.dot(1 - y, np.log(1 - y_pred).T))
    # Remove single-dim entries from the shape of cost
    return np.squeeze(cost)

#3 Backward Propagation

def one_layer_backprop(dA, Z_curr, A_prev, W_curr, b_curr, activation="sigmoid"):
    
    # Calc dZ with the derivative of sigmoid or relu
    if activation == "sigmoid":
        dZ = dA * sigmoid(Z_curr) * (1 - sigmoid(Z_curr))
    elif activation == "relu":
        dZ = np.array(dA, copy=True)
        dZ[Z_curr <= 0] = 0
    else:
        raise Exception("You can only choose between relu and sigmoid here.")
    
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W_curr.T, dZ)

    return dA_prev, dW, db
    

def backward_prop(Y, Y_hat, caches, nn_params, neural_net_architecture):
    gradients = {}
    L = len(caches)
    m = Y_hat.shape[1]
    Y = Y.reshape(Y_hat.shape)

    # derivative of the cross-entropy loss function
    # y_hat = predicted value
    dA_prev = -(np.divide(Y, Y_hat) - np.divide(1-Y, 1-Y_hat))

    for idx, layers in reversed(list(enumerate(neural_net_architecture))):
        layer_idx = idx + 1
        activation = layers["activation"]

        dA_curr = dA_prev

        A_prev = caches["A{0}".format(idx)]
        Z_curr = caches["Z{0}".format(layer_idx)]
        W_curr = nn_params["W{0}".format(layer_idx)]
        b_curr = nn_params["b{0}".format(layer_idx)]
        
        dA_prev, dW_curr, db_curr = one_layer_backprop(dA_curr, Z_curr, A_prev, W_curr, b_curr, activation)
        
        gradients["dW{0}".format(layer_idx)] = dW_curr
        gradients["db{0}".format(layer_idx)] = db_curr
    
    return gradients


def update_params(nn_params, gradients, learning_rate):
    L = len(nn_params) // 2
    for l in range(L):
        nn_params["W{0}".format(l+1)] = nn_params["W{0}".format(l+1)] - learning_rate * gradients["dW{0}".format(l+1)]
        nn_params["b{0}".format(l+1)] = nn_params["b{0}".format(l+1)] - learning_rate * gradients["db{0}".format(l+1)]

    return nn_params


def train(X, Y, neural_net_architecture, epochs, learning_rate):
    nn_params = init_neural_net(neural_net_architecture)
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, caches = forward_prop(X, nn_params, neural_net_architecture)
        cost = calc_cost(Y, Y_hat)
        cost_history.append(cost)
        acc = calc_accuracy(Y, Y_hat)
        accuracy_history.append(acc)
        gradients = backward_prop(Y, Y_hat, caches, nn_params, neural_net_architecture)
        nn_params = update_params(nn_params, gradients, learning_rate)
    
    return nn_params, cost_history, accuracy_history

def calc_accuracy(Y, Y_hat):
    Y_hat_ = np.copy(Y_hat)
    Y_hat_[Y_hat_ > 0.5] = 1
    Y_hat_[Y_hat_ <= 0.5] = 0
    return (Y_hat_ == Y).all(axis=0).mean()


#4 Test our Implementation

def main():

    # Lets load a standard scikit-learn dataset
    X, y = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.2)

    X_plt = X.copy()
    y_plt = y.copy()
    
    X = np.transpose(X)
    y = np.transpose(y.reshape((y.shape[0], 1)))

    

    # print(X.shape)
    # print(y.shape)
    

    neural_net_architecture = [
        {"input_dim": 2, "output_dim": 5, "activation": "relu"},
        {"input_dim": 5, "output_dim": 4, "activation": "relu"},
        {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
    ]

    
    nn_params, cost_history, acc_history = train(X, y, neural_net_architecture, 600000, 0.0001)
    
    #plt.plot(acc_history, "g")
    #plt.plot(cost_history, "r")
    #plt.show()

    # idea is to use forward prop to calc output of the model using the learned parameters
    x_db = np.arange(-2, 3, 0.01).tolist()
    
    X_db = np.transpose([np.tile(x_db, len(x_db)), np.repeat(x_db, len(x_db))])
    
    print(X_db.shape)

    db, _ = forward_prop(X_db.T, nn_params, neural_net_architecture)
    db = np.squeeze(db)

    db_ = np.copy(db)
    db_[db_ > 0.5] = 1
    db_[db_ <= 0.5] = 0


    #plt.plot(db)
    plt.scatter(X_db[:, 0], X_db[:, 1], c=db_)
    plt.show()
    # then we will plot the decision boundary

    # forward_prop(X, nn_params, neural_net_architecture)
    

    # plt.plot(cost_history)
    # plt.show()
 

if __name__ == "__main__":
    main()
