import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pickle
import pygame
import operator
import copy

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


# functions to safe and load 
def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


#4 Test our Implementation

def main():

    # Lets load a standard scikit-learn dataset
    X, y = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.2)

    X_plt = X.copy()
    y_plt = y.copy()
    
    X = np.transpose(X)
    y = np.transpose(y.reshape((y.shape[0], 1)))
    

    neural_net_architecture = [
        {"input_dim": 2, "output_dim": 5, "activation": "relu"},
        {"input_dim": 5, "output_dim": 6, "activation": "relu"},
        {"input_dim": 6, "output_dim": 5, "activation": "relu"},
        {"input_dim": 5, "output_dim": 4, "activation": "relu"},
        {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
    ]

    # create a list of all neurons we want to display
    neurons = []
    input_layer = ["x" + str(i) for i in range(1, len(X) + 1)]
    neurons.append(input_layer)

    for i in range(len(neural_net_architecture)):
        neurons.append(["a({0},{1})".format(i+1, j) for j in range(1, neural_net_architecture[i]["output_dim"] + 1) ])

    # print(neurons)

    # first of all, lets build a function which visualizes our network on the screen
    # I think we use pygame for this
    # initialize pygame
    pygame.init()

    # create colors
    WHITE = pygame.Color(255, 255, 255)
    RED = pygame.Color(255, 0, 0)
    BLUE = pygame.Color(0, 0, 255)
    GREEN = pygame.Color(0, 255, 0)
    
    # set width and height of the screen
    width = 1000
    height = 900

    # maybe we want to adjust the size of the neural net visualization space
    nn_width = width
    nn_height = height

    screen = pygame.display.set_mode((width, height))

    # calc the width per section or level horizontally based on the nn_architecture
    horizontal_levels_width = nn_width // (len(neural_net_architecture) + 1)
    # print("Horizontal width " + str(horizontal_levels_width))

    # calc the height per section or level
    max_val = 0
    nn_architecture = copy.deepcopy(neural_net_architecture)
    
    for dic in nn_architecture:
        dic.pop("activation", None)
        maximum = max(dic.values())
        if maximum > max_val:
            max_val = maximum


    vertical_levels_height = nn_height // max_val
    # print("Vertical height " + str(vertical_levels_height))

    # set the title of the display
    pygame.display.set_caption("Neural Network: Visualization")
    pygame.mouse.set_visible(1)
    pygame.key.set_repeat(1, 30)

    # create a clock object to control the frame rate
    clock = pygame.time.Clock()

    font = pygame.font.Font('freesansbold.ttf', 16) 

    # create a list of neuron connections (tuples)
    neuron_connections = []
    counter = 0
    for i in range(len(neurons)-1):
        layer_1 = neurons[counter]
        layer_2 = neurons[counter + 1]
        neuron_connections.append([[a,b] for a in layer_1 for b in layer_2])
        counter += 1

    
    
    # lets draw a single neuron
    # we have to base the radius on the size of the grid
    rad_max = min(vertical_levels_height, horizontal_levels_width)
    radius = rad_max // 2 * 0.5
    radius = int(radius)

    neuron_coordinates = {}

    for x, layer in enumerate(neurons):
        # x value for all neurons per layer
        x_neuron = horizontal_levels_width // 2 + (x * horizontal_levels_width)
        # set the y-steop value
        y_step = height // (len(layer) + 1)
            # init y with step value
        y_neuron = y_step
        for neuron in layer:
            neuron_coordinates[neuron] = (x_neuron, y_neuron)
            y_neuron += y_step

    
    #try to load paramters from pickle files
    nn_params = load_obj("nn_parameters")
    cost_history = load_obj("cost_history")
    acc_history = load_obj("acc_history") 

    max_param = 1
    min_param = -1  

    for i in range(1,6):
        # ndarray with parameter values
        parameters = nn_params["W{0}".format(i)].T
        # flatten the parameter list to ensure it has the same form as connections
        flatten_params = []
        for sublist in parameters:
            for item in sublist:
                flatten_params.append(item)

        connection = neuron_connections[i-1]
        
        count = 0
        for el in connection:
            val = flatten_params[count]
            if val > max_param:
                max_param = val
            elif val < min_param:
                min_param = val
            el.append(val)
            count += 1


    param_range = max_param - min_param

    print("maximum weight value: " + str(max_param))
    print("minimum weight value: " + str(min_param))
    print("param range: "+ str(param_range))

    print("------------------------")
    
    # print(neuron_connections[0][0])
    # erst x1 zu allen a1 neuronen, dann x2 zu allen a1 neuronen


    running = True
    while running:
        # set the frame rate to 20 per second
        clock.tick(40)
 
        # fill screen-surface (RGB = 0, 0, 0)
        screen.fill((255, 255, 255))

        # so now we have to draw lines between the neurons
        for connections in neuron_connections:
            for connection in connections:
                start = connection[0]
                end = connection[1]
                line_width = int((connection[2] + np.absolute(min_param) + 1))
                x_start, y_start = neuron_coordinates[start]
                x_end, y_end = neuron_coordinates[end]
                pygame.draw.line(screen, RED, (x_start, y_start), (x_end, y_end), line_width)

        for neuron in neuron_coordinates.items():
            pygame.draw.circle(screen, BLUE, (neuron[1][0], neuron[1][1]), radius)
            text = font.render(neuron[0], True, WHITE, BLUE)
            textRect = text.get_rect()
            textRect.center = (neuron[1][0], neuron[1][1])
            screen.blit(text, textRect)

        


        # event loop
        for event in pygame.event.get():
            # quit the game
            if event.type == pygame.QUIT:
                running = False
 
            
            if event.type == pygame.KEYDOWN:
                # if you click on escape the loop ends
                if event.key == pygame.K_ESCAPE:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
 
        # show content of screen
        pygame.display.flip()


    # nn_params, cost_history, acc_history = train(X, y, neural_net_architecture, 300000, 0.0001)
    
    # safe these values
    # important -> key/value pairs are converted to strings, so if you load the 
    # dictionary from the json file, it may be not the same as before

    # save_obj(nn_params, "nn_parameters")
    # save_obj(cost_history, "cost_history")
    # save_obj(acc_history, "acc_history")

    # print(neuron_coordinates)

    

    # print(nn_params["W1"][0][0])
    # plt.plot(acc_history, "g", label="accuracy")
    # plt.plot(cost_history, "r", label="cost")
    # plt.legend()
    # plt.show()

    # idea is to use forward prop to calc output of the model using the learned parameters
    # feed the forward prop the cartesian product between -2 and 3 to plot the decision boundary 
    x_db = np.arange(-2, 2.5, 0.05).tolist()
    X_db = np.transpose([np.tile(x_db, len(x_db)), np.repeat(x_db, len(x_db))])

    db, _ = forward_prop(X_db.T, nn_params, neural_net_architecture)
    db = np.squeeze(db)

    db_ = np.copy(db)
    db_[db_ > 0.5] = 1
    db_[db_ <= 0.5] = 0


    colors = ['g' if i == 1 else 'r' for i in y_plt]

    # plt.scatter(X_db[:, 0], X_db[:, 1], c=db_)
    # plt.scatter(X_plt[:, 0], X_plt[:, 1], c=colors)
    # plt.show()
 
    
 

if __name__ == "__main__":
    main()
