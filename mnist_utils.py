import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score

def theta(Z):

    ez = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    T = ez / np.sum(ez, axis=0)
    return T, Z


def theta_backward(dA, cache):
    Z = cache

    ez = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    T = ez / np.sum(ez, axis=0)

    dZ = dA * T * (1-T)
    return dZ


def relu(Z):
    R = np.maximum(0, Z)
    return R, Z


def relu_backward(dA, cache):
    Z = cache

    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ


def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],
                                                   layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.random.randn(layer_dims[l], 1) * 0.01

    return parameters


def linear_forward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "theta":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = theta(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A, parameters['W' + str(L)], parameters['b' + str(L)], activation="theta")
    caches.append(cache)
    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = - (1./m) * np.sum(np.multiply(Y, np.log(AL)))
    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1./m) * np.dot(dZ, A_prev.T)
    db = (1./m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "theta":
        dZ = theta_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)
                                                        ] = linear_activation_backward(dAL, current_cache, activation="theta")

    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - \
            learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - \
            learning_rate * grads["db" + str(l+1)]

    return parameters


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[1]
    mini_batches = []
    np.random.seed(seed)

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((10, m))

    num_complete_minibatches = math.floor(m / mini_batch_size)

    # Partition
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k *
                                  mini_batch_size:(k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k *
                                  mini_batch_size:(k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        end = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = shuffled_X[:,
                                  num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches
                                  * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def L_layer_model(X, Y, layers_dims, num_epochs=3000, learning_rate=0.0075, mini_batch_size=32, print_cost=False):
    costs = []                         # keep track of cost

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    seed = 14
    # Loop (gradient descent)
    for i in range(num_epochs):
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> THETA.
            AL, caches = L_model_forward(minibatch_X, parameters)

            # Compute cost.
            cost = compute_cost(AL, minibatch_Y)
            cost_total += cost
            # Backward propagation.
            grads = L_model_backward(AL, minibatch_Y, caches)

            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)

            # Print the cost every 100 training example

        cost_avg = cost_total / len(minibatches)
        if print_cost:
            print("Cost after iteration %i: %f" % (i, cost_avg))
        if print_cost:
            costs.append(cost_avg)  

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def predict(X, Y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    prediction = np.round(probas)

    return prediction