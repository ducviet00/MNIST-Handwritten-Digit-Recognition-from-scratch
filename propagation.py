import numpy as np
from mnist_utils import *


def initialize_parameters(layer_dims):
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
        A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="theta")

    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]

    loss = - np.sum(np.multiply(Y, np.log(AL)))
    cost = loss / m
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

    
