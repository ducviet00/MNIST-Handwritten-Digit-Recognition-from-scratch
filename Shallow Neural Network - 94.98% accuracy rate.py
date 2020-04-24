import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score, confusion_matrix
from mnist_utils import *
from data_processing import load_data


X_train, Y_train, X_val, Y_val = load_data()


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = theta(Z2)

    cache = {"A1": A1,
             "Z1": Z1,
             "A2": A2,
             "Z2": Z2}
    return A2, cache


def compute_cost(A2, Y):
    m = Y.shape[1]  # number of training examples
    loss = - np.sum(np.multiply(Y, np.log(A2)))
    cost = loss / m
    cost = np.squeeze(cost)
    return cost


def backward_propagation(X, Y, parameters, cache):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.array(dA1, copy=True)
    dZ1[Z1 <= 0] = 0
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW2": dW2,
             "db2": db2,
             "dW1": dW1,
             "db1": db1}

    return grads


def update_parameters(parameters, grads, learning_rate=0.5):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def shallow_model(X, Y, layer_dims, num_epochs=3000, learning_rate=0.5, mini_batch_size=64, print_cost=False):
    n_x = X.shape[0]  # number of input layer
    n_h = 20  # number of nodes in hidden layer
    n_y = Y.shape[0]  # number of output layer
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    seed = 10
    m = X.shape[1]
    costs = []

    for i in range(num_epochs):
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # foward propagation
            A2, cache = forward_propagation(minibatch_X, parameters)

            cost_total += compute_cost(A2, minibatch_Y)

            grads = backward_propagation(
                minibatch_X, minibatch_Y, parameters, cache)
            # update parameter
            parameters = update_parameters(parameters, grads, learning_rate)

        cost_avg = cost_total / len(minibatches)
        if print_cost:
            print("Cost after iteration %i: %f" % (i, cost_avg))
        if print_cost:
            costs.append(cost_avg)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    return parameters


def predict(parameters, X):
    A, cache = forward_propagation(X, parameters)
    predictions = np.round(A)
    return predictions


parameters_model = shallow_model(
    X_train, Y_train, num_epochs=50, learning_rate=0.05, mini_batch_size=32, print_cost=True)


Y_hat = predict(parameters_model, X_val)

# Print image
image_id = 188
plt.imshow(X_val[:, image_id].reshape(28, 28), cmap=matplotlib.cm.binary)
plt.axis("off")
plt.show()

# Prediction
Y_hat_label = np.argmax(Y_hat, axis=0)
print(Y_hat_label[image_id])

# True label
Y_val_label = np.argmax(Y_val, axis=0)
print(Y_val_label[image_id])
score = 100*accuracy_score(Y_val_label, Y_hat_label)
print("Model accuracy score: %0.2f" % score)
