import numpy as np
import pandas as pd
from mnist_utils import *
from data_processing import load_data


X_train, Y_train, X_val, Y_val = load_data()

layers_dims = [784, 10]


parameters_model = L_layer_model(X_train, Y_train, num_epochs=100, layers_dims=layers_dims,
                            learning_rate=0.05, mini_batch_size=32,
                            print_cost=True)

predict_model = predict(X_val, Y_val, parameters_model)