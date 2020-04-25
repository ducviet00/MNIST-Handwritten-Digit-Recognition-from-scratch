import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def to_categorical(y, num_classes, dtype='float32'):
    # from keras utils
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1

    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)

    return categorical


def load_data():
    train = pd.read_csv("./digit-recognizer/train.csv")
    #test = pd.read_csv("/content/drive/My Drive/20192/digit-recognizer/test.csv")

    Y_train = train["label"]

    # Drop 'label' column
    X_train = train.drop(labels=["label"], axis=1)

    # Free some space
    del train

    # Normalize the data
    X_train = np.array(X_train) / 255.0

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=8)

    # One-hot coding
    Y_train = to_categorical(Y_train, num_classes=10)
    Y_val = to_categorical(Y_val, num_classes=10)

    # Transpose X, Y
    X_train = np.array(X_train.T)
    Y_train = np.array(Y_train.T)
    X_val = np.array(X_val.T)
    Y_val = np.array(Y_val.T)

    return X_train, Y_train, X_val, Y_val
