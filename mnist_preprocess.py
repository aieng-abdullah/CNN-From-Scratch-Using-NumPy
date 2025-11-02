import numpy as np
from keras.datasets import mnist

def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    x_train = x_train[:2000]
    y_train = y_train[:2000]
    x_test = x_test[:500]
    y_test = y_test[:500]

    return x_train, y_train, x_test, y_test

#Test
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    print("MNIST preprocessed successfully")
    print("x_train:", x_train.shape, "y_train:", y_train.shape)
    print("x_test:", x_test.shape, "y_test:", y_test.shape)
