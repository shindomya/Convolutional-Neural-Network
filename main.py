import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import time

#start timer
start = time.perf_counter()

#training data
def get_data():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train),(X_test, y_test) = mnist.load_data()
    return (X_train,y_train),(X_test,y_test)
#minimize pixel value 
def normalize_data(y):
    return y / 255
#shape 2D vektor - 1D vektor
def shape2_1(x):
    x = x.reshape(-1,784)
    return x

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        #0.1 because of the bell-curve (value might go ober 1.0)
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self,inputs):
        #deletes all nums under 0
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self,inputs):
        #e raised by all values from last output - e raised to highest value from last output
        #axis=1 calculates the rows instead of default None (coloums)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


(X_train,y_train),(X_test,y_test) = get_data()
X_train_normalized = normalize_data(X_train)
X_train_flattened = shape2_1(X_train_normalized) 

layer1 = Layer_Dense(784, 64)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(64,10)
activation2 = Activation_Softmax()

layer1.forward(X_train_flattened[0])
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

#print(activation2.output)




#print(X_train_normalized)
#print(X_train_flattened[0])
#print(X_train_flattened[0].max())
#print(X_train_flattened[0].sum())
#plt.imshow(X_train[0], cmap = "gray")
#plt.show()

#end timer
end = time.perf_counter()
print(f"Code execution took : {end-start:.4f} seconds")
