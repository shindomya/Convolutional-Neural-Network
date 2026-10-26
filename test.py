import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import time
import os

#start timer
start = time.perf_counter()

#training data
def get_data():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train),(X_test, y_test) = mnist.load_data()
    return (X_test,y_test)
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
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
        
class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        #deletes all nums under 0
        self.output = np.maximum(0, inputs)
        

class Activation_Softmax:
    def forward(self,inputs):
        #e raised by all values from last output - e raised to highest value from last output
        #axis=1 calculates the rows instead of default None (coloums)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


(X_test,y_test) = get_data()
X_test_normalized = normalize_data(X_test)
X_test_flattened = shape2_1(X_test_normalized) 

layer1 = Layer_Dense(784, 64)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(64,10)
activation2 = Activation_Softmax()


if os.path.exists("mnist_weights.npz"):
    saved = np.load("mnist_weights.npz")

    layer1.weights = saved["weights1"]
    layer1.biases = saved["biases1"]

    layer2.weights = saved["weights2"]
    layer2.biases = saved["biases2"]

rand_num = np.random.randint(0,len(X_test))

layer1.forward(X_test_flattened[rand_num])
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(f"Correct answer: {y_test[rand_num]}")
print(f"Guessed number: {np.argmax(activation2.output)}")
print(f"Confidence: {np.max(activation2.output) * 100:.2f}%")


end = time.perf_counter()
print(f"Code execution took : {end-start:.4f} seconds")
