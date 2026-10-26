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
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

        
class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        #deletes all nums under 0
        self.output = np.maximum(0, inputs)
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0] = 0

        
        

class Activation_Softmax:
    def forward(self,inputs):
        #e raised by all values from last output - e raised to highest value from last output
        #axis=1 calculates the rows instead of default None (coloums)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss_CategoricalCrossentropy:
    def calculate(self,y_pred,y_solution):
        #number of poss answers
        samples = len(y_pred)
        #0 log error / set limits
        #1e = 10**
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        correct_confidences = y_pred_clipped[range(samples),y_solution]
        neg_log = -np.log(correct_confidences)
        return np.mean(neg_log)
    def one_hot_matrix(self,labels):
        #initialize a matrix with 0 only
        clean_matrix = np.zeros((len(labels),10))
        #put the correct answers into the matrix (1=answer,0=everything else)
        clean_matrix[np.arange(len(labels)), labels] = 1
        return clean_matrix
    def backward(self,output_value,clean_matrix,division):
        dvalue = (output_value - clean_matrix) / division
        return dvalue


(X_train,y_train),(X_test,y_test) = get_data()
X_train_normalized = normalize_data(X_train)
X_train_flattened = shape2_1(X_train_normalized) 

layer1 = Layer_Dense(784, 64)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(64,10)
activation2 = Activation_Softmax()

cce = Loss_CategoricalCrossentropy()


if os.path.exists("mnist_weights.npz"):
    saved = np.load("mnist_weights.npz")

    layer1.weights = saved["weights1"]
    layer1.biases = saved["biases1"]

    layer2.weights = saved["weights2"]
    layer2.biases = saved["biases2"]

learning_rate = 0.01
for epoche in range(50): 
    for i in range(0, len(X_train_flattened), 32):
        layer1.forward(X_train_flattened[i:i+32])
        activation1.forward(layer1.output)

        layer2.forward(activation1.output)
        activation2.forward(layer2.output)

        loss = cce.calculate(activation2.output,y_train[i:i+32])
        matrix_solution = cce.one_hot_matrix(y_train[i:i+32])
    
        dvalues = cce.backward(activation2.output,matrix_solution,32)
    
        layer2.backward(dvalues)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)

        layer1.weights -= learning_rate * layer1.dweights
        layer1.biases -= learning_rate * layer1.dbiases

        layer2.weights -= learning_rate * layer2.dweights
        layer2.biases -= learning_rate * layer2.dbiases
    print(f"loss {epoche}: {loss}")
    np.savez(
            "mnist_weights.npz",
            weights1 = layer1.weights,
            biases1 = layer1.biases,
            weights2 = layer2.weights,
            biases2 = layer2.biases
            )
        
       

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
