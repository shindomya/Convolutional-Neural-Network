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

def normalize_data(X):
    return X / 255


(X_train,y_train),(X_test,y_test) = get_data()

X_train_normalized = normalize_data(X_train)


print(X_train[0])
#plt.imshow(X_train[0], cmap = "gray")
#plt.show()
#end timer


end = time.perf_counter()
print(f"Code execution took : {end-start:.4f} seconds")
