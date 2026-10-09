import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib as plt
import time
#start timer
start = time.perf_counter()

#training data
mnist = tf.keras.datasets.mnist
(X_train, y_train),(X_test, y_test) + mnist.load_data()

#end timer
end = time.perf_counter()
print(f"Code execution took : {end-start:.4f} seconds")
