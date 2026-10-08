from keras.datasets import fashion_mnist
(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()
import numpy as np
import matplotlib.pyplot as plt

print("Training data shape : ", train_X.shape, train_Y.shape)
print("Testing data shape : ", test_X.shape, test_Y.shape)

#find unique testing number
classes = np.unique(train_Y)
nClasses = len(classes)
print("Total number of outputs : ", nClasses)
print("Output classes : ", classes)

plt.figure(figsize=[5,5])
#display first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap="gray")
plt.title("Ground Truth : {}".format(train_Y[0]))
#display first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap="gray")
plt.title("Ground Truth : {}".format(test_Y[0]))
