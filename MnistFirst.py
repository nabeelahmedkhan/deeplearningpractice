# from keras.datasets import mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print(train_images.shape)
# my_slice = train_images[10:100]
# print(my_slice.shape)
import numpy as np
x = np.array([[23,4,3],
               [3,-3,54]])
y = np.array([[2,34,43],
             [34,-34,67]])
print(x.shape[0])
print(x.shape[1])

print(x.ndim)
print(y.shape)
def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x
print(naive_relu(x))
def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
                x[i, j] += y[i, j]
    return x
print(naive_add(x,y))
z = np.array([3,4])
print(x)
print(x+z)