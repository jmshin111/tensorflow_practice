import tensorflow as tf

import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


x = 0

y = 1
w = tf.random.normal([1], 0, 1)
b = tf.random.normal([1], 0, 1)

# sigmoid function

#
# for i in range(1000):
#     output = sigmoid(x+w+1*b)
#     error = y - output
#     w = w + x * 0.1*error
#     b = b +1*0.1*error
#     if i % 100 == 99:
#         print (i, error, output )


# 첫번째 신경만 네트워크 AND:
import numpy as np

x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y = np.array([[1], [0], [0], [0]])
w = tf.random.normal([2], 0, 1)
b = tf.random.normal([1], 0, 1)

b_x = 1

for i in range(2000):
    error_sum = 0
    for j in range(4):
        output = sigmoid(np.sum(x[j] * w) + b_x * b)
        error = y[j][0] - output
        w = w + x[j] * 0.1 * error
        b = b + b_x * 0.1 * error
        error_sum += error

    if i * 200 == 199:
        print(i, error_sum)
