# Inspired by: https://codesachin.wordpress.com/2016/01/23/predicting-trigonometric-waves-few-steps-ahead-with-lstms-in-tensorflow/

import tensorflow as tf
from tensorflow.models.rnn.rnn import *
from numpy import array, sin, cos, pi
from random import random
import matplotlib.pyplot as plt


# TASK 1: try different training sizes
TRAIN_SIZE = 100000
TEST_SIZE = 1000

# In the following, there are two time series
INPUT_DIM = 2

# How many steps ahead we are trying to predict
LAG = 23


def create_plot(x_axis, prediction, ground_truth, title):
    plt.figure()
    plt.title(title)
    plt.plot(x_axis, prediction, 'r-', label='Prediction')
    plt.plot(x_axis, ground_truth, 'b-', label='Ground truth')
    plt.legend()
    return


# Random initial angles
angle1 = random()
angle2 = random()

# The total 2*pi cycle would be divided into 'frequency'
# number of steps
frequency1 = 300
frequency2 = 200


# TASK 2: change generation of time series
def get_sample():
    """
    Returns a sample of both time series.
    """
    global angle1, angle2
    angle1 += 2 * pi / float(frequency1)
    angle2 += 2 * pi / float(frequency2)
    angle1 %= 2 * pi
    angle2 %= 2 * pi
    return array([array([
        15 + 5 * sin(angle1) + 10 * cos(angle2),
        25 + 7 * sin(angle2) + 14 * cos(angle1)])])


sliding_window = []

for i in range(LAG - 1):
    sliding_window.append(get_sample())


def get_pair():
    """
    Returns an (current, later) pair, where 'later' is 'lag'
    steps ahead of the 'current' on the wave(s) as defined by the
    frequency.
    """
    global sliding_window
    sliding_window.append(get_sample())
    input_value = sliding_window[0]
    output_value = sliding_window[-1]
    sliding_window = sliding_window[1:]
    return input_value, output_value


# To maintain state
last_value = array([0 for i in range(INPUT_DIM)])
last_derivative = array([0 for i in range(INPUT_DIM)])


def get_total_input_output():
    """
    Returns the overall Input and Output as required by the model.
    The input is a concatenation of the wave values, their first and
    second derivatives.
    """
    global last_value, last_derivative
    raw_i, raw_o = get_pair()
    raw_i = raw_i[0]
    l1 = list(raw_i)
    derivative = raw_i - last_value
    l2 = list(derivative)
    last_value = raw_i
    l3 = list(derivative - last_derivative)
    last_derivative = derivative
    return array([l1 + l2 + l3]), raw_o


# Task 3: amend network, such as different learning rate or add more LSTM layers

# Input layer for 6 inputs, batch size 1
input_layer = tf.placeholder(tf.float32, [1, INPUT_DIM * 3])

# Initialization of LSTM layer
lstm_layer = rnn_cell.BasicLSTMCell(INPUT_DIM * 3)
# LSTM state, initialized to 0
lstm_state = tf.Variable(tf.zeros([1, lstm_layer.state_size]))
# Connect input layer to LSTM
lstm_output, lstm_state_output1 = lstm_layer(input_layer, lstm_state)
# Update of LSTM state
lstm_update = lstm_state.assign(lstm_state_output1)

# Regression output layer
# Weights and biases
output_W = tf.Variable(tf.truncated_normal([INPUT_DIM * 3, INPUT_DIM]))
output_b = tf.Variable(tf.zeros([INPUT_DIM]))
output_layer = tf.matmul(lstm_output, output_W) + output_b

# Input for correct output (for training)
output_ground_truth = tf.placeholder(tf.float32, [1, INPUT_DIM])

# Sum of squared error terms
error = tf.pow(tf.sub(output_layer, output_ground_truth), 2)

# Adam optimizer
optimizer = tf.train.AdamOptimizer(0.0006).minimize(error)

# Session
sess = tf.Session()
sess.run(tf.initialize_all_variables())

print 'Training'

ground_truth1 = []
ground_truth2 = []
prediction1 = []
prediction2 = []
x_axis = []

for i in range(1, TRAIN_SIZE+1):
    input_v, output_v = get_total_input_output()
    _, _, network_output = sess.run([lstm_update,
                                     optimizer,
                                     output_layer],
                                    feed_dict={
                                        input_layer: input_v,
                                        output_ground_truth: output_v})

    ground_truth1.append(output_v[0][0])
    ground_truth2.append(output_v[0][1])
    prediction1.append(network_output[0][0])
    prediction2.append(network_output[0][1])
    x_axis.append(i)
    if i % 1000 == 0:
        print 'Trained on {0} pairs'.format(i)


# Task 4: look at parts of different size of the time series
#create_plot(x_axis[94000:], prediction1[94000:], ground_truth1[94000:], 'Training performance on time series 1')
create_plot(x_axis, prediction1, ground_truth1, 'Training performance on time series 1')
plt.show()
create_plot(x_axis, prediction2, ground_truth2, 'Training performance on time series 2')
plt.show()

print 'Testing'

for i in range(200):
    get_total_input_output()

# Flush LSTM state for testing (learned weights do not change)
sess.run(lstm_state.assign(tf.zeros([1, lstm_layer.state_size])))

ground_truth1 = []
ground_truth2 = []
prediction1 = []
prediction2 = []
x_axis = []

for i in range(TEST_SIZE):
    input_v, output_v = get_total_input_output()
    _, network_output = sess.run([lstm_update,
                                  output_layer],
                                 feed_dict={
                                     input_layer: input_v,
                                     output_ground_truth: output_v})

    ground_truth1.append(output_v[0][0])
    ground_truth2.append(output_v[0][1])
    prediction1.append(network_output[0][0])
    prediction2.append(network_output[0][1])
    x_axis.append(i)

create_plot(x_axis, prediction1, ground_truth1, 'Test performance on time series 1')
plt.show()
create_plot(x_axis, prediction2, ground_truth2, 'Test performance on time series 2')
plt.show()
