import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np

def new_weights(shape, mu=0, sigma=0.1, name='W'):
    return tf.Variable(tf.truncated_normal(shape, mean=mu, stddev=sigma),name=name)

def new_biases(length, name='B'):
    return tf.Variable(tf.zeros(length),name=name)

def evaluate(accuracy_operation, X_data, y_data, x, y, keep_prob, BATCH_SIZE=256):
    num_examples = len(X_data)
    total_accuracy = 0

    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        feed_dict = {x: batch_x, y: batch_y, keep_prob: 1}
        accuracy = sess.run(accuracy_operation, feed_dict=feed_dict)
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True, name='LayerX'):  # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape, name='{}W'.format(name))
    biases = new_biases(length=num_filters, name='{}B'.format(name))
    layer = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME', name=name)
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME', name='{}pooling'.format(name))

    layer = tf.nn.relu(layer, name='{}relu'.format(name))
    print ('Layer: {}'.format(shape))
    return layer, weights

def flatten_layer(layer, name='flatten'):
    return flatten(layer)

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True, name='fclayer'): # Use Rectified Linear Unit (ReLU)?
    weights = new_weights(shape=[num_inputs, num_outputs], name='{}w'.format(name))
    biases = new_biases(length=num_outputs, name='{}b'.format(name))
    layer = tf.matmul(input, weights, name='{}matul'.format(name)) + biases
    if use_relu:
        layer = tf.nn.relu(layer,name='{}Relu'.format(name))

    return layer


def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys
