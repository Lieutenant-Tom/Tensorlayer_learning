import tensorlayer as tl
import tensorflow as tf
import numpy as np

init_scale = 0.1
batch_size = 20
hidden_size = 200
num_steps = 4
sequence_length = 90


word_embeddings = np.load("vec.npy")
word_embedding = tf.get_variable(initializer=word_embeddings,name = 'word_embedding')
input_data = tf.placeholder(tf.int32, [batch_size, sequence_length])

network = tl.layers.InputLayer(tf.nn.embedding_lookup(word_embedding, input_data), name='input_layer')
# network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, W_init=tf.truncated_normal_initializer(stddev=0.1), name ='relu_layer')
network = tl.layers.DropoutLayer(network, keep=0.5, name="drop1")
network = tl.layers.RNNLayer(network,
                        cell_fn=tf.nn.rnn_cell.BasicLSTMCell, #tf.nn.rnn_cell.BasicLSTMCell,
                        cell_init_args={'forget_bias': 0.0, 'state_is_tuple': True},
                        n_hidden=hidden_size,
                        initializer=tf.random_uniform_initializer(-init_scale, init_scale),
                        n_steps=num_steps,
                        return_last=False,
                        name='basic_lstm_layer1')
lstm1 = network


# network.print_layers()







