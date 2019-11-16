import tensorflow as tf

from src.neuralnetwork.NeuralNetwork import NeuralNetwork


class MultilayerPerceptronNeuralNetwork(NeuralNetwork):

    n_nodes_hl = 500
    n_nodes_h2 = 1000
    n_nodes_h3 = 5000

    def neural_network_model(self, data):
        hidden_layer_1 = {'widths': tf.Variable(tf.random_normal([self.in_data_size, self.n_nodes_hl])),
                        'biases': tf.Variable(tf.random_normal([self.n_nodes_hl]))}

        hidden_layer_2 = {'widths': tf.Variable(tf.random_normal([self.n_nodes_hl, self.n_nodes_h2])),
                        'biases': tf.Variable(tf.random_normal([self.n_nodes_h2]))}

        hidden_layer_3 = {'widths': tf.Variable(tf.random_normal([self.n_nodes_h2, self.n_nodes_h3])),
                        'biases': tf.Variable(tf.random_normal([self.n_nodes_h3]))}

        output_layer = {'widths': tf.Variable(tf.random_normal([self.n_nodes_h3, self.n_classes])),
                        'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        layer_1 = tf.add(tf.matmul(data, hidden_layer_1['widths']), hidden_layer_1['biases'])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['widths']), hidden_layer_2['biases'])
        layer_2 = tf.nn.relu(layer_2)
        layer_3 = tf.add(tf.matmul(layer_2, hidden_layer_3['widths']), hidden_layer_3['biases'])
        layer_3 = tf.nn.relu(layer_3)

        output = tf.add(tf.matmul(layer_3, output_layer['widths']), output_layer['biases'])

        return output
