import tensorflow as tf

from src.neuralnetwork.NeuralNetwork import NeuralNetwork


class ConvolutionNeuralNetwork(NeuralNetwork):

    def __conv_2d(self, x, weights, strides=1):
        return tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding='SAME')

    def __max_pool_2d(self, x, size=2):
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

    def neural_network_model(self, data):
        weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
                   'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                   'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
                   'out': tf.Variable(tf.random_normal([1024, self.n_classes]))}

        biases = {'B_conv1': tf.Variable(tf.random_normal([32])),
                  'B_conv2': tf.Variable(tf.random_normal([64])),
                  'B_fc': tf.Variable(tf.random_normal([1024])),
                  'out': tf.Variable(tf.random_normal([self.n_classes]))}

        x = tf.reshape(data, shape=[-1, 25, 25, 1])

        conv1 = self.__conv_2d(x, weights['W_conv1'])
        conv1 = self.__max_pool_2d(conv1)

        conv2 = self.__conv_2d(conv1, weights['W_conv2'])
        conv2 = self.__max_pool_2d(conv2)

        fc = tf.reshape(conv2, [-1, 7*7*64])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['B_fc'])

        output = tf.matmul(fc, weights['out'])+biases['out']

        return output
