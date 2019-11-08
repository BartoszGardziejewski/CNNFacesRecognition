import tensorflow as tf
import numpy as np


class NeuralNetwork:

    def __init__(self, train_x, train_y, test_x, test_y, number_of_classes, batch_size=100):

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.n_classes = number_of_classes
        self.batch_size = batch_size

        self.in_data_size = len(train_x[0])
        self.x = tf.placeholder('float', [None, self.in_data_size])
        self.y = tf.placeholder('float')

    n_nodes_hl = 500

    def __neural_network_model(self, data):
        hidden_layer = {'widths': tf.Variable(tf.random_normal([self.in_data_size, self.n_nodes_hl])),
                        'biases': tf.Variable(tf.random_normal([self.n_nodes_hl]))}

        output_layer = {'widths': tf.Variable(tf.random_normal([self.n_nodes_hl, self.n_classes])),
                        'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        layer_1 = tf.add(tf.matmul(data, hidden_layer['widths']), hidden_layer['biases'])
        layer_1 = tf.nn.relu(layer_1)

        output = tf.add(tf.matmul(layer_1, output_layer['widths']), output_layer['biases'])

        return output

    def train_neural_network(self):
        prediction = self.__neural_network_model(self.x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        # learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        hm_epochs = 100

        with tf.Session() as session:
            session.run(tf.initialize_all_variables())

            # training
            for epoch in range(hm_epochs):
                epoch_loss = 0

                for index in range(0, len(self.train_x), self.batch_size):
                    start = index
                    end = index + self.batch_size

                    batch_x = np.array(self.train_x[start:end])
                    batch_y = np.array(self.train_y[start:end])

                    feedback, c = session.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})
                    epoch_loss += c

                print('Epoch: ' + str(epoch + 1) + ' out of: ' + str(hm_epochs) + ' loss: ' + str(epoch_loss))

            # print current accuracy
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy: ' + str(accuracy.eval({self.x: self.test_x, self.y: self.test_y})))
