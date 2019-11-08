import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../tmp/data/", one_hot=True)


class NeuralNetwork:

    def __init__(self, batch_size=100):

        self.n_classes = 10
        self.batch_size = batch_size

        self.input_data = input_data
        self.in_data_size = 784
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

        hm_epochs = 20

        with tf.Session() as session:
            session.run(tf.initialize_all_variables())

            # training
            for epoch in range(hm_epochs):
                epoch_loss = 0
                for feedback in range(int(mnist.train.num_examples / self.batch_size)):
                    epoch_x, epoch_y = mnist.train.next_batch(self.batch_size)
                    feedback, c = session.run([optimizer, cost], feed_dict={self.x: epoch_x, self.y: epoch_y})
                    epoch_loss += c
                print('Epoch: ' + str(epoch + 1) + ' out of: ' + str(hm_epochs) + ' loss: ' + str(epoch_loss))

            # print current accuracy
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy: ' + str(accuracy.eval({self.x: mnist.test.images, self.y: mnist.test.labels})))
