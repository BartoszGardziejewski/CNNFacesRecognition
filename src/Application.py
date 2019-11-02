import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../tmp/data/", one_hot=True)

n_nodes_hl = 500

n_classes = 10
batch_size = 100

in_data_size = 784  # photos 28x28
x = tf.placeholder('float', [None, in_data_size])
y = tf.placeholder('float')


def neural_network_model(data):

    hidden_layer = {'widths': tf.Variable(tf.random_normal([in_data_size, n_nodes_hl])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl]))}

    output_layer = {'widths': tf.Variable(tf.random_normal([n_nodes_hl, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    layer_1 = tf.add(tf.matmul(data, hidden_layer['widths']), hidden_layer['biases'])
    layer_1 = tf.nn.relu(layer_1)

    output = tf.add(tf.matmul(layer_1, output_layer['widths']), output_layer['biases'])

    return output


def train_neural_network(train_data):
    prediction = neural_network_model(train_data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 20

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        # training
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for feedback in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                feedback, c = session.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch: ' + str(epoch+1) + ' out of: ' + str(hm_epochs) + ' loss: ' + str(epoch_loss))

        # print current accuracy
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ' + str(accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))


train_neural_network(x)
