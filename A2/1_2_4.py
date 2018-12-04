import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

He_init = tf.variance_scaling_initializer()

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# Load data sets
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1,28*28)/255.0
X_test  = X_test.astype(np.float32).reshape(-1,28*28)/255.0
y_train = y_train.astype(np.int32)
y_test  = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# Initialize parameters
path = "./model_bn.ckpt"
node_num = 100
output_num = 5
learning_rate = 0.005
epoch_num = 1000
train_batch_size = 20
train_batch_num = int(epoch_num / train_batch_size)
max_check = 20
momentums = [0.85, 0.9, 0.95, 0.99]

# Modify data sets
X_train1 = X_train[y_train < 5]
y_train1 = y_train[y_train < 5]
X_valid1 = X_valid[y_valid < 5]
y_valid1 = y_valid[y_valid < 5]
X_test1  = X_test[y_test < 5]
y_test1  = y_test[y_test < 5]

X_test1_pic = np.reshape(X_test1, [-1, 28, 28])

# Initialize placeholders
X = tf.placeholder(tf.float32, name="X", shape=[None,28*28])
Y = tf.placeholder(tf.int64, name="Y", shape=[None,])
batch_size = tf.placeholder(tf.int64)
training = tf.placeholder_with_default(False, shape=(), name='training')
momentum = tf.placeholder(tf.float32)

# Initialize the data set
dataset = tf.data.Dataset.from_tensor_slices((X, Y))

# Shuffle, repeat, and batch the examples.
dataset = dataset.shuffle(X_train1.shape[0]).repeat().batch(batch_size)

# Create iterator
iter = dataset.make_initializable_iterator()
batch_X, batch_Y = iter.get_next()

# Construct a neural network
H1 = tf.layers.dense(batch_X, node_num, activation=tf.nn.elu,
                        kernel_initializer=He_init, name="H1")
BN1 = tf.layers.batch_normalization(H1, training=training, momentum=momentum)

H2 = tf.layers.dense(BN1, node_num, activation=tf.nn.elu,
                        kernel_initializer=He_init, name="H2")
BN2 = tf.layers.batch_normalization(H2, training=training, momentum=momentum)

H3 = tf.layers.dense(BN2, node_num, activation=tf.nn.elu,
                        kernel_initializer=He_init, name="H3")
BN3 = tf.layers.batch_normalization(H3, training=training, momentum=momentum)

H4 = tf.layers.dense(BN3, node_num, activation=tf.nn.elu,
                        kernel_initializer=He_init, name="H4")
BN4 = tf.layers.batch_normalization(H4, training=training, momentum=momentum)

H5 = tf.layers.dense(BN4, node_num, activation=tf.nn.elu,
                        kernel_initializer=He_init, name="H5")
BN5 = tf.layers.batch_normalization(H5, training=training, momentum=momentum)

Y_hat = tf.layers.dense(BN5, output_num, name="outputs")

# Calculate the loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2( \
    labels=tf.one_hot(batch_Y, 5, dtype=tf.float32), logits=Y_hat)
loss = tf.reduce_mean(cross_entropy)

# Calculate the accuracy
predict = tf.argmax(Y_hat, axis=1)
correct = tf.cast(tf.equal(batch_Y, predict), tf.float32)
accuracy = tf.reduce_mean(correct)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize tensorflow
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

# Initialize optimizer for batch normalization
BN_optimizer = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# Initialize saver
saver = tf.train.Saver()

# Iterate for different momentums
for curr_momentum in momentums :
    # Initialize parameters for early stopping
    check_count = 0
    best_loss = np.infty
    best_acc  = 0

    # Initialize plotting coordinates
    epoch_axis  = []
    epoch_loss_axis   = []
    epoch_acc_axis  = []
    valid_loss_axis   = []
    valid_acc_axis  = []

    # Start training
    with tf.Session() as sess :
        # Run the initializer
        sess.run(init_g)
        sess.run(init_l)

        for epoch in range(epoch_num) :
            # train for one epoch
            sess.run(iter.initializer, feed_dict={X:X_train1, Y:y_train1,
                        batch_size:train_batch_size})

            for batch in range(train_batch_num) :
                sess.run([optimizer, BN_optimizer],
                            feed_dict={training: True, momentum:curr_momentum})


            # Compute current loss and accuracy
            sess.run(iter.initializer, feed_dict={X:X_train1, Y:y_train1,
                        batch_size:X_train1.shape[0]})

            epoch_loss, epoch_acc = sess.run([loss, accuracy],
                        feed_dict={momentum:curr_momentum})
            epoch_acc = epoch_acc * 100

            print("Epoch =", epoch, "loss =", "{:.9f}".format(epoch_loss),
                    "precision =", "{:.2f}".format(epoch_acc), "per cent")

            # Calculate validation loss and accuracy for this epoch
            sess.run(iter.initializer, feed_dict={X:X_valid1, Y:y_valid1,
                        batch_size:X_valid1.shape[0]})

            valid_loss, valid_acc = sess.run([loss, accuracy],
                        feed_dict={momentum:curr_momentum})
            valid_acc = valid_acc * 100

            # Update the coordinates
            epoch_axis.append(epoch)
            epoch_loss_axis.append(epoch_loss)
            epoch_acc_axis.append(epoch_acc)
            valid_loss_axis.append(valid_loss)
            valid_acc_axis.append(valid_acc)

            if valid_loss < best_loss :
                save_path = saver.save(sess, path)
                best_loss = valid_loss
                best_acc  = valid_acc
                check_count = 0
            else :
                check_count += 1
                if check_count > max_check :
                    print("Early stopping!")
                    break

        # Calculate test precision using the best model
        saver.restore(sess, path)
        sess.run(iter.initializer, feed_dict={X:X_test1, Y:y_test1,
                    batch_size:X_test1.shape[0]})
        test_acc, test_correct, test_predict = sess.run([accuracy, correct,
                    predict], feed_dict={momentum:curr_momentum})
        precision = test_acc * 100
        print("Test precision is {:.2f} per cent".format(precision))

    # Show the final validation results
    print("Final validation loss =", "{:.9f}".format(best_loss))
    print("Final validation acc =", "{:.2f}".format(best_acc), "per cent")

    # Plot and show the graphs
    plt.plot(epoch_axis, epoch_loss_axis, label='Training Loss, momentum='+str(curr_momentum))
    plt.plot(epoch_axis, valid_loss_axis, label='Validation Loss, momentum='+str(curr_momentum))
    plt.legend()
    plt.show()

    plt.plot(epoch_axis, epoch_acc_axis, label='Training Precision in percentage, momentum='+str(curr_momentum))
    plt.plot(epoch_axis, valid_acc_axis, label='Validation Precision in percentage, momentum='+str(curr_momentum))
    plt.legend()
    plt.show()
