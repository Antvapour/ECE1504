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
path = "./model2.ckpt"
node_num = 140
output_num = 5
learning_rate = 0.1
epoch_num = 1000
train_batch_size = 500
train_batch_num = int(epoch_num / train_batch_size)
max_check = 30
check_count = 0
best_loss = np.infty
best_error  = 0

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

# Initialize the data set
dataset = tf.data.Dataset.from_tensor_slices((X, Y))

# Shuffle, repeat, and batch the examples.
dataset = dataset.shuffle(X_train1.shape[0]).repeat().batch(batch_size)

# Create iterator
iter = dataset.make_initializable_iterator()
batch_X, batch_Y = iter.get_next()

# Construct a neural network
H1 = tf.layers.dense(batch_X, node_num, kernel_initializer=He_init, name="H1")
H1_out = tf.nn.leaky_relu(H1, alpha=0.1)

H2 = tf.layers.dense(H1_out, node_num, kernel_initializer=He_init, name="H2")
H2_out = tf.nn.leaky_relu(H1, alpha=0.1)

H3 = tf.layers.dense(H2_out, node_num, kernel_initializer=He_init, name="H3")
H3_out = tf.nn.leaky_relu(H1, alpha=0.1)

H4 = tf.layers.dense(H3_out, node_num, kernel_initializer=He_init, name="H4")
H4_out = tf.nn.leaky_relu(H1, alpha=0.1)

H5 = tf.layers.dense(H4_out, node_num, kernel_initializer=He_init, name="H5")
H5_out = tf.nn.leaky_relu(H1, alpha=0.1)

Y_hat = tf.layers.dense(H5_out, output_num, name="outputs")

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

# Initialize saver
saver = tf.train.Saver()

# Initialize plotting coordinates
epoch_axis  = []
epoch_loss_axis   = []
epoch_error_axis  = []
valid_loss_axis   = []
valid_error_axis  = []

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
            sess.run(optimizer)


        # Compute current loss and error
        sess.run(iter.initializer, feed_dict={X:X_train1, Y:y_train1,
                    batch_size:X_train1.shape[0]})

        epoch_loss, epoch_acc = sess.run([loss, accuracy])
        epoch_error = (1.0 - epoch_acc) * 100

        print("Epoch =", epoch, "loss =", "{:.9f}".format(epoch_loss),
                "error =", "{:.2f}".format(epoch_error), "per cent")

        # Calculate validation loss and error for this epoch
        sess.run(iter.initializer, feed_dict={X:X_valid1, Y:y_valid1,
                    batch_size:X_valid1.shape[0]})

        valid_loss, valid_acc = sess.run([loss, accuracy])
        valid_error = (1.0 - valid_acc) * 100

        # Update the coordinates
        epoch_axis.append(epoch)
        epoch_loss_axis.append(epoch_loss)
        epoch_error_axis.append(epoch_error)
        valid_loss_axis.append(valid_loss)
        valid_error_axis.append(valid_error)

        if valid_loss < best_loss :
            save_path = saver.save(sess, path)
            best_loss = valid_loss
            best_error  = valid_error
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
    test_acc = sess.run(accuracy)
    precision = test_acc * 100
    print("Test precision is {:.2f} per cent".format(precision))

# Show the final validation results
print("Final validation loss =", "{:.9f}".format(best_loss))
print("Final validation error =", "{:.2f}".format(best_error), "per cent")

# Plot and show the graphs
plt.plot(epoch_axis, epoch_loss_axis, label='Training Loss')
plt.plot(epoch_axis, valid_loss_axis, label='Validation Loss')
plt.legend()
plt.show()

plt.plot(epoch_axis, epoch_error_axis, label='Training Error in percentage')
plt.plot(epoch_axis, valid_error_axis, label='Validation Error in percentage')
plt.legend()
plt.show()
