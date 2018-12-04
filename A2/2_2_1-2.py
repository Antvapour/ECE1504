import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

He_init = tf.variance_scaling_initializer()

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# Load data sets
(X_train, y_train), (X_test, y_test)=tf.keras.datasets.mnist.load_data()
X_train=X_train.astype(np.float32).reshape(-1,28*28)/255.0
X_test=X_test.astype(np.float32).reshape(-1,28*28)/255.0
y_train=y_train.astype(np.int32)
y_test=y_test.astype(np.int32)
X_valid, X_train=X_train[:5000], X_train[5000:]
y_valid, y_train=y_train[:5000], y_train[5000:]
X_train2_full=X_train[y_train>=5]
y_train2_full=y_train[y_train>=5]-5
X_valid2_full=X_valid[y_valid>=5]
y_valid2_full=y_valid[y_valid>=5]-5
X_test2=X_test[y_test>=5]
y_test2=y_test[y_test>=5]-5

# Initialize parameters
path = "./model0.ckpt"
new_path = "./transfer0.ckpt"
epoch_num = 1000
train_batch_size = 20
train_batch_num = int(epoch_num / train_batch_size)
max_check = 20

# Import graph
saver = tf.train.import_meta_graph(path+".meta")

# Initialize new saver
new_saver = tf.train.Saver()

# Restore placeholders
X = tf.get_default_graph().get_tensor_by_name("X:0")
Y = tf.get_default_graph().get_tensor_by_name("Y:0")
batch_size = tf.get_default_graph().get_tensor_by_name("batch_size:0")
learning_rate = tf.get_default_graph().get_tensor_by_name("Adam/learning_rate:0")
loss = tf.get_default_graph().get_tensor_by_name("loss:0")
accuracy = tf.get_default_graph().get_tensor_by_name("accuracy:0")
graph = tf.get_default_graph()

# Restore the dataset initialization operation
dataset_init_op = graph.get_operation_by_name('dataset_init')

# Freeze lower layers
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="Y_hat")
    optimizer = optimizer.minimize(loss, var_list=train_vars)

# Initialize plotting coordinates
epoch_axis  = []
epoch_loss_axis   = []
epoch_error_axis  = []
valid_loss_axis   = []
valid_error_axis  = []

check_count = 0
best_loss = np.infty
best_error  = 0

# Start training
with tf.Session() as sess :
    # Restore the session
    saver.restore(sess, path)

    for epoch in range(epoch_num) :
        # train for one epoch
        sess.run(dataset_init_op, feed_dict={X:X_train2_full, Y:y_train2_full,
                    batch_size:train_batch_size})

        for batch in range(train_batch_num) :
            sess.run(optimizer)


        # Compute current loss and error
        sess.run(dataset_init_op, feed_dict={X:X_train2_full, Y:y_train2_full,
                    batch_size:X_train2_full.shape[0]})

        epoch_loss, epoch_acc = sess.run([loss, accuracy])
        epoch_error = (1.0 - epoch_acc) * 100

        print("Epoch =", epoch, "loss =", "{:.9f}".format(epoch_loss),
                "error =", "{:.2f}".format(epoch_error), "per cent")

        # Calculate validation loss and error for this epoch
        sess.run(dataset_init_op, feed_dict={X:X_valid2_full, Y:y_valid2_full,
                    batch_size:X_valid2_full.shape[0]})

        valid_loss, valid_acc = sess.run([loss, accuracy])
        valid_error = (1.0 - valid_acc) * 100

        # Update the coordinates
        epoch_axis.append(epoch)
        epoch_loss_axis.append(epoch_loss)
        epoch_error_axis.append(epoch_error)
        valid_loss_axis.append(valid_loss)
        valid_error_axis.append(valid_error)

        if valid_loss < best_loss :
            save_path = new_saver.save(sess, new_path)
            best_loss = valid_loss
            best_error  = valid_error
            check_count = 0
        else :
            check_count += 1
            if check_count > max_check :
                print("Early stopping!")
                break

    # Calculate test precision using the best model
    new_saver.restore(sess, new_path)
    sess.run(dataset_init_op, feed_dict={X:X_test2, Y:y_test2,
                batch_size:X_test2.shape[0]})
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

################################################################################
# Reduce the number of samples per digits
def sample_n_instances_per_class(X, y, n) :
    Xs, ys = [], []
    for label in np.unique(y):
        idx = (y == label)
        Xc = X[idx][:n]
        yc = y[idx][:n]
        Xs.append(Xc)
        ys.append(yc)
    return np.concatenate(Xs), np.concatenate(ys)

# Initialize plotting coordinates
epoch_axis  = []
epoch_loss_axis   = []
epoch_error_axis  = []
valid_loss_axis   = []
valid_error_axis  = []

check_count = 0
best_loss = np.infty
best_error  = 0
new_path = "./transfer0_1.ckpt"

# Start training
with tf.Session() as sess :
    # Restore the session
    saver.restore(sess, path)

    for epoch in range(epoch_num) :
        # train for one epoch
        feed_X, feed_y = sample_n_instances_per_class(X_train2_full,
                    y_train2_full, 100)
        sess.run(dataset_init_op, feed_dict={X:feed_X, Y:feed_y,
                    batch_size:train_batch_size})

        for batch in range(train_batch_num) :
            sess.run(optimizer)


        # Compute current loss and error
        feed_X, feed_y = sample_n_instances_per_class(X_train2_full,
                    y_train2_full, 100)
        sess.run(dataset_init_op, feed_dict={X:feed_X, Y:feed_y,
                    batch_size:X_train2_full.shape[0]})

        epoch_loss, epoch_acc = sess.run([loss, accuracy])
        epoch_error = (1.0 - epoch_acc) * 100

        print("Epoch =", epoch, "loss =", "{:.9f}".format(epoch_loss),
                "error =", "{:.2f}".format(epoch_error), "per cent")

        # Calculate validation loss and error for this epoch
        feed_X, feed_y = sample_n_instances_per_class(X_valid2_full,
                    y_valid2_full, 100)
        sess.run(dataset_init_op, feed_dict={X:feed_X, Y:feed_y,
                    batch_size:X_valid2_full.shape[0]})

        valid_loss, valid_acc = sess.run([loss, accuracy])
        valid_error = (1.0 - valid_acc) * 100

        # Update the coordinates
        epoch_axis.append(epoch)
        epoch_loss_axis.append(epoch_loss)
        epoch_error_axis.append(epoch_error)
        valid_loss_axis.append(valid_loss)
        valid_error_axis.append(valid_error)

        if valid_loss < best_loss :
            save_path = new_saver.save(sess, new_path)
            best_loss = valid_loss
            best_error  = valid_error
            check_count = 0
        else :
            check_count += 1
            if check_count > max_check :
                print("Early stopping!")
                break

    # Calculate test precision using the best model
    new_saver.restore(sess, new_path)
    feed_X, feed_y = sample_n_instances_per_class(X_test2,
                y_test2, 100)
    sess.run(dataset_init_op, feed_dict={X:feed_X, Y:feed_y,
                batch_size:X_test2.shape[0]})
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
