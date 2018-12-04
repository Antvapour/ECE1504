import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Loading data
with np.load("notMNIST.npz") as data :
    Data, Target = data["images"], data["labels"]
    posClass = 2
    negClass = 9
    dataIndx = (Target==posClass) + (Target==negClass)
    Data = Data[dataIndx]/255
    Target = Target[dataIndx].reshape(-1,1)
    Target[Target==posClass] = 1
    Target[Target==negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget   = Data[3600:], Target[3600:]

    trainData = np.asarray(trainData)
    trainData = np.reshape(trainData, (trainData.shape[0], trainData.shape[1]*trainData.shape[2]))

    validData = np.asarray(validData)
    validData = np.reshape(validData, (validData.shape[0], validData.shape[1]*validData.shape[2]))

# Initialize parameters
learning_rate = 0.001
batch_size = 500
decay_coef = 0.01
iteration = 5000
display_step = 10
train_size = 3500

# Initialize placeholders
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Initialize w and b
W = tf.Variable(tf.random_uniform([trainData.shape[1], 1]), name="W")
b = tf.Variable(tf.random_uniform([1,1]), name="b")

# Compute the loss
Y_hat = tf.add(tf.matmul(X, W), b)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_hat)
cross_entropy_loss = tf.reduce_mean(cross_entropy)
weight_decay_loss = decay_coef * tf.nn.l2_loss(W)
loss = cross_entropy_loss + weight_decay_loss

# Store the initializers for different optimize models
optimizers = []
optimizers.append(tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, name='GradientDescentOptimizer'))
optimizers.append(tf.train.AdamOptimizer(learning_rate).minimize(loss, name='AdamOptimizer'))

# Iterate for different optimize models
for optimizer in optimizers :

    # Initialize
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess :

        # Run the initializer
        sess.run(init)

        # Initialize counters for epochs
        epoch_counter = 0
        batch_index   = 0

        # Initialize plotting coordinates
        epoch_num  = []
        loss_val   = []

        for i in range(iteration) :
            # Update the training epoch if last epoch is finished
            if batch_index == train_size :
                # Update the counters
                epoch_counter += 1
                batch_index    = 0

                # Shuffle the training set
                randIndx = np.arange(len(trainData))
                np.random.shuffle(randIndx)
                trainData, trainTarget = trainData[randIndx], trainTarget[randIndx]

                # Display logs per 10 epochs
                if epoch_counter % display_step ==0 :
                    # Compute current loss
                    curr_loss = sess.run(loss, feed_dict={X:trainData, Y:trainTarget})
                    predict_soft = sess.run(Y_hat, feed_dict={X:validData})
                    predict = (predict_soft >= 0.5).astype(int)
                    correct = (predict == validTarget).astype(int)
                    accuracy = np.sum(correct)
                    print("Epoch =", epoch_counter, "loss =", "{:.9f}".format(curr_loss))

                    # Update the coordinates
                    epoch_num.append(epoch_counter)
                    loss_val.append(curr_loss)

            # Update the training batch
            if batch_index + batch_size > train_size :
                batch_X, batch_Y = trainData[batch_index:], trainTarget[batch_index:]
                batch_index = train_size
            else :
                batch_X, batch_Y = trainData[batch_index:batch_index + batch_size], \
                    trainTarget[batch_index:batch_index + batch_size]
                batch_index += batch_size

            # Run the optimizer
            sess.run(optimizer, feed_dict={X:batch_X, Y:batch_Y})

        print("Optimization Finished!")
        # compute final loss
        final_loss = sess.run(loss, feed_dict={X:trainData, Y:trainTarget})
        print("Epoch =", epoch_counter+1, "Final loss =", final_loss)

    # Plot the graph
    plt.plot(epoch_num, loss_val, label=optimizer.name)

# Show the graph
plt.legend()
plt.show()
