import numpy as np
import tensorflow as tf

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

    testData  = np.asarray(testData)
    testData  = np.reshape(testData, (testData.shape[0], testData.shape[1]*testData.shape[2]))

# Initialize parameters
learning_rates = [0.03, 0.01, 0.003, 0.001]
decay_coef = 0
iteration = 5000
display_step = 500
train_size = 3500
test_size = 145

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

# Iterate for different learning rates
for learning_rate in learning_rates :
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, name='AdamOptimizer')

    # Initialize
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess :

        # Run the initializer
        sess.run(init)

        for i in range(iteration) :
            # Run the optimizer
            sess.run(optimizer, feed_dict={X:trainData, Y:trainTarget})

            # Display logs per 500 epochs
            if i % display_step == display_step-1 :
                # Compute current loss
                curr_loss = sess.run(loss, feed_dict={X:trainData, Y:trainTarget})
                print("Epoch =", i+1, "loss =", "{:.9f}".format(curr_loss))

        print("Optimization Finished!")
        # Compute final loss
        final_loss = sess.run(loss, feed_dict={X:trainData, Y:trainTarget})

        # Compute train, validation and test accuracy
        train_predict_soft = sess.run(Y_hat, feed_dict={X:trainData})
        train_predict = (train_predict_soft >= 0.5).astype(int)
        train_correct = (train_predict == trainTarget).astype(int)
        train_accuracy = np.sum(train_correct) / train_size * 100

        valid_predict_soft = sess.run(Y_hat, feed_dict={X:validData})
        valid_predict = (valid_predict_soft >= 0.5).astype(int)
        valid_correct = (valid_predict == validTarget).astype(int)
        valid_accuracy = np.sum(valid_correct) # / 100 * 100

        test_predict_soft = sess.run(Y_hat, feed_dict={X:testData})
        test_predict = (test_predict_soft >= 0.5).astype(int)
        test_correct = (test_predict == testTarget).astype(int)
        test_accuracy = np.sum(test_correct) / test_size * 100

    # Show the results
    print("Final loss =", final_loss)
    print("train_accuracy = {:.2f} per cent".format(train_accuracy))
    print("valid_accuracy =", valid_accuracy, "per cent")
    print("test_accuracy = {:.2f} per cent".format(test_accuracy))
