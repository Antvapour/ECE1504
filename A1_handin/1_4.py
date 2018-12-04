import numpy as np
import tensorflow as tf
import time

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
learning_rate = 0.005
decay_coef = 0
iteration = 20000
display_step = 1000

# Initialize placeholders
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Initialize w and b
W = tf.Variable(tf.random_uniform([trainData.shape[1], 1]), name="W")
b = tf.Variable(tf.random_uniform([1,1]), name="b")

# Compute the loss
Y_hat = tf.add(tf.matmul(X, W), b)
MSE_loss = tf.losses.mean_squared_error(Y_hat, Y)
weight_decay_loss = decay_coef * tf.nn.l2_loss(W)
loss = MSE_loss + weight_decay_loss

# Start the timer before the Initialization of the optimizer
start_time = time.time()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Initialize
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess :

    # Run the initializer
    sess.run(init)

    for i in range(iteration) :
        # Run the optimizer
        sess.run(optimizer, feed_dict={X:trainData, Y:trainTarget})

        # Display logs per 1000 epochs
        if i % display_step == display_step-1 :
            # Compute current loss
            curr_loss = sess.run(loss, feed_dict={X:trainData, Y:trainTarget})
            print("Epoch =", i+1, "loss =", "{:.9f}".format(curr_loss))

    print("Optimization Finished!")
    # compute final loss and time spent
    final_loss = sess.run(loss, feed_dict={X:trainData, Y:trainTarget})
    print("Final loss =", final_loss)
    end_time = time.time()

    # compute accuracy for validation data
    predict_soft = sess.run(Y_hat, feed_dict={X:validData})
    predict = (predict_soft >= 0.5).astype(int)
    correct = (predict == validTarget).astype(int)
    accuracy = np.sum(correct) # / 100 * 100

# Show the results
print("validation accuracy =", accuracy, "per cent")
print("Time spent =", end_time-start_time)
