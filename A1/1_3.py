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
learning_rate = 0.005
batch_size = 500
decay_coefs = [0, 0.001, 0.1, 1]
iteration = 20000
display_step = 1000
train_size = 3500

# Initialize placeholders
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Initialize w and b
W = tf.Variable(tf.random_uniform([trainData.shape[1], 1]), name="W")
b = tf.Variable(tf.random_uniform([1,1]), name="b")

# Iterate for different decay coefficients
for decay_coef in decay_coefs :
    # Compute the loss
    Y_hat = tf.add(tf.matmul(X, W), b)
    MSE_loss = tf.losses.mean_squared_error(Y_hat, Y)
    weight_decay_loss = decay_coef * tf.nn.l2_loss(W)
    loss = MSE_loss + weight_decay_loss

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Initialize
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess :

        # Run the initializer
        sess.run(init)

        # Initialize counters for epochs
        epoch_counter = 0
        batch_index   = 0

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

                # Display logs per 1000 epochs
                if epoch_counter % display_step ==0 :
                    # Compute current loss
                    curr_loss = sess.run(loss, feed_dict={X:trainData, Y:trainTarget})
                    print("Epoch =", epoch_counter, "loss =", "{:.9f}".format(curr_loss))

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

        # compute accuracy for validation and test data
        predict_soft = sess.run(Y_hat, feed_dict={X:validData})
        predict = (predict_soft >= 0.5).astype(int)
        correct = (predict == validTarget).astype(int)
        accuracy = np.sum(correct) # / 100 * 100

        test_predict_soft = sess.run(Y_hat, feed_dict={X:testData})
        test_predict = (test_predict_soft >= 0.5).astype(int)
        test_correct = (test_predict == testTarget).astype(int)
        test_accuracy = np.sum(test_correct) / 145 * 100

    # Show the final loss, accuracy of the validation set and test set
    print("Epoch =", epoch_counter+1, "Final loss =", final_loss)
    print("validation accuracy =", accuracy, "per cent")
    print("test_accuracy = {:.2f} per cent".format(test_accuracy))
