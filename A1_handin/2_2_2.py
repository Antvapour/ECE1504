import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Loading data
def data_segmentation(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255
    data = np.reshape(data, [-1, 32*32])
    target = np.load(target_path)

    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)

    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))

    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
        data[rnd_idx[trBatch+1:trBatch + validBatch],:], \
        data[rnd_idx[trBatch + validBatch+1:-1],:]

    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
        target[rnd_idx[trBatch+1:trBatch + validBatch], task], \
        target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

trainData, validData, testData, trainTarget, validTarget, testTarget = \
    data_segmentation("data_facescrub.npy", "target_facescrub.npy", task = 0)

validData_copy = np.reshape(validData, [-1, 32, 32])
trainTarget = np.transpose(trainTarget)
validTarget = np.transpose(validTarget)
testTarget = np.transpose(testTarget)


# Initialize parameters
N_train = trainData.shape[0]
N_valid = validData.shape[0]
N_test  = testData.shape[0]
learning_rates = [0.01, 0.003, 0.001, 0.0003, 0.0001] # 0.003
batch_size = 300
decay_coefs = [0.1, 0.03, 0.01, 0.003, 0.001]  # 0.003
iteration = 30000
display_step = 100

# Initialize placeholders
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.int32, name="Y")

# Initialize w and b
W = tf.Variable(tf.random_uniform([trainData.shape[1], 6]), name="W")
b = tf.Variable(tf.random_uniform([1, 6]), name="b")

# Iterate for different weight decay coefficients
for decay_coef in decay_coefs :
    # Iterate for different learning rates
    for learning_rate in learning_rates :
        # Compute the loss
        Y_hat = tf.add(tf.matmul(X, W), b)
        # tf.nn.softmax_cross_entropy_with_logits is out of date
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2( \
            labels=tf.one_hot(Y, 6, dtype=tf.float32), logits=Y_hat)
        cross_entropy_loss = tf.reduce_mean(cross_entropy)
        weight_decay_loss = decay_coef * tf.nn.l2_loss(W)
        loss = cross_entropy_loss + weight_decay_loss

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

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
            accur_val  = []

            for i in range(iteration) :
                # Update the training epoch if last epoch is finished
                if batch_index == N_train :
                    # Update the counters
                    epoch_counter += 1
                    batch_index    = 0

                    # Shuffle the training set
                    randIndx = np.arange(len(trainData))
                    np.random.shuffle(randIndx)
                    trainData, trainTarget = trainData[randIndx], trainTarget[randIndx]

                    # Display logs per 500 epochs
                    if epoch_counter % display_step ==0 :
                        # Compute current loss and accuracy
                        curr_loss = sess.run(loss, feed_dict={X:trainData, Y:trainTarget})
                        predict_soft = sess.run(Y_hat, feed_dict={X:validData})
                        predict = np.argmax(predict_soft, axis=1)
                        correct = (predict == validTarget).astype(int)
                        accuracy = np.sum(correct) / N_valid * 100
                        print("Epoch =", epoch_counter, "loss =", "{:.9f}".format(curr_loss))

                        # Update the coordinates
                        epoch_num.append(epoch_counter)
                        loss_val.append(curr_loss)
                        accur_val.append(accuracy)

                # Update the training batch
                if batch_index + batch_size > N_train :
                    batch_X, batch_Y = trainData[batch_index:], trainTarget[batch_index:]
                    batch_index = N_train
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
            valid_predict_soft = sess.run(Y_hat, feed_dict={X:validData})
            valid_predict = np.argmax(valid_predict_soft, axis=1)
            valid_correct = (valid_predict == validTarget).astype(int)
            valid_accuracy = np.sum(valid_correct) / N_valid * 100

            test_predict_soft = sess.run(Y_hat, feed_dict={X:testData})
            test_predict = np.argmax(test_predict_soft, axis=1)
            test_correct = (test_predict == testTarget).astype(int)
            test_accuracy = np.sum(test_correct) / N_test * 100

            # When learning_rate is 0.003 and decay_coef is 0.003,
            # plot the graphs as required.
            # Also display a miss-predicted picture and the prediction.
            if learning_rate == 0.003 and decay_coef == 0.003 :
                # Plot and show the graphs
                plt.plot(epoch_num, loss_val, label='Cross Entropy Loss for lambda='+ \
                    str(decay_coef)+', learning rate='+str(learning_rate))
                plt.legend()
                plt.show()
                plt.plot(epoch_num, accur_val, label='Accuracy for lambda='+ \
                    str(decay_coef)+', learning rate='+str(learning_rate))
                plt.legend()
                plt.show()

                valid_one_fail = np.argmin(valid_correct)
                plt.imshow(validData_copy[valid_one_fail])
                plt.show()
                print("Ground truth is:", validTarget[valid_one_fail])
                print("Miss-predicted as:", valid_predict[valid_one_fail])

        # Show the results
        print("Final loss =", final_loss)
        print("Validation accuracy = {:.2f} per cent".format(valid_accuracy))
        print("Test accuracy = {:.2f} per cent".format(test_accuracy))
