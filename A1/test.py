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

    X= np.hstack((trainData, np.ones((3500,1))))
    X=X.astype('float32')
    validData= np.hstack((validData, np.ones((100,1))))
# Initialize parameters
decay_coefs = 0


X= np.hstack((trainData, np.ones((3500,1))))
X_Trans=tf.tofloat(tf.transpose(X))
XX = tf.matmul(X_Trans, X.astype)
XX_inv = tf.matrix_inverse(XX)
product = tf.matmul(XX_inv, tf.transpose(X))
w = tf.matmul(product,trainTarget)

# Initialize placeholders
data = tf.placeholder(tf.float32, name="data")
Y = tf.placeholder(tf.float32, name="Y")
# Compute the loss
Y_hat = tf.matmul(data, w)
MSE_loss = 0.5*tf.losses.mean_squared_error(Y_hat,Y )
weight_decay_loss = decay_coefs * tf.nn.l2_loss(w)
loss = MSE_loss + weight_decay_loss

   
with tf.Session() as sess:
     # Start the timer before the Initialization of the optimizer
     start_time = time.time()
     final_loss = sess.run(loss, feed_dict={X:trainData, Y:trainTarget})
     print("Final loss =", final_loss)
     end_time = time.time()
 

    # compute accuracy for validation data
     predict_soft = sess.run(Y_hat, feed_dict={X:validData, Y:validTarget})
     predict = (predict_soft >= 0.5).astype(int)
     correct = (predict == validTarget).astype(int)
     accuracy = np.sum(correct) # / 100 * 100

# Show the results
print("validation accuracy =", accuracy, "per cent")
print("Time spent =", end_time-start_time)





