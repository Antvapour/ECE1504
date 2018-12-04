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
decay_coef = 0
iteration = 5000
display_step = 100
train_size = 3500

# Initialize placeholders
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Initialize w and b
W = tf.Variable(tf.random_uniform([trainData.shape[1], 1]), name="W")
b = tf.Variable(tf.random_uniform([1,1]), name="b")

# Compute the losses
Y_hat = tf.add(tf.matmul(X, W), b)
MSE_loss = tf.losses.mean_squared_error(Y_hat, Y)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_hat)
cross_entropy_loss = tf.reduce_mean(cross_entropy)
weight_decay_loss = decay_coef * tf.nn.l2_loss(W)
loss_Linear = MSE_loss + weight_decay_loss
loss_Logistic = cross_entropy_loss + weight_decay_loss

losses = {}
losses["Linear"] = loss_Linear
losses["Logistic"] = loss_Logistic

plot_data = {}

# Iterate for different losses
for name, loss in losses.items() :
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, name='AdamOptimizer')

    # Initialize
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess :

        # Run the initializer
        sess.run(init)

        # Initialize plotting coordinates
        epoch_num = []
        loss_val  = []
        accur_val  = []

        for i in range(iteration) :
            # Run the optimizer
            sess.run(optimizer, feed_dict={X:trainData, Y:trainTarget})

            # Display logs per 100 epochs
            if i % display_step == display_step-1 :
                # Compute current loss and accuracy
                curr_loss = sess.run(loss, feed_dict={X:trainData, Y:trainTarget})
                predict_soft = sess.run(Y_hat, feed_dict={X:validData})
                predict = (predict_soft >= 0.5).astype(int)
                correct = (predict == validTarget).astype(int)
                accuracy = np.sum(correct)
                print("Epoch =", i+1, "loss =", "{:.9f}".format(curr_loss))

                # Update the coordinates
                epoch_num.append(i+1)
                loss_val.append(curr_loss)
                accur_val.append(accuracy)

        print("Optimization Finished!")
        # Compute loss of the validation set
        valid_loss = sess.run(loss, feed_dict={X:validData, Y:validTarget})
    print("Valid loss =", valid_loss)

    # Store the data for plotting
    plot_data[name+"_epoch"] = epoch_num
    plot_data[name+"_loss"] = loss_val
    plot_data[name+"_accuracy"] = accur_val

# Plot and show the graphs
for name in losses :
    plt.plot(plot_data[name+"_epoch"],
            plot_data[name+"_loss"], label=name+" loss")

plt.legend()
plt.show()

for name in losses :
    plt.plot(plot_data[name+"_epoch"],
            plot_data[name+"_accuracy"], label=name+" accuracy")

plt.legend()
plt.show()
