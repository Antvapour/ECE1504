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
path = "./model0.ckpt.meta"

# Import graph
saver = tf.train.import_meta_graph(path)

#print(tf.get_default_graph().get_operation_by_name("Adam/learning_rate"))
filename = "log.txt"
f = open(filename,'w')

for op in tf.get_default_graph().get_operations() :
    print(op.name, file=f)
    #time.sleep(1)
