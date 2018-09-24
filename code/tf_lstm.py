from __future__ import print_function
# This file is to implement LSTM in tensorflow
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
import keras
tf.reset_default_graph()
# Training Parameters
learning_rate = 0.001
training_steps = 3000
batch_size = 128
display_step = 100

# Network Parameters
num_input = 1 # this is feature for each timesteps
time_steps = 90  # timesteps (remembering previous 20 days of stock data)
num_hidden = 128 # hidden layer num of features
num_classes = 3 # stock data classes

return_data = pd.read_csv('../data/vol_sp500_price_return') # processed return data
return_data = return_data.iloc[:,1:].as_matrix()
y = return_data[:,91].astype('float32') # this is 1, -1, 0
seq = return_data[:,:91].astype('float32') # this is stock return sequence;

vol_data = pd.read_csv('../data/volume_sp500.csv', sep='\t') # volume data
vol = vol_data.iloc[:,1:]
vol_normed = normalize(vol.as_matrix(), axis=1).astype('float32') # normalized volume data for corresponding return of data; this is covariates in lstm

# Split data into training set and test set
xx_train, xx_test, y_train, y_test = train_test_split(seq, y, stratify=y, test_size=0.2)
x_train = xx_train[:,:90]
x_test = xx_test[:,:90]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

X = tf.placeholder("float", [None, time_steps, num_input])  # store the input data x
Y = tf.placeholder("float", [None, num_classes]) # store the data y

# Define weights of hidden layers
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes])) #variable is ought to be changed during training process
}

biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def lstm(x, weights, biases):

    x = tf.unstack(x, time_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0) #define the LSTM cell in each timestep

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


logits = lstm(X, weights, biases) # generate a RNN cell in this step
prediction = tf.nn.softmax(logits)

#define the loss function and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_op)

# create evaluation graph
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # initialize global variables in this session;

    for step in range(1, training_steps+1):
        batch_x, batch_y = next_batch(128, x_train, y_train)
        batch_x = batch_x.reshape((batch_size, time_steps, num_input))

        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y}) # train the model by next batch data
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y:batch_y})

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print('Training Process finished')
    # evaluate our test data
    x_test = x_test.reshape((x_test.shape[0], time_steps, num_input))
    predicted_y = sess.run(tf.nn.softmax(logits), feed_dict={X: x_test, Y: y_test})
    print("Test accuracy is:", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))

# Survival analysis using deep learning output as features
increase_indices = np.where(y_test[:,1] == 1)[0]
pre_increase_day = np.reshape(xx_test[increase_indices,90], (increase_indices.shape[0],1))
pre_increase_x = np.reshape(predicted_y[increase_indices,:], (-1,3))
event_col = [1]*pre_increase_x.shape[0]
sur_data = np.column_stack((pre_increase_day, pre_increase_x))
sur_data = np.column_stack((sur_data, event_col))
df = pd.DataFrame(data=sur_data)
df = df.drop([1], axis=1)

# predict survival hazard
from lifelines import CoxPHFitter
cph = CoxPHFitter()
cph.fit(df, duration_col=0, event_col=4)
cph.print_summary()  # access the results using cph.summary
cph.plot()

X = df.drop([0, 4], axis=1)
cph.predict_partial_hazard(X)
sur_pred=cph.predict_survival_function(X)



