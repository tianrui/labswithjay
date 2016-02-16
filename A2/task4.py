import numpy as np
import tensorflow as tf
from util import *

if __name__ == '__main__':
    pred = np.random.randn(10, 100)

    dataset = dataset_gen("notMNIST.npz", flatten=True, onehot=True)
    batchsize = 128
    patience = 30
    imgheight = 28
    imgwidth = 28
    channels = 1
    hidden_dim = 500
    out_dim = 10
    learning_rate = 1e-4
    iters = 1001
    
    train_log = np.zeros(iters)
    valid_log = np.zeros(iters)
    train_error = np.zeros(iters)
    valid_error = np.zeros(iters)

    task2_g = tf.Graph()
    with task2_g.as_default():
        tf_train = tf.placeholder(tf.float32, shape=(batchsize, imgheight * imgwidth * channels))
        tf_label = tf.placeholder(tf.float32, shape=(batchsize, out_dim))
        tf_valid = tf.constant(dataset['x_valid'])
        tf_valid_label = tf.constant(dataset['y_valid'], dtype='float32')
        tf_test = tf.constant(dataset['x_test'])

        W_h1 = tf.Variable(tf.truncated_normal([imgheight*imgwidth*channels, hidden_dim], stddev=0.1))
        b_h1 = tf.Variable(tf.constant(1.0, shape=[hidden_dim]))
        W_h2 = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1))
        b_h2 = tf.Variable(tf.constant(1.0, shape=[hidden_dim]))

        W_out = tf.Variable(tf.truncated_normal([hidden_dim, out_dim], stddev=0.1))
        b_out = tf.Variable(tf.constant(1.0, shape=[out_dim]))

        def model(data):
            h1 = tf.nn.relu(tf.matmul(data, W_h1) + b_h1)
            h2 = tf.nn.relu(tf.matmul(h1, W_h2) + b_h2)
            return tf.matmul(h2, W_out) + b_out

        logits = model(tf_train)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_label))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        yhat_train = tf.nn.softmax(logits)
        yhat_valid = tf.nn.softmax(model(tf_valid))
        validation_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yhat_valid, tf_valid_label))
        yhat_test = tf.nn.softmax(model(tf_test))

    with tf.Session(graph=task2_g) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        stop_counter = 0
        for step in range(iters):
            offset = (step * batchsize) % (dataset['y_train'].shape[0] - batchsize)
            x_batch = dataset['x_train'][offset:(offset + batchsize), :]
            y_batch = dataset['y_train'][offset:(offset + batchsize), :]
            feed_dict = {tf_train: x_batch, tf_label: y_batch}
            _, l, vl, predictions = session.run([optimizer, loss, validation_loss, yhat_train], feed_dict=feed_dict)
            """
            if (step > iters/10 and vl > valid_log[np.nonzero(valid_log)].min()):
                    stop_counter += 1
            else:
                # if we improve, reset counter
                stop_counter = 0
            if (stop_counter > patience):
                print ('Early stopping in step %d with counter at %d' % (step, stop_counter))
                raw_input('pres enter')
                break
            """
            train_log[step] = l
            valid_log[step] = vl
            train_error[step] = errors(predictions, y_batch)
            valid_error[step] = errors(yhat_valid.eval(), dataset['y_valid'])
            if (step % 100 == 0):
                print ('Minibatch loss at step %d: %f, validation loss %f' % (step, l, vl))
                print 'Minibatch accuracy: ', accuracy(predictions, y_batch)
                print 'Validation accuracy: ', accuracy(yhat_valid.eval(), dataset['y_valid'])
                print 'Validation errors: ', errors(yhat_test.eval(), dataset['y_test'])
                print 'Test accuracy: ', accuracy(yhat_test.eval(), dataset['y_test'])
                print 'Test errors: ', errors(yhat_test.eval(), dataset['y_test'])

        log_dict = {'learning_rate': learning_rate,
                    'batch_size': batchsize,
                    'batch_train_log': train_log,
                    'valid_log': valid_log,
                    'train_error': train_error,
                    'valid_error': valid_error}
                        
        np.savez('t4log_likehood%d.npz' % (-np.log(learning_rate)), 
                    learning_rate = learning_rate,
                    batch_size = batchsize,
                    batch_train_log = train_log,
                    valid_log = valid_log,
                    train_error = train_error,
                    valid_error = valid_error)
