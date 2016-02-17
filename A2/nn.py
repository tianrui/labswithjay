"""
nn.py
train neural networks for assignment 2
"""
import os
import numpy as np
import tensorflow as tf
import pdb


"""
Utility for initializing parameters of the model as variables
"""
def W_var(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def b_var(shape, name=None):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def conv2d(x, W, strides=[1,1,1,1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def max_pool(x, ksize):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')


"""
Define an MLP graph
"""
def mlp(input_dim, output_dim, drop_rate, hiddens=[], nonlinearity='relu'):
    assert nonlinearity == 'relu'
    assert len(hiddens) > 0

    g = tf.Graph()
    with g.as_default():
        tf_input = tf.placeholder(tf.float32, shape=[None, input_dim])
        tf_output = tf.placeholder(tf.float32, shape=[None, output_dim])
        weights = []
        biases = []

        previous_dim = input_dim
        previous_out = tf_input
        layer_i = 0
        for hidden in hiddens:
            layer_i += 1
            weight = W_var([previous_dim, hidden], name='W_%d'%layer_i)
            b = b_var([previous_dim, hidden], name='b_%d'%layer_i)
            weights.append(weight)
            biases.append(b)
            
            output = tf.nn.relu(tf.matmul(previous_out, weight) + b)
            previous_dim = hidden
            previous_out = output
        output = tf.matmul(previous_out, W_var([previous_dim, output_dim])) + b_var([previous_dim, output_dim])
    
    return g

def convo():
    return

def convnet(imgwidth, imgheight, output_dim, configs):
    model = mlp(28*28, 10, 0, hiddens=[128]) 

    # Do it the old fashioned way
    assert configs['filter_width']
    assert configs['stride']

    #Wc1 = W_var([5, 5, 1, imgwidth])

def task6():
    seed = os.time()
    np.random.seed(seed)

    

    dataset = dataset_gen("notMNIST.npz", flatten=True, onehot=True)
    batchsize = 128
    patience = 40
    imgheight = 28
    imgwidth = 28
    channels = 1
    layers = np.random.randint(1, 4)
    hidden_dim = np.random.randint(100, 501)
    out_dim = 10
    drop_rate = np.random.choice([0., 0.5])
    out_dir = "./"

    learning_rate = np.exp(-1. * np.random.randint(-4, -1))
    iters = 1001
    train_log = np.zeros(iters)
    valid_log = np.zeros(iters)
    train_error = np.zeros(iters)
    valid_error = np.zeros(iters)

    _g = tf.Graph()
    with _g.as_default():
        tf_train = tf.placeholder(tf.float32, shape=(batchsize, imgheight * imgwidth * channels))
        tf_label = tf.placeholder(tf.float32, shape=(batchsize, out_dim))
        tf_valid = tf.constant(dataset['x_valid'])
        tf_valid_label = tf.constant(dataset['y_valid'], dtype='float32')
        tf_test = tf.constant(dataset['x_test'])
        
        W_h = []
        b_h = []
        upper_dim = imgheight*imgwidth*channels
        lower_dim = hidden_dim
        for layer_i in arange(layers):
            W_h.append(tf.Variable(tf.truncated_normal([upper_dim, lower_dim], stddev=0.1)))
            b_h.append(tf.Variable(tf.constant(1.0, shape=[lower_dim])))
            upper_dim = lower_dim

        W_out = tf.Variable(tf.truncated_normal([lower_dim, out_dim], stddev=0.1))
        b_out = tf.Variable(tf.constant(1.0, shape=[out_dim]))

        def model_drop(data):
            hidden = []
            input = data
            for layer_i in arange(layers):
                hidden[layer_i] = tf.nn.relu(tf.nn.dropout(tf.matmul(input, W_h1), drop_rate) + b_h1)
            return tf.matmul(h1, W_out) + b_out
        def model_nodrop(data):
            h1 = tf.nn.relu(0.5 * tf.matmul(data, W_h1) + b_h1) # reduce activity in forward prop
            return tf.matmul(h1, W_out) + b_out

        logits = model_drop(tf_train)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_label))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        yhat_train = tf.nn.softmax(logits)
        yhat_valid = tf.nn.softmax(model_nodrop(tf_valid))
        validation_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yhat_valid, tf_valid_label))
        yhat_test = tf.nn.softmax(model_nodrop(tf_test))

    with tf.Session(graph=task2_g) as session:

        # Checkpoints
        check_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        check_prefix = os.path.join(check_dir, "model")
        # Create dir
        if not os.path.exists(check_dir):
            os.makedirs(check_dir)
        saver = tf.train.Saver(tf.all_variables())

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
            if (step > iters/4 and vl > valid_log[np.nonzero(valid_log)].min()):
                stop_counter += 1
            else:
                # if we improve, reset counter
                stop_counter = 0
            if (stop_counter > patience):
                print ('Early stopping in step %d with counter at %d' % (step, stop_counter))
                print ('Minibatch loss at step %d: %f, validation loss %f' % (step, l, vl))
                print 'Minibatch accuracy: ', accuracy(predictions, y_batch)
                print 'Validation accuracy: ', accuracy(yhat_valid.eval(), dataset['y_valid'])
                print 'Validation errors: ', errors(yhat_test.eval(), dataset['y_test'])
                print 'Test accuracy: ', accuracy(yhat_test.eval(), dataset['y_test'])
                print 'Test errors: ', errors(yhat_test.eval(), dataset['y_test'])
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
                        
        np.savez('t5log_likehood.npz' % (-np.log(learning_rate)), 
                    learning_rate = learning_rate,
                    batch_size = batchsize,
                    batch_train_log = train_log,
                    valid_log = valid_log,
                    train_error = train_error,
                    valid_error = valid_error)

def task3():
    graph = mlp(28*28, 10, 0.5, [1000, 100])

if __name__ == '__main__':
    #convnet(28, 28, 10, {'filter_width':5, 'stride':1})
    #task2()
    #task6()
