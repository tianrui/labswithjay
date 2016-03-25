import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb

import utils
#import AttrDict



def kmeans(data, lr, K, epochs=800):
    """
    example of kmeans algorithm
    """
    M, D = data.shape
    train_data = data[:2*M/3]
    valid_data = data[2*M/3:]

    g = tf.Graph() 
    with g.as_default():
        x = tf.placeholder(tf.float32, shape=(None, D))
        mu = tf.Variable(tf.truncated_normal([K, D], dtype=tf.float32))

        cost = tf.reduce_sum(tf.reduce_min(utils.L2_dist(x, mu), 1))
        optimizer = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(cost)


    with tf.Session(graph=g) as session:
        tf.initialize_all_variables().run()

        l = []
        for epoch in range(epochs):
            x_batch = train_data
            feed_dict = {x: x_batch}
            _, c = session.run([optimizer, cost], feed_dict=feed_dict)
            l.append(c)
            if epoch % 100 == 0:
                print "Epoch %03d, training loss: %.1f" % (epoch, c)
        feed_dict = {x:valid_data}
        c, mu = session.run([cost, mu], feed_dict=feed_dict)
        print "Validation loss: %.1f" % c

    return  {'training_loss': l,
             'validation_loss': c,
             'mu': mu}

def classify(x, mu):
    M, D = x.shape
    K, _ = mu.shape
    dist = utils.L2_dist_np(x, mu)
    ind = np.argmin(dist, 1)
    sorted_x = x[ind.argsort()]
    t = []
    for k in range(K):
        t.append(x[np.where(ind==k)[0]])
        print t[-1].shape
    
    return t
    
                
def t1_2():
    data = utils.load_data('data2D.npy')
    rvals = kmeans(data, 1e-3, 3)
    t_loss = rvals['training_loss']
    v_loss = rvals['validation_loss']
    fig = plt.figure(1, figsize=(16,12))
    plt.plot(np.arange(len(t_loss)), t_loss)
    plt.savefig("t12_2.png")

def t1_3():
    data = utils.load_data('data2D.npy').astype("float32")
    for k in [3]:
        rvals = kmeans(data, 1e-3, k, epochs=1000)
        t_loss = rvals['training_loss']
        v_loss = rvals['validation_loss']
        mu = rvals['mu']
        plt.clf()
        fig = plt.figure(1, figsize=(16,12))
        plt.plot(np.arange(len(t_loss)), t_loss)
        plt.savefig("t12_2_k%d.png" % k)

        t = classify(data, mu)
        colors = iter(cm.rainbow(np.linspace(0, 1, len(t))))
        plt.clf()
        #fig = plt.figure(1, figsize=(16,12))
        for i in range(len(t)):
            print 'plotting scatter...'
            print 'cluster x, y shape ', t[i][:, 0].shape, t[i][:, 1].shape
            
            s = plt.scatter(t[i][:, 0], t[i][:, 1], color=next(colors))
            #print "returned ", s
        plt.show() 
        plt.savefig('t12_3_scatter_k%d.png' % (i))

 



if __name__=='__main__':
    t1_3()

