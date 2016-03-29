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
    
    return t

def mog_classify(x, norm_dist):
    M, D = x.shape
    M, K = norm_dist.shape
    ind = np.argmin(norm_dist, 1)
    sorted_x = x[ind.argsort()]
    t = []
    for k in range(K):
        t.append(x[np.where(ind==k)[0]])
    
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
            print 'cluster x, y shape ', t[i][:, 0].shape, t[i][:, 1].shape
            
            s = plt.scatter(t[i][:, 0], t[i][:, 1], color=next(colors))
            #print "returned ", s
        plt.show() 
        plt.savefig('t12_3_scatter_k%d.png' % (i))

# Calculating likelihood of a given cluster
def std_z(x, mu, sigma):
    """
    x: M x D
    mu: K x D
    sigma: K
    """
    x_norm = tf.reduce_sum(x*x, 1, keep_dims=True)
    mu_norm = tf.transpose(tf.reduce_sum(mu*mu, 1, keep_dims=True))
    dist = x_norm + mu_norm - 2.*tf.matmul(x, tf.transpose(mu))
    res = tf.reduce_sum(dist, 0) / (-2. * sigma*sigma)
    return res

def mog_dist(x, mu, sigma):
    """
    x: M x D
    mu: K x D
    sigma: K
    """
    dist = utils.L2_dist(x, mu)
    res = dist / (2. * sigma*sigma)
    return res

def mog_log_likelihood_z(x, mu, sigma):
    xshape = tf.cast(tf.shape(x), tf.float32)
    norm_dist = utils.L2_dist(x, mu)
    #norm_likelihood = norm_dist / tf.reduce_sum(norm_dist, reduction_indices=1, keepdims=True)
    log_like = tf.log(tf.pow(1/(2*np.pi), xshape[1]/2.)) - tf.log(sigma) + -0.5 * norm_dist * norm_dist / sigma
    return log_like, norm_dist*norm_dist/sigma

def mog_logprob(log_likelihood_z):
    return log_likelihood_z - utils.reduce_logsumexp(log_likelihood_z, keep_dims=True)

def t2(lr=0.005, K=3):
    data = utils.load_data('data2D.npy').astype("float32")
    M, D = data.shape

    graph = tf.Graph()
    with graph.as_default():
        x_train = tf.placeholder(tf.float32, shape=(None, D))
        mu  = tf.Variable(tf.truncated_normal([K, D], dtype=tf.float32))
        # Assume isotropic variance
        sigma  = tf.Variable(tf.truncated_normal([K], dtype=tf.float32))
        phi  = tf.Variable(tf.truncated_normal([K], dtype=tf.float32))

        likelihood = std_z(x_train, mu, sigma)
        log_like_z, z = mog_log_likelihood_z(x_train, mu, sigma)
        #logprob_kn = log_like_z + (tf.log(utils.logsoftmax(phi)))
        logProb = mog_logprob(log_like_z)

        norm_dist = mog_dist(x_train, mu, sigma)
        cost = utils.reduce_logsumexp(likelihood, 0)
        #cost = tf.reduce_sum(utils.reduce_logsumexp(log_like_z + tf.tile(tf.log(utils.logsoftmax(phi)), [M, 1]), 1), 0)
        optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(cost)

    epochs = 100

    with tf.Session(graph=graph) as sess:

        tf.initialize_all_variables().run()
        cost_l = []
 
        for epoch in range(epochs):

            x_batch = data
            feed_dict={x_train:x_batch}
            
            _, c, like, log_pz, logp, zval = sess.run([optim, cost, likelihood, log_like_z, logProb, z], feed_dict=feed_dict)
            cost_l.append(c)
            ind = np.argmin(like)
            val = np.min(like)
            if epoch % 10 == 0:
                #print log_pz.shape, logp
                #print log_pz[:10]
                print zval
                print("Epoch %03d, cost = %.2f. %02d cluster has lowest likelihood %.2f" % (epoch, c, ind, val))
                #print("Log prob %.2f" % (logp))
        feed_dict = {x_train:x_batch}
        _, c, normdist, like, mu = sess.run([optim, cost, norm_dist, likelihood, mu], feed_dict=feed_dict)
        ind = np.argmin(like)
        val = np.min(like)
        print("Final result cost = %.2f. %02d cluster has lowest likelihood %.2f" % (c, ind, val))
# Plotting scatter
        t = mog_classify(data, normdist)
        colors = iter(cm.rainbow(np.linspace(0, 1, len(t))))
        plt.clf()
        for i in range(len(t)):
            print 'plotting scatter...'
            print 'cluster x, y shape ', t[i][:, 0].shape, t[i][:, 1].shape
            color_i=next(colors)
            plt.scatter(t[i][:, 0], t[i][:, 1], color=color_i)
            plt.scatter(mu[i][0], mu[i][1], marker='x', color=color_i)
            #print "returned ", s
        plt.show() 
        plt.savefig('t22_3_scatter_k%d.png' % (i))


        print like

    return cost_l, mu

def t2_validation(lr=0.005, K=3):
    data = utils.load_data('data2D.npy').astype("float32")
    M, D = data.shape

    graph = tf.Graph()
    with graph.as_default():
        x_train = tf.placeholder(tf.float32, shape=(None, D))
        mu  = tf.Variable(tf.truncated_normal([K, D], dtype=tf.float32))
        # Assume isotropic variance
        sigma  = tf.Variable(tf.truncated_normal([K], dtype=tf.float32))

        likelihood = std_z(x_train, mu, sigma)
        log_like_z, z = mog_log_likelihood_z(x_train, mu, sigma)
        logProb = mog_logprob(log_like_z)

        norm_dist = mog_dist(x_train, mu, sigma)
        cost = utils.reduce_logsumexp(likelihood, 0)
        optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(cost)

    epochs = 100

    with tf.Session(graph=graph) as sess:

        tf.initialize_all_variables().run()
        cost_l = []
 
        for epoch in range(epochs):

            x_batch = data[:2*M/3]
            feed_dict={x_train:x_batch}
            
            _, c, like, log_pz, logp, zval = sess.run([optim, cost, likelihood, log_like_z, logProb, z], feed_dict=feed_dict)
            cost_l.append(c)
            ind = np.argmin(like)
            val = np.min(like)
            if epoch % 10 == 0:
                #print log_pz.shape, logp
                #print log_pz[:10]
                print zval
                print("Epoch %03d, cost = %.2f. %02d cluster has lowest likelihood %.2f" % (epoch, c, ind, val))
                #print("Log prob %.2f" % (logp))


        feed_dict = {x_train:data[2*M/3:]}
        _, c, normdist, like, mu = sess.run([optim, cost, norm_dist, likelihood, mu], feed_dict=feed_dict)
        ind = np.argmin(like)
        val = np.min(like)
        print("Validation result cost = %.2f. %02d cluster has lowest likelihood %.2f" % (c, ind, val))
# Plotting scatter
        t = mog_classify(data, normdist)
        colors = iter(cm.rainbow(np.linspace(0, 1, len(t))))
        plt.clf()
        for i in range(len(t)):
            print 'plotting scatter...'
            print 'cluster x, y shape ', t[i][:, 0].shape, t[i][:, 1].shape
            color_i=next(colors)
            plt.scatter(t[i][:, 0], t[i][:, 1], color=color_i)
            plt.scatter(mu[i][0], mu[i][1], marker='x', color=color_i)
            #print "returned ", s
        plt.show() 
        plt.savefig('t22_3_scatter_k%d_with_validation.png' % (i))


        print like

    return cost_l, mu


if __name__=='__main__':
    #t1_3()
    t2()

