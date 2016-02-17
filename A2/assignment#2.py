import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def preprocess():
    # images 28 x 28 x 18720 convert to 18720 x 784
    # labels 18720 x 1 convert to labels1hot 18720 x 10
    with np.load("notMNIST.npz") as data:
        images, labels = data["images"],data["labels"]

    images = images.reshape(28*28,-1)
    images = images.T

    temp = labels.reshape(-1)
    labels1hot = np.zeros((len(temp),10))
    labels1hot[np.arange(len(temp)),temp]=1

    images = images.astype("float32")
    labels = labels.astype("float32")
    labels1hot = labels1hot.astype("float32")

    return images,labels,labels1hot

def LRsoftmax(x_train,x_valid,x_test,y_train,y_valid,y_test,Lrate,Mrate,epoch,batchsize):
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32,shape=[None,784])
    y = tf.placeholder(tf.float32,shape=[None,10])

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    logits = tf.matmul(x,W)+b
    cost_batch = tf.nn.softmax_cross_entropy_with_logits(logits,y)
    cost = tf.reduce_mean(cost_batch)
    train = tf.train.MomentumOptimizer(Lrate,Mrate).minimize(cost)

    y_hat = tf.nn.softmax(logits)
    correction_prediction = tf.equal(tf.argmax(y,1),tf.arg_max(y_hat,1))
    accuracy = tf.reduce_mean(tf.cast(correction_prediction,"float"))

    likelihood = tf.reduce_sum(y*tf.log(1e-20+y_hat))

    init = tf.initialize_all_variables()
    sess.run(init)

    trainerror = np.zeros(epoch/10+1)
    validerror = np.zeros(epoch/10+1)
    trainerrorP = np.zeros(epoch/10+1)
    validerrorP = np.zeros(epoch/10+1)
    trainL = np.zeros(epoch/10+1)
    validL = np.zeros(epoch/10+1)
    testerror = np.zeros(epoch/10+1)
    testerrorP = np.zeros(epoch/10+1)

    for step in xrange(epoch):
        batch = np.random.choice(x_train.shape[0],batchsize)
        sess.run(train, feed_dict={x:x_train[batch,:],y:y_train[batch,:]})
        if step % 10 ==0:
            trainerror[step/10]=15000-np.sum(correction_prediction.eval(feed_dict={x:x_train,y:y_train}))
            validerror[step/10]=1000-np.sum(correction_prediction.eval(feed_dict={x:x_valid,y:y_valid}))
            trainerrorP[step/10]=1-accuracy.eval(feed_dict={x:x_train,y:y_train})
            validerrorP[step/10]=1-accuracy.eval(feed_dict={x:x_valid,y:y_valid})
            trainL[step/10] = likelihood.eval(feed_dict={x:x_train,y:y_train})
            validL[step/10] = likelihood.eval(feed_dict={x:x_valid,y:y_valid})
            testerror[step/10]=2720-np.sum(correction_prediction.eval(feed_dict={x:x_test,y:y_test}))
            testerrorP[step/10]=1-accuracy.eval(feed_dict={x:x_test,y:y_test})

    return trainerror,validerror,trainerrorP,validerrorP,trainL,validL,testerror,testerrorP

def task1():
    images,labels,labels1hot=preprocess()
    x_train = images[0:15000,:]
    x_valid = images[15000:16000,:]
    x_test = images[16000:,:]
    y_train = labels1hot[0:15000,:]
    y_valid = labels1hot[15000:16000,:]
    y_test = labels1hot[16000:,:]

    trainerror,validerror,trainerrorP,validerrorP,trainL,validL,testerror,testerrorP=LRsoftmax(x_train,x_valid,x_test,y_train,y_valid,y_test,1.0,0.999,301,5000)

    epoch = 10.*(range(31)+np.zeros(31))

    plt.figure(1)
    plt.title('Task 1 Classification Error using LR, mini-batch size 5000')

    plt.subplot(311)
    plt.plot(epoch,trainL,label='training log-likelihood')
    plt.plot(epoch,validL,label='validation log-likelihood')
    plt.legend()
    plt.ylabel('log-likelihood')

    plt.subplot(312)
    plt.plot(epoch,trainerror,label='training error count')
    plt.plot(epoch,validerror,label='validation error count')
    plt.legend()
    plt.ylabel('number of errors')

    plt.subplot(313)
    plt.plot(epoch,trainerrorP,label='training error percentage')
    plt.plot(epoch,validerrorP,label='validation error percentage')
    plt.legend()
    plt.xlabel('number of epochs')
    plt.ylabel('errors in percentage')

    test = np.concatenate((epoch.reshape(-1,1),testerror.reshape(-1,1),testerrorP.reshape(-1,1)),axis=1)
    print test

if __name__ == '__main__':
    task1()
    #plt.show()
    plt.savefig('logistic.png', bbox_inches='tight')
