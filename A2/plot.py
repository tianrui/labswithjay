import numpy as np
import matplotlib.pyplot as plt
import pdb


def gen_plot(fname, title):
    with np.load(fname) as data:
        train_err, valid_err, eta, batch_size, valid_loss, batch_train_loss = data["train_error"], data["valid_error"], data["learning_rate"], data["batch_size"], data["valid_log"], data["batch_train_log"]

    epoch = range(len(train_err))
    fig = plt.figure(1, figsize=(8, 6))
    plt.subplot(211)
    plt.plot(epoch, batch_train_loss, label='training')
    plt.plot(epoch, valid_loss, label='validation')
    plt.legend()
    plt.subplot(211).set_title(title)

    plt.subplot(212)
    plt.plot(epoch, train_err, label='training')
    plt.plot(epoch, valid_err, label='validation')
    plt.legend()
    plt.subplot(212).set_title('Errors, validation size=1000, batch size=%d' % batch_size)

    #pdb.set_trace()
    plt.savefig('plot_%s.png' % fname, bbox_inches='tight')
    fig.clf()
    return

def best_test_results(fname):
    with np.load(fname) as data:
        train_err, valid_err, test_err, eta, batch_size, valid_loss, batch_train_loss = data["train_error"], data["valid_error"], data["test_error"], data["learning_rate"], data["batch_size"], data["valid_log"], data["batch_train_log"]

    best_iter = np.argmin(valid_err[np.nonzero(valid_err)])
    print '%s has best validation loss at %d iterations\n
           validation loss = %d\n
           validation errors = %d\n
           test errors = %d' % (fname, best_iter, valid_loss[best_iter], valid_err[best_iter], test_err[best_iter],

    

def plot_all():
    
    gen_plot('log_likehood11.npz', 'Log likelihood, learning rate 1e-5')
    gen_plot('log_likehood9.npz', 'Log likelihood, learning rate 1e-4')
    gen_plot('log_likehood6.npz', 'Log likelihood, learning rate 1e-3')
    gen_plot('log_likehood4.npz', 'Log likelihood, learning rate 1e-2')
    gen_plot('log_likehood2.npz', 'Log likelihood, learning rate 1e-1')

    gen_plot('t3log_likehood100.npz', 'Log likelihood, 100 hiddens')
    gen_plot('t3log_likehood500.npz', 'Log likelihood, 500 hiddens')
    gen_plot('t3log_likehood1000.npz', 'Log likelihood, 1000 hiddens')

    gen_plot('t4log_likehood9.npz', 'Log likelihood, 2 layer, 500 hiddens each, learning rate 1e-4')

    gen_plot('t5log_likehood.npz', 'Log likelihood, 1000 hiddens, 50% dropout, learning rate 1e-4')
    
    for iter in range(0, 5):
        hypers = np.load('t6hypers%d.npz' % iter)
        layers, keep, eta, hiddens = hypers['layers'], hypers['keep_rate'], hypers['learning_rate'], hypers['hidden_dim'] 
        gen_plot('t6log_likehood%d.npz' % iter, 'Log likelihood, %d layers, %d hiddens each, %d %% dropout, learning rate %f' % (layers, hiddens, 100*(1 - keep), eta))

    

if __name__ == '__main__':
    plot_all()
