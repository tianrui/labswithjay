import numpy as np

def accuracy(pred, labels):
    return 100.0 * np.sum(np.argmax(pred, 1) == np.argmax(labels, 1)) / pred.shape[0]

def errors(pred, labels):
    return np.sum(np.argmax(pred, 1) != np.argmax(labels, 1))

def dataset_gen(filename, flatten=False, onehot=False):
    with np.load(filename) as data:
        images, labels = data["images"], data["labels"]
    assert labels.shape[0] > 17000
    #flatten image
    if flatten:
        images = np.reshape(np.rollaxis(images, 2), (images.shape[2], images.shape[0] * images.shape[1]))
        x_train = images[:15000, :]
        x_valid = images[15000:16000, :]
        x_test = images[16000:, :]
    else:
        x_train = images[:, :, :15000]
        x_valid = images[:, :, 15000:16000]
        x_test = images[:, :, 16000:]

    if onehot:
        y_train = np.zeros((15000, 10), dtype=int)
        y_valid = np.zeros((1000, 10), dtype=int)
        y_test = np.zeros((labels.shape[0] - 16000, 10), dtype=int)
        y_train[np.arange(15000), labels[:15000, 0]] = 1
        y_valid[np.arange(1000), labels[15000:16000, 0]] = 1
        y_test[np.arange(labels.shape[0] - 16000), labels[16000:, 0]] = 1

    else:
        y_train = labels[:15000, :]
        y_valid = labels[15000:16000, :]
        y_test = labels[16000:, :]

    return { 'x_train':x_train.astype('float32'),
             'x_valid':x_valid.astype('float32'),
             'x_test':x_test.astype('float32'),
             'y_train':y_train,
             'y_valid':y_valid,
             'y_test':y_test}


