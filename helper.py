import numpy as np


def one_hot_encode(arr, n_labels):
    # making an array of zeros size (m, ny)
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


def get_batches(arr, batch_size, seq_length):
    # obtaining the total number of batches we can create
    batch_size_total = batch_size * seq_length
    # number of batches in int that one can create
    # length of array divided by total num  of characters
    n_batches = len(arr)//batch_size_total
    # keep enough characters to make a full batch
    arr = arr[:n_batches * batch_size_total]
    # reshaping it so it's m x nx
    # wanting the number of rows to equal or batchsize
    # -1 is a dimension placeholder
    arr = arr.reshape((batch_size, -1))
    # iterating through the array one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # creating the features
        x = arr[:, n:n+seq_length]
        # creating the targets with x shifted by 1
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
