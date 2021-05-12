import numpy as np
import torch
import argparse
from torch import nn
from helper import one_hot_encode, get_batches
from model import Network


def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001):
    """
    This function is resposible for training the lstm model.
    Args:
        net: the lstm model
        data: the input data that will be used in the model
        epochs: number of iterations
        batch_size: number of batches given
        seq_length: n examples in training set default is 50
        lr: learning rate
    Returns:
        None
    """
    clip = 5
    val_frac = 0.1
    print_every = 10
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    val_idx = int(len(data) * (1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
        print("using GPU {}".format(torch.cuda.get_device_name()))
        net = net.cuda()
    else:
        print("using CPU")
        net = net.cpu()
    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        # initializing hidden units
        h = net.init_hidden(batch_size)
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            x = one_hot_encode(x, n_chars)
            inputs = torch.from_numpy(x)
            targets = torch.from_numpy(y)
            # if gpu is there use it.
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.long().cuda()
            h = tuple([each.data for each in h])
            net.zero_grad()
            output, h = net(inputs, h)
            loss = criterion(output, targets.view(batch_size*seq_length))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            if counter % print_every == 0:
                val_h = net.init_hidden(batch_size)
                val_losses = []
                for x, y in get_batches(val_data, batch_size, seq_length):
                    x = one_hot_encode(x, n_chars)
                    x = torch.from_numpy(x)
                    y = torch.from_numpy(y)
                    val_h = tuple([each.data for each in val_h])
                    inputs, targets = x, y
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        targets = targets.long().cuda()
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output,
                                         targets.view(batch_size*seq_length))
                    val_losses.append(val_loss.item())
                    print('epoch: {}/{}'.format(e+1, epochs),
                          'steps: {}'.format(counter),
                          'Loss {:.4f}'.format(loss.item()),
                          'val_loss {:.4f}'.format(np.mean(val_losses)))
    return None


with open('./data/x_files_dataset.txt', 'r', encoding="latin-1") as f:
    text = f.read()


# We need turn our data into numerical tokens
# Neural networks can only learn from numerical data
chars = tuple(set(text))
# obtaining all of the unique characters being used in the text
chars = tuple(set(text))

# coverting the chars into a dictionary with the index
# being the key, and the unique chars being the value
int2char = dict(enumerate(chars))
# Creating a dictionary where the we map the unique characters as key
# the values being the digits
char2int = {ch: i for i, ch in int2char.items()}

# Looping through the text, and pulling the interger values
# char2int dictionary then coverting to array

encoded = np.array([char2int[ch] for ch in text])


def main(encoded):
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', type=int, help='batch size', default=64)
    parser.add_argument('-e', type=int, help=' number of epochs', default=10)
    parser.add_argument('-lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('-sl', type=int, help="sequence length", default=100)
    parser.add_argument('-nh', type=int, help="number of hidden", default=128)
    parser.add_argument('-nl', type=int, help="number of layers", default=2)
    parser.add_argument('-dropout', type=float, help="dropout", default=0.5)
    args = parser.parse_args()
    net = Network(chars,
                  n_hidden=args.nh,
                  n_layers=args.nl,
                  drop_prob=0.5,
                  lr=args.lr)
    train(net,
          encoded,
          epochs=args.e,
          batch_size=args.bs,
          seq_length=args.sl)
    model_name = "rnn_{}_epochs.net".format(args.e)
    checkpoint = {"n_hidden": net.n_hidden,
                  "n_layers": net.n_layers,
                  "state_dict": net.state_dict(),
                  "tokens": net.chars}
    with open("weights/"+model_name, mode="wb") as f:
        torch.save(checkpoint, f)


if __name__ == '__main__':
    main(encoded)
