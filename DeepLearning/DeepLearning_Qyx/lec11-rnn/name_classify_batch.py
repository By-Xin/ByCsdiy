import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import itertools
import collections
import matplotlib.pyplot as plt

# Read in data
df = pd.read_csv("Chinese_Names_Corpus_Gender（120W）.txt", header=2)
df = df[df.sex != "未知"]
names = df["dict"].values

# Compute character frequency
chars = [list(name) for name in names]
chars_flatten = list(itertools.chain(*chars))
freq = collections.Counter(chars_flatten)
freq = pd.DataFrame(freq.items(), columns=["char", "freq"])
freq = freq.sort_values(by="freq", ascending=False)

# Power law (?)
char_rank = np.arange(freq.shape[0])
char_freq = freq["freq"].values
plt.plot(char_rank, char_freq)
plt.plot(np.log(1.0 + char_rank), np.log(char_freq))

# Prepare data
dict_size = 500
dict = list(freq["char"].values[:dict_size])
dict_set = set(dict)
filtered = list(filter(lambda item: set(item[1]).issubset(dict_set), enumerate(names)))
ind = [idx for idx, name in filtered]
dat = df.iloc[ind]
dat["y"] = np.where(dat["sex"] == "男", 0, 1)

# Split training set and test set
# train = dat.sample(frac=0.8, random_state=123)
# test = dat.drop(train.index)
train = dat.sample(n=10000, random_state=123)
test = dat.sample(n=1000, random_state=321)

# One-hot encoding
def char2index(char):
    return dict.index(char)

def name2index(name):
    return [char2index(char) for char in name]

def name2tensor(name):
    tensor = torch.zeros(len(name), 1, dict_size)
    for i, char in enumerate(name):
        tensor[i, 0, char2index(char)] = 1
    return tensor

def names2tensor(names):
    n = names.shape[0]
    lens = [len(name) for name in names]
    max_len = np.max(lens)
    tensor = torch.zeros(max_len, n, dict_size)
    for j in range(n):
        for i, char in enumerate(names[j]):
            tensor[i, j, char2index(char)] = 1
    return tensor, np.array(lens)

char2index("李")
name2index("李兴")
name2tensor("李兴")
names2tensor(np.array(["李兴", "王胜利"]))



# Build model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, 1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=1)
        hidden = torch.tanh(self.i2h(combined))
        output = torch.sigmoid(self.h2o(hidden))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

# n_hidden = 128
# rnn = RNN(dict_size, n_hidden)
# input = name2tensor("李兴")
# hidden = rnn.init_hidden()
# output, next_hidden = rnn(input[0], hidden)



np.random.seed(123)
torch.random.manual_seed(123)

n = train.shape[0]
n_hidden = 64
nepoch = 5
bs = 100

rnn = RNN(dict_size, n_hidden)
opt = torch.optim.Adam(rnn.parameters(), lr=0.001)
train_ind = np.arange(n)
losses = []

t1 = time.time()
for k in range(nepoch):
    np.random.shuffle(train_ind)
    # Update on mini-batches
    for j in range(0, n, bs):
        # Create mini-batch
        mb = train.iloc[train_ind[j:(j + bs)]]
        mb_size = mb.shape[0]
        names = mb["dict"].values
        input, actual_len = names2tensor(names)
        hidden = rnn.init_hidden(mb_size)
        y = mb["y"].values
        output_rec = []
        for s in range(input.shape[0]):
            output, hidden = rnn(input[s], hidden)
            output_rec.append(output.squeeze())
        output = [output_rec[actual_len[i] - 1][i] for i in range(mb_size)]
        loss = 0.0
        for i in range(mb_size):
            ll = y[i] * torch.log(output[i]) + (1.0 - y[i]) * torch.log(1.0 - output[i])
            loss = loss - ll
        loss = loss / mb_size

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if j // bs % 10 == 0:
            print(f"epoch {k}, batch {j // bs}, loss = {loss.item()}")
t2 = time.time()
print(t2 - t1)

plt.plot(losses)

# Prediction on test set
ntest = test.shape[0]
true_label = test["y"].values
pred = np.zeros(ntest)
rnn.eval()
for i in range(ntest):
    input = name2tensor(test["dict"].values[i])
    hidden = rnn.init_hidden(batch_size=1)
    with torch.no_grad():
        for s in range(input.shape[0]):
            output, hidden = rnn(input[s], hidden)
    pred[i] = output.item()
    if i % 100 == 0:
        print(f"processed {i}")
loss = -np.mean(true_label * np.log(pred) + (1.0 - true_label) * np.log(1.0 - pred))
print(loss)
pred_label = (pred > 0.5).astype(int)
print(np.mean(pred_label == true_label))

# Random cases
np.random.seed(123)
torch.random.manual_seed(123)
ind = np.random.choice(ntest, 10)
ypred = 1 * (pred[ind] > 0.5)
print(test.iloc[ind])
print(test["y"].values[ind])
print(ypred)



names = ["李", "李雪", "李雪峰"]
for name in names:
    input = name2tensor(name)
    hidden = rnn.init_hidden(batch_size=1)
    with torch.no_grad():
        for s in range(input.shape[0]):
            output, hidden = rnn(input[s], hidden)
    pred = output.item()
    print(f"namae: {name}, P(female) = {pred}")
