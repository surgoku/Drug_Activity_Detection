"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.1.11
matplotlib
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

torch.manual_seed(1)    # reproducible

# make fake data
def load_data():
    f_train = open("../train.dat", "r")
    f_test = open("../test.dat")
    X = []
    Y = []

    X_test = []

    set_lens = set()
    set_items = set()
    for line in f_train:
        l = line.strip().split('\t')
        x = [int(i) for i in l[1].split()]
        y = int(l[0])
        Y.append(y)
        X.append(x)
        for i in x:
            set_items.add(i)

        set_lens.add(len(x))

    for line in f_test:
        l = line.strip()
        x = [int(i) for i in l.split()]
        X_test.append(x)
        #for i in x:
        #    set_items.add(i)

    max_item = max(set_items)

    items = sorted(list(set_items))
    max_feats = len(items)
    feat_index = {}
    for i,it in enumerate(items):
        feat_index[it] = i

    trans_X = []
    print max_feats
    for x in X:
        #feat = [0]*max_item
        feat = [0] * max_feats
        for i in x:
            #feat[i-1] = 1
            feat[feat_index[i]] = 1
        trans_X.append(feat)

    trans_X_test = []
    for x in X_test:
        #feat = [0]*max_item
        feat = [0] * max_feats
        for i in x:
            #feat[i-1] = 1
            if i in feat_index:
                feat[feat_index[i]] = 1
        trans_X_test.append(feat)


    return trans_X, Y, trans_X_test


X, Y, original_X_test = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
#n_data = torch.ones(100, 2)
#x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
#y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
#x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
#y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)



#x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
#y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# torch can only train on Variable, so convert them to Variable
x = torch.FloatTensor(X)
y = torch.FloatTensor(Y)
x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x

net = Net(n_feature=88119, n_hidden=10, n_output=2)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

plt.ion()   # something about plotting

for t in range(100):
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()