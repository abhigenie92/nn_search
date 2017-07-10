from __future__ import print_function
import argparse
import torch,ipdb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self,hyper_params):
        super(Net, self).__init__()
        self.convs=[]
        self.conv_drop = nn.Dropout2d()
        in_channels=1
        for curr_layer in hyper_params:
            self.convs.append(nn.Conv2d(in_channels, curr_layer['num_filters']
                ,kernel_size=(curr_layer['filter_height'],curr_layer['filter_width'])
                ,stride=(curr_layer['stride_height'],curr_layer['stride_width'])))
            in_channels=curr_layer['num_filters']
        input_shape=(1,28,28)
        n_size = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(n_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        for curr_layer in self.convs:
            x = F.relu(F.max_pool2d(self.conv_drop(curr_layer(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for curr_layer in self.convs:
            x = F.relu(F.max_pool2d(self.conv_drop(curr_layer(x)), 2))
        return x

    def _forward_features(self, x):
        for curr_layer in self.convs:
            x = F.relu(F.max_pool2d(self.conv_drop(curr_layer(x)), 2))
        return x


class Model:
    batch_size=64
    test_batch_size=1000    
    seed=1
    log_interval=10
    no_cuda=True
    cuda = not no_cuda and torch.cuda.is_available()
    def __init__(self, epochs=10,lr=0.01,momentum=0.5,debug=False):
        # Training settings    
        self.epochs=epochs
        self.lr=lr
        self.momentum=momentum
        self.debug=debug
    
        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)
        self.load_data()

    def load_data(self):
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.batch_size, shuffle=True, **kwargs)

    def build_model(self,hyper_params):
        self.model = Net(hyper_params)
        if self.cuda:
            self.model.cuda()


    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % self.log_interval == 0 and self.debug :
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.data[0]))

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        if self.debug:
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))
        return 100. * correct / len(self.test_loader.dataset)

if __name__ == '__main__':
    model=Model(epochs=1)
    model.build_model([{'filter_height':2,'filter_width':2,'num_filters':32,'stride_height':1,
                    'stride_width':1},{'filter_height':2,'filter_width':2,'num_filters':32,'stride_height':1,
                    'stride_width':1}])
    model.train()
    print (model.test())
