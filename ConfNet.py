#! -*- coding=utf-8 -*-
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt

# Trianing settings
parser = argparse.ArgumentParser(description='Pytorch ConfNet Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default:64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs',type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model',action='store_true', default=True,
                    help='For Saving the current Model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {}


class ConfNet(nn.Module):
    def __init__(self):
        super(ConfNet, self).__init__()
        self.fc1 = nn.Linear(7, 121)
        self.fc2 = nn.Linear(121, 242)
        self.fc3 = nn.Linear(242, 121)
        self.fc4 = nn.Linear(121, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def data_loader():
    data = list()
    target = list()
    with open('/tmp/pycharm_project_551/result_allfactor_2pt_shuf.txt') as f:
        for line in f:
            line = line.strip()
            subline = line.split(' ')
            if float(subline[5]) < 0.3:
                tmp = [float(subline[5]),float(subline[6])/25,float(subline[7])/50,float(subline[8])*np.pi/180/5,float(subline[9])/10,float(subline[10])/10,float(subline[11])/20]
            # data.append(float(subline[5:]))
                data.append(tmp)
                target.append(float(subline[2]))
    return data, target

# Init model
# model = ConfNet()
# set optimizer
# optimizer = optim.Adam(model.parameters(), lr=args.lr)


xxx = []
def train(epoch):
    model.train()
    data, target = data_loader()


    for i in range(0,len(data)/100-1):
        batch_idx = i
        data_, target_ = data[batch_idx*100 : batch_idx*100 + 100], target[batch_idx*100 : batch_idx*100 + 100]
        optimizer.zero_grad()

        data_array = np.array(data_)
        target_array = np.array(target_)

        torch_data = torch.from_numpy(data_array)
        torch_target = torch.from_numpy(target_array)

        torch_data = torch.autograd.Variable(torch_data.float())
        torch_target = torch.autograd.Variable(torch_target.float())

        output = model(torch_data)
        # print(output)
        loss = F.mse_loss(output, torch_target)
        # loss = F.l1_loss(output, torch_target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print("Train Epoch: {:.0f} [{:.0f}/{:.0f} ({:.0f}%)]".format(
        #         epoch, batch_idx*100, 1540.0, 100. * batch_idx / 1540
        #     ))
        print("train loss: {}".format(loss.data))
        tmp = loss.data.numpy()
        xxx.append(tmp)



def test():
    model.eval()
    test_loss = 0
    # with torch.no_grad():
    data, target = data_loader()
    for i in range(len(data)/100-1, len(data)/100):
        batch_idx = i
        data_, target_ = data[batch_idx*100 : batch_idx*100 + 100], target[batch_idx*100 : batch_idx*100 + 100]

        data_array = np.array(data_)
        target_array = np.array(target_)

        torch_data = torch.from_numpy(data_array)
        torch_target = torch.from_numpy(target_array)

        torch_data = torch.autograd.Variable(torch_data.float())
        torch_target = torch.autograd.Variable(torch_target.float())


        output = model(torch_data)
        test_loss += F.mse_loss(output, torch_target)



    test_loss /= 100.0
    # print("test_loss: {}".format(test_loss))


if __name__ == "__main__":
    # for epoch in range(1, args.epochs + 1):
    #     train(epoch)
    #     test()
    # if (args.save_model):
    #     torch.save(model.state_dict(), "ConfNet.pt")

    model = ConfNet()
    model.load_state_dict(torch.load('ConfNet.pt'))
    model.eval()

    data, target = data_loader()

    eval = []
    for i in range(0,len(data)):
        test_data = data[i]
        data_array = np.array(test_data)
        torch_data = torch.from_numpy(data_array)
        torch_data = torch.autograd.Variable(torch_data.float())
        out = model(torch_data)
        print(".............{}".format(out))
        # print("predict: {}".format(out - target[i]))
        tmp = out - target[i]

        tmp = tmp.data.numpy()
        eval.append(tmp)
    print(np.std(eval),np.mean(eval))

    plt.plot(eval)
    plt.show()

    # plt.plot(xxx)
    # plt.show()

