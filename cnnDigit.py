
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt#
import numpy as np
import torch.nn.init as init
# Training settings
parser = argparse.ArgumentParser(description='Digital Recognition CNN for IPP')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='enable verbose display')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 16, 'pin_memory': True} if args.cuda else {}


train=torch.load('./data/trainingData.pt')
test=torch.load('./data/testData.pt')

train_loader = torch.utils.data.DataLoader(train,
    batch_size=args.batch_size, shuffle=True,**kwargs )#**kwargs
test_loader = torch.utils.data.DataLoader(test,
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,64 , kernel_size=5,padding=2)
        self.BN1 =nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5,padding=2)
        self.BN2 =nn.BatchNorm2d(128)       
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3,padding=1)
        self.BN3 =nn.BatchNorm2d(256)    
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3,padding=1)
        self.BN4 =nn.BatchNorm2d(512)   
        #self.conv3_drop = nn.Dropout2d()  
        # self.conv4 = nn.Conv2d(32, 32, kernel_size=3)
        # self.BN4 =nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(3072,2048)
        self.BN5 =nn.BatchNorm1d(2048) 
        self.fc2 = nn.Linear(2048,11)   
        nn.init.kaiming_uniform(self.conv1.weight)
        nn.init.kaiming_uniform(self.conv2.weight)
        nn.init.kaiming_uniform(self.conv3.weight)
        nn.init.kaiming_uniform(self.conv4.weight) 
        nn.init.kaiming_uniform(self.fc1.weight) 
        nn.init.kaiming_uniform(self.fc2.weight)
        

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.BN1(self.conv1(x)), 2,stride=2))
        x = F.relu(F.max_pool2d(self.BN2(self.conv2(x)), 2,stride=2))
        x = F.relu(F.max_pool2d(self.BN3(self.conv3(x)), 2,stride=2))
        x = F.relu(F.max_pool2d(self.BN4(self.conv4(x)), 2,stride=2))
        # x = F.relu(F.avg_pool2d(self.BN4(self.conv4(x)), 2,stride=1))
        x = x.view(-1, 3072)
        # x = F.relu(self.BN4(self.fc1(x))) 
        # x = F.dropout(x, training=self.training)  
        x = F.relu(self.BN5(self.fc1(x))) 
        x = F.dropout(x, training=self.training) 
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
    #     self.BN1 =nn.BatchNorm2d(16)
    #     self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
    #     self.BN2 =nn.BatchNorm2d(32)
    #     self.fc1 = nn.Linear(3072, 768)
    #     self.BN3 =nn.BatchNorm1d(768)
    #     self.fc2 = nn.Linear(768, 128)
    #     self.BN4 =nn.BatchNorm1d(128)
    #     self.fc3 = nn.Linear(128, 11)
    #     nn.init.kaiming_uniform(self.conv1.weight)
    #     nn.init.constant(self.conv1.bias, 0)
    #     nn.init.kaiming_uniform(self.conv2.weight)
    #     nn.init.constant(self.conv2.bias, 0)
    #     nn.init.kaiming_uniform(self.fc1.weight)
    #     nn.init.constant(self.fc1.bias, 0)
    #     nn.init.kaiming_uniform(self.fc2.weight)
    #     nn.init.constant(self.fc2.bias, 0)
    #     nn.init.kaiming_uniform(self.fc3.weight)
    #     nn.init.constant(self.fc3.bias, 0)

    # def forward(self, x):
    #     x=self.conv1(x)
    #     x=self.BN1(x)
    #     x = F.relu(F.max_pool2d(x, 2))
    #     x=self.conv2(x)
    #     x=self.BN2(x)       
    #     x = F.relu(F.max_pool2d(x, 2))
    #     x = x.view(-1, 3072)
    #     x=self.fc1(x)
    #     x=self.BN3(x) 
    #     x = F.relu(x)
    #     x=self.fc2(x)   
    #     x=self.BN4(x) 
    #     x = F.relu(x)   
    #     x = F.relu(self.fc3(x))
    #     return F.log_softmax(x)


#######Training Part##########
model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                        momentum=args.momentum, weight_decay=0.0005)

def train(epoch):
    model.train()
    train_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss+=loss.data[0]
        if batch_idx % args.log_interval == 0 and args.verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    train_loss /= len(train_loader) 
    return train_loss

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    test_accuracy=100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_accuracy))
    return test_loss, test_accuracy

trainCurveY=[]
accCurveY=[]
CurveX=[]
for epoch in range(1, args.epochs + 1):
    train_loss=train(epoch)
    test_loss,test_accuracy=test(epoch)
    CurveX.append(epoch)
    trainCurveY.append(train_loss)
    accCurveY.append(test_accuracy)
    
torch.save(model.state_dict(), 'modelSD.pt')
torch.save(model, 'model.pt')

X=np.array(CurveX)
Y1=np.array(trainCurveY)
Y2=np.array(accCurveY)
plt.plot(X,Y1,'b-',lw=2.5)
plt.axis([1,args.epochs,0,3])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('training curve')
plt.grid(True)
plt.show()
plt.plot(X,Y2,'r-',lw=2.5)
plt.axis([1,args.epochs,90,100])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('test accuracy')
plt.grid(True)
plt.show()
filters= model.conv1.weight.data.numpy()
filters=np.transpose(filters,(0,2,3,1))
plotLocation=321
fig=plt.figure()
for i in xrange(6):
    ax=fig.add_subplot(plotLocation+i)
    ax.imshow(np.array(filters[i]))
plt.show()

####################################


########Predict Part############3
# model= torch.load('model.pt') 

# def imshow(img):
#     img = img / 2 + 0.5 # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2,0)))
#     plt.show()

# dataiter = iter(test_loader)
# images, labels = dataiter.next()

# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s'%labels[j]for j in range(10)))

# outputs =model(Variable(images))
# _, predicted = torch.max(outputs.data, 1)

# print('Predicted: ', ' '.join('%5s'% predicted[j][0] for j in range(10)))
