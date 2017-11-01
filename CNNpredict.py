import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from numpy import *
import torch.nn.init as init
import cv2
import sys
N=1

def Rfilt(im,N,w,h):    
        temp=np.arange((2*N+1)*(2*N+1)*w*h)
        temp=temp.reshape(w*h,(2*N+1)*(2*N+1))
        tempy=np.mod(np.mod(temp,(2*N+1))+np.mod(temp,(h*(2*N+1)*(2*N+1)))/((2*N+1)*(2*N+1)-N),h)
        tempx=np.mod(np.mod(temp/(2*N+1),(2*N+1))+temp/((2*N+1)*(2*N+1)*h)-N,w)
        tempma=im[tempy,tempx]
        tempma.reshape(w*h,(2*N+1)*(2*N+1))
        output=np.amax(tempma,axis=1)-np.amin(tempma,axis=1)    
        output=output.reshape(w,h)
        output=np.transpose(output)
        return output
        
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
        self.fc1 = nn.Linear(3072,2048)
        self.BN5 =nn.BatchNorm1d(2048) 
        self.fc2 = nn.Linear(2048,11)   
        # nn.init.kaiming_uniform(self.conv1.weight)
        # nn.init.kaiming_uniform(self.conv2.weight)
        # nn.init.kaiming_uniform(self.conv3.weight)
        # nn.init.kaiming_uniform(self.conv4.weight) 
        # nn.init.kaiming_uniform(self.fc1.weight) 
        # nn.init.kaiming_uniform(self.fc2.weight)
        

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.BN1(self.conv1(x)), 2,stride=2))
        x = F.relu(F.max_pool2d(self.BN2(self.conv2(x)), 2,stride=2))
        x = F.relu(F.max_pool2d(self.BN3(self.conv3(x)), 2,stride=2))
        x = F.relu(F.max_pool2d(self.BN4(self.conv4(x)), 2,stride=2))
        x = x.view(-1, 3072) 
        x = F.relu(self.BN5(self.fc1(x))) 
        #x = F.dropout(x, training=self.training) 
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

model = Net()

imgPath="PATH"
imgPath=sys.argv[1]
#model= torch.load('model.pt')
model.load_state_dict(torch.load('modelSD.pt'))
model.eval()
img= cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

img=Rfilt(img,N,44,60)
#cv2.imwrite("T" + str(1 % 10) + ".bmp", img)
img= np.multiply(img, 1/255.0) 
img = img-np.mean(img)
img=img.reshape(1,1,60,44)
#img.dtype='double'
print img
#print array(img).shape
outputs =model(Variable(torch.from_numpy(img).float(),volatile=True))
print outputs.data
predicted =outputs.data.max(1)[1]
print (predicted)
