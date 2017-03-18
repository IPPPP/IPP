from captcha.image import ImageCaptcha
import cv2
import numpy as np
from numpy import *
from PIL import Image, ImageDraw
import torch
import random
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms

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

def dataGen(dataSize):
	width=44
	height=60
	image = ImageCaptcha(width, height)
	N=1
	data=[]
	label=[]
	for i in xrange(dataSize):
		num=random.randint(0,10)
		if num==10:
			img = image.generate(' ')
		else: 
			img = image.generate(str(num))
		img = np.fromstring(img.getvalue(), dtype='uint8')
		img= cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
		#img = img.transpose(2, 0, 1)
		img=Rfilt(img,N,width,height)
		#cv2.imwrite("./img/tmp" + str(i % 10) + ".bmp", img)
		img= np.multiply(img, 1/255.0) 
		img = img-np.mean(img)
		data.append([img])
		label.append(num)
	tensor = torch.rand(3).long()
	features=torch.Tensor(np.array(data))
	print features.size()
	label=torch.Tensor(np.array(label)).type(tensor.type())
	train=data_utils.TensorDataset(features,label)
	return train

train=dataGen(16000)
test=dataGen(10000)
torch.save(train, './data/trainingData.pt')
torch.save(test, './data/testData.pt')
