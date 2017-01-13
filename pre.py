#!/usr/bin/python
# encoding: utf-8
from numpy import *
import numpy as np
from PIL import Image, ImageDraw
from time import clock

def histeq(im,nbr_bins=256):
	imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
	cdf = imhist.cumsum()
	cdf = 255 * cdf / cdf[-1] 
  	im2 = interp(im.flatten(),bins[:-1],cdf)
  	return im2.reshape(im.shape), cdf


def Rfilt(im,N,w,h):	
	temp=np.arange((2*N+1)*(2*N+1)*w*h)
	temp=temp.reshape(w*h,(2*N+1)*(2*N+1))
	tempy=(temp%(2*N+1)+temp%(h*(2*N+1)*(2*N+1))/((2*N+1)*(2*N+1))-N)%h
	tempx=(temp/(2*N+1)%(2*N+1)+temp/((2*N+1)*(2*N+1)*h)-N)%w
	tempma=im[tempy,tempx]
	tempma.reshape(w*h,(2*N+1)*(2*N+1))
	output=np.amax(tempma,axis=1)-np.amin(tempma,axis=1)	
	output=output.reshape(w,h)
	output=np.transpose(output)
	return output

def preprocess(im0):
	N=2
	im0=im0.resize((1024,600))
	# if w<h:
	# 	im=im.tranpose(Image.ROTATE_90)
	# if w>1024&&h>600:
	# 	im=im.crop(w/2-512,h/2-300,w/2+512,h/2+300)
	# else:
	# 	im=
	w,h=im0.size
	im1=0.30*array(im0)[:,:,0]+0.59*array(im0)[:,:,1]+0.11*array(im0)[:,:,2]
	#im1,cdf=histeq(im1)
	start=clock()
	im1=Rfilt(im1,N,w,h)
	finish=clock()
	print finish-start
	im=array(im0)
	im[:,:,0]=im1
	im[:,:,1]=0
	im[:,:,2]=0	
	return im

if __name__ == "__main__":
	try:
		for i in [55] :
			i=i+1
			im0=Image.open("%d_1024x600.jpg"%i)
			im=preprocess(im0)
			imPIL= Image.fromarray(np.uint8(im))
			imPIL.save("%d_1024x600_nt.bmp"%i,"bmp")
	except IOError:
		print('IO error!')





			
		

	


