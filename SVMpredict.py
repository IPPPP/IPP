#!/usr/bin/python
# encoding: utf-8
from numpy import *
import numpy as np
from sklearn import svm
from PIL import Image, ImageDraw
from time import clock
from sklearn.externals import joblib
import pre

def segP(im00,sw,sh,w,h):
	nx=w/sw
	ny=h/sh
	k=0
	feature=np.arange(nx*ny*sw*sh,dtype=float)
	feature=feature.reshape(nx*ny,sw*sh)
	for j in xrange(ny):
		for i in xrange(nx):
			feature[k]=im00[j*sh:j*sh+sh,i*sw:i*sw+sw,0].reshape(sw*sh)			
			k=k+1
	return feature

def obtainBound(label,nx,ny,valve):
	lin=array(map(sum,label))
	lin1=np.roll(lin,-1)
	linS=lin+lin1
	linIndex= linS>=(2*valve-nx*2)
	k=0
	index=ones(1)	
	for i in xrange(ny):
		if (k==0 and linIndex[i]):
			upper=i
			lower=i+1
			k=1
			continue
		if k==1:
			if linIndex[i]:
				lower=i+1
			else:
				index=np.hstack((index,upper,lower))
				k=0
	index=delete(array(index,dtype=int),0)
	return index

if __name__ == "__main__":
	imNo=14
	sw=21
	sh=30
	valve1=22
	valve2=1
	try:
		classifier=joblib.load('model02.pkl')
		im0=Image.open("t%d_1024x600.jpg"%imNo)
		im=pre.preprocess(im0)
		w,h=im0.size
		nx=w/sw
		ny=h/sh
		feature=segP(im,sw,sh,w,h)/255
		#print feature
		start=clock()
		label=classifier.predict(feature)
		finish=clock()
		print finish-start	
		label=label.reshape(ny,nx)
		print label

		index=obtainBound(label,nx,ny,valve1)
		##abs(linS-np.roll(linS,-1))
		print index
		isize=index.size/2
		index=index.reshape(isize,2)
		indexNx=zeros(isize*2,dtype=int).reshape(isize,2)
		outputIm=zeros(h*w*3*isize).reshape(h,w,3,isize)
		for i in xrange(isize):
			tempL=label[index[i,0]:(index[i,1]+1)].T
			col=array(map(sum,tempL))
			colIndex=col>=(2*valve2-(-index[i,0]+(index[i,1]+1)))
			k=0
			for j in xrange(nx):
				if (k==0 and colIndex[j]):
					indexNx[i,0]=j
					indexNx[i,1]=j
					k=1
					continue
				if k==1:
					if colIndex[j]:
						indexNx[i,1]=j

			xrangeN=-indexNx[i,0]*sw+indexNx[i,1]*sw+sw
			yrangeN=-index[i,0]*sh+index[i,1]*sh+sh
			tempMat=np.arange(xrangeN*yrangeN,dtype=int)
			x=indexNx[i,0]*sw+tempMat%xrangeN
			yN=index[i,0]*sh+tempMat/xrangeN
			outputIm[yN,x,:,i]=array(im0)[yN,x,:]
			imPIL= Image.fromarray(np.uint8(outputIm[:,:,:,i]))
			imPIL.show()
	except IOError:
		print('IO error!')





			
		

	


