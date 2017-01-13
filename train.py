#!/usr/bin/python
# encoding: utf-8
from numpy import *
import numpy as np
from sklearn import svm
from PIL import Image, ImageDraw
from time import clock
from sklearn.externals import joblib

def histeq(im,nbr_bins=256):

	imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
	cdf = imhist.cumsum()
	cdf = 255 * cdf / cdf[-1] 
  	im2 = interp(im.flatten(),bins[:-1],cdf)

  	return im2.reshape(im.shape), cdf

def seg(im00,im0,sw,sh,w,h):
	nx=w/sw
	ny=h/sh
	k=0
	feature=np.arange(nx*ny*sw*sh,dtype=float)
	feature=feature.reshape(nx*ny,sw*sh)
	label=np.arange(nx*ny)
	for j in xrange(ny):
		for i in xrange(nx):
			feature[k]=im00[j*sh:j*sh+sh,i*sw:i*sw+sw,0].reshape(sw*sh)			
			if 255 in im0[j*sh:j*sh+sh,i*sw:i*sw+sw,1]:
				label[k]=1
			else:
				label[k]=-1
			#print feature[k],label[k]
			k=k+1

	return feature,label

if __name__ == "__main__":
	sw=21
	sh=30
	featureSum=ones(sw*sh)
	labelSum=ones(1)	
	try:
		for i in xrange(39) :
			i=i+1
			im0=Image.open("%dx_1024x600_n.bmp"%i)
			im00=array(Image.open("%dx_1024x600_nt.bmp"%i))
			w,h=im0.size
			im0=array(im0)
			feature,label=seg(im00,im0,sw,sh,w,h)
			#print feature,label
			featureSum=np.vstack((featureSum,feature))
			labelSum=np.hstack((labelSum,label))

		featureSum=delete(featureSum,0,axis=0)
		labelSum=delete(labelSum,0)
		classifier=svm.SVC(gamma=0.07,cache_size=1000,class_weight='balanced')
		#featureSum,cdf=histeq(featureSum)
		featureSum=featureSum/255
		start=clock()
		classifier.fit(featureSum,labelSum)
		finish=clock()
		print finish-start
		joblib.dump(classifier,'model03.pkl')
	except IOError:
		print('IO error!')



			
		

	


