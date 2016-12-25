#!/usr/bin/python
# encoding: utf-8
import heapq
from numpy import *
import numpy as np
from scipy import misc
from PIL import Image, ImageDraw
from time import clock

# def denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
# 	m,n = im.shape 

# 	U = U_init
#   	Px = im
#   	Py = im 
#   	error = 1

# 	while (error > tolerance):
#   		Uold = U

#  		GradUx = roll(U,-1,axis=1)-U 
#   		GradUy = roll(U,-1,axis=0)-U 

#   		PxNew = Px + (tau/tv_weight)*GradUx
#   		PyNew = Py + (tau/tv_weight)*GradUy
#   		NormNew = maximum(1,sqrt(PxNew**2+PyNew**2))

#   		Px = PxNew/NormNew 
#   		Py = PyNew/NormNew 

#   		RxPx = roll(Px,1,axis=1) 
#   		RyPy = roll(Py,1,axis=0) 

#   		DivP = (Px-RxPx)+(Py-RyPy) 
#   		U = im + tv_weight*DivP
#   		error = linalg.norm(U-Uold)/sqrt(n*m);

#   	return U,im-U 

def histeq(im,nbr_bins=256):

	imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
	cdf = imhist.cumsum()
	cdf = 255 * cdf / cdf[-1] 
  	im2 = interp(im.flatten(),bins[:-1],cdf)

  	return im2.reshape(im.shape), cdf


def Rfilt(im,N,w,h):	
	output=im.copy()
	# matT=np.arange(w*h)
	# matT=matT.reshape(h,w)
	# y=matT/w
	# x=matT%w
	# tempxm=np.maximum(x-N,0)
	# tempxM=np.minimum(w,x+N+1)
	# tempym=np.maximum(y-N,0)
	# tempyM=np.minimum(h,y+N+1)
	temp=np.arange((2*N+1)*(2*N+1)*w*h)
	temp=temp.reshape(w*h,(2*N+1)*(2*N+1))
	tempy=(temp%(2*N+1)+temp%(h*(2*N+1)*(2*N+1))/((2*N+1)*(2*N+1))-N)%h
	tempx=(temp/(2*N+1)%(2*N+1)+temp/((2*N+1)*(2*N+1)*h)-N)%w
	tempma=im[tempy,tempx]
	tempma.reshape(w*h,(2*N+1)*(2*N+1))
	#print tempma
	output=np.amax(tempma,axis=1)-np.amin(tempma,axis=1)	
	output=output.reshape(w,h)
	output=np.transpose(output)
	return output

if __name__ == "__main__":
	N=1;
	try:
		
		im0= Image.open("20_1024x600.jpg")
		im=array(im0)
		w,h=im0.size
		im1=array(im0.convert('L'))
		#im1,cdf=histeq(im1)
		#im1,T=denoise(im1,im1)
		start=clock()
		im1=Rfilt(im1,N,w,h)
		finish=clock()
		print finish-start
		# if w<h:
		# 	im=im.tranpose(Image.ROTATE_90)
		# if w>1024&&h>600:
		# 	im=im.crop(w/2-512,h/2-300,w/2+512,h/2+300)
		# else:
		# 	im=
		#new=Rfilt(im,N)

		im[:,:,0]=im1
		im[:,:,1]=0
		im[:,:,2]=0	
		imPIL= Image.fromarray(np.uint8(im))
		imPIL.show()
		#misc.imsave(im,"20_1024x600G.bmp")
	except IOError:
		print('IO error!')
	#print(grey)




			
		

	


