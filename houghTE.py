#!/usr/bin/python
# encoding: utf-8
import cv2
import numpy as np
from numpy import *

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
    #   im=im.tranpose(Image.ROTATE_90)
    # if w>1024&&h>600:
    #   im=im.crop(w/2-512,h/2-300,w/2+512,h/2+300)
    # else:
    #   im=
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
    return im1


def denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
    m,n = im.shape
    U = U_init
    Px = im 
    Py = im 
    error = 1

    while (error > tolerance):
        Uold = U

        GradUx = roll(U,-1,axis=1)-U 
        GradUy = roll(U,-1,axis=0)-U 


        PxNew = Px + (tau/tv_weight)*GradUx
        PyNew = Py + (tau/tv_weight)*GradUy
        NormNew = maximum(1,sqrt(PxNew**2+PyNew**2))

        Px = PxNew/NormNew
        Py = PyNew/NormNew

        RxPx = roll(Px,1,axis=1)
        RyPy = roll(Py,1,axis=0) 

        DivP = (Px-RxPx)+(Py-RyPy) 
        U = im + tv_weight*DivP 
        error = linalg.norm(U-Uold)/sqrt(n*m);

        return U,im-U 

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = float(Dx) / D
        y = float(Dy) / D
        return x,y
    else:
        return False

houghPa=170

img = cv2.imread('36x_1024x600.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#gray=Rfilt(gray,2,1024,600)
gray,T=denoise(gray,gray)
gray=array(gray,dtype=uint8)


gray = cv2.GaussianBlur(gray, (5,5), 15, 5) 
# edges=cv2.Laplacian(gray,cv2.CV_16S,ksize=3)
# edges=array(edges,dtype=uint8)
edges = cv2.Canny(gray,10,100)

#imPIL= Image.fromarray(np.uint8(edges))
#imPIL.show()

#hough transform
lines = cv2.HoughLines(edges,1,np.pi/190,houghPa)

L_horizontal=[] ## list of almost horizontal lines in form (a,b,c) of ax+by=c
L_vertical=[] ## list of almost vertical lines in form (a,b,c) of ax+by=c

for rho,theta in lines[:,0]: 
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = x0 + 1000*(-b)
    y1 = y0 + 1000*(a)
    x2 = x0 - 1000*(-b)
    y2 = y0 - 1000*(a)

    L_tmp = line([x1,y1],[x2,y2])
    if L_tmp[1]==0:
        L_vertical.append(L_tmp)
    else:
        ratio = -L_tmp[0]/L_tmp[1]
        if ratio > 1 or ratio < -1:
            L_vertical.append(L_tmp)
        else:
            L_horizontal.append(L_tmp)

## find two lines with the longest distance in vertical/horizonal lines
max = -100000
min = 100000
for L in L_horizontal:  ## set x=0, find max and min y
    y = L[2]/L[1]
    if y > max:
        max = y
        L_horizontal_max = L
    if y < min:
        min = y
        L_horizontal_min = L

max = -100000
min = 100000
for L in L_vertical:  ## set y=0, find max and min x
    x = L[2]/L[0]
    if x > max:
        max = x
        L_vertical_max = L
    if x < min:
        min = x
        L_vertical_min = L

x1,y1 = intersection(L_vertical_min, L_horizontal_max)
x2,y2 = intersection(L_vertical_min, L_horizontal_min)
x3,y3 = intersection(L_vertical_max, L_horizontal_min)
x4,y4 = intersection(L_vertical_max, L_horizontal_max)
cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),1)
cv2.line(img,(int(x2),int(y2)),(int(x3),int(y3)),(255,0,0),1)
cv2.line(img,(int(x3),int(y3)),(int(x4),int(y4)),(255,0,0),1)
cv2.line(img,(int(x4),int(y4)),(int(x1),int(y1)),(255,0,0),1)
cv2.imwrite("test1.jpg",img)
