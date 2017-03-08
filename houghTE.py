#!/usr/bin/python
# encoding: utf-8
import cv2
import numpy as np
from numpy import *
from PIL import Image, ImageDraw
from time import clock

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

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

houghPa=170
img = cv2.imread('36x_1024x600.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray,T=denoise(gray,gray)
gray=array(gray,dtype=uint8)
gray = cv2.GaussianBlur(gray, (5,5), 15, 5) 
# edges=cv2.Laplacian(gray,cv2.CV_16S,ksize=3)
# edges=array(edges,dtype=uint8)
edges = cv2.Canny(gray,10,100)
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
imgt=img.copy()
imgt[:,:,0]=img[:,:,2]
imgt[:,:,1]=img[:,:,1]
imgt[:,:,2]=img[:,:,0]
img0= Image.fromarray(np.uint8(imgt))
coeffs = find_coeffs(
        [(0, 0), (1024, 0), (1024, 600), (0, 600)],
        [(x2,y2), (x3,y3), (x4,y4), (x1,y1)])
img0.transform((1024,600), Image.PERSPECTIVE, coeffs,
        Image.BICUBIC).save("test1111.bmp")
