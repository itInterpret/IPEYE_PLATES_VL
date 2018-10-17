# -*- coding: utf-8 -*-

import os
import sys
import random
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from math import *

DEMOSIZE  = 1
TESTSIZE  = 100
TRAINSIZE = 1000

FILE_PATH ="./"
FILE_PATH_TRAIN = "./train"
FILE_PATH_TEST  = "./test"
FILE_PATH_DEMO  = "./demo"
IMAGE_PATH = "./images/"

IMG_FORMATE = ".jpg"
IMG = Image.open(IMAGE_PATH + "p/10.jpg")
smu = cv2.imread(IMAGE_PATH +"smu2.jpg")
# smu2= cv2.imread(IMAGE_PATH +"smu2.jpg")

NoPlates = IMAGE_PATH + "NoPlates"
noplates_path = []
for parent,parent_folder,filenames in os.walk(NoPlates):
	for filename in filenames:
		# print(filename)
		path = parent+"/"+filename;
	noplates_path.append(path);

def AddSmudginess(img, Smu):
    rows = Smu.shape[0] - 50
    cols = Smu.shape[1] - 50
    adder = Smu[rows:rows + 50, cols:cols + 50];
    adder = cv2.resize(adder, (50, 50));
    #   adder = cv2.bitwise_not(adder)
    img = cv2.resize(img,(50,50))
    img = cv2.bitwise_not(img)
    img = cv2.bitwise_and(adder, img)
    img = cv2.bitwise_not(img)
    return img

def rot(img,angle,shape,max_angle):

    size_o = [shape[1],shape[0]]
    size = (shape[1]+ int(shape[0]*cos((float(max_angle)/180) * 3.14)),shape[0])

    interval = abs( int( sin((float(angle) /180) * 3.14)* shape[0]))
    pts1 = np.float32([[0,0],[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])

    if(angle>0):
        pts2 = np.float32([[interval,0],[0,size[1]],[size[0],0],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]],[size[0]-interval,0],[size[0],size_o[1]]])

    M  = cv2.getPerspectiveTransform(pts1,pts2);
    dst = cv2.warpPerspective(img,M,size);

    return dst;

def rotRandrom(img, factor, size):
    shape = size;
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[factor,factor],[factor,shape[0]-factor],[shape[1]-factor,factor],[shape[1]-factor,shape[0]-factor]])
    M = cv2.getPerspectiveTransform(pts1, pts2);
    dst = cv2.warpPerspective(img, M, size);
    return dst;

def tfactor(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV);

    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2);
    hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7);
    hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8);

    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR);
    return img

def random_envirment(img,data_set):
	index=len(data_set)
	env = cv2.imread(data_set[index])
	env = cv2.resize(env,(img.shape[1],img.shape[0]))
	bak = (img==0);
	bak = bak.astype(np.uint8)*255;
	inv = cv2.bitwise_and(bak,env)
	img = cv2.bitwise_or(inv,img)
	return img

def AddGauss(img, level):
	return cv2.blur(img, (level * 2 + 1, level * 2 + 1));

def r(val):
	return int(np.random.random() * val)

def AddNoiseSingleChannel(single):
	diff = 255-single.max();
	noise = np.random.normal(0,1+r(6),single.shape);
	noise = (noise - noise.min())/(noise.max()-noise.min())
	noise= diff*noise;
	noise= noise.astype(np.uint8)
	dst = single + noise
	return dst

def addNoise(img,sdev = 0.5,avg=10):
	img[:,:,0] =  AddNoiseSingleChannel(img[:,:,0]);
	img[:,:,1] =  AddNoiseSingleChannel(img[:,:,1]);
	img[:,:,2] =  AddNoiseSingleChannel(img[:,:,2]);
	return img;

filepath = os.path.join(FILE_PATH,"test/images/")
if not os.path.exists(filepath):
    os.makedirs(filepath)

for x in range(TESTSIZE):

	A_0 = random.choice (['A', 'B', 'B', 'D', 'E','H', 'K', 'M', 'O', 'P','T', 'X', 'Y'])
	A_1 = random.choice (['A', 'B', 'B', 'D', 'E','H', 'K', 'M', 'O', 'P','T', 'X', 'Y'])
	A_2 = random.choice (['A', 'B', 'B', 'D', 'E','H', 'K', 'M', 'O', 'P','T', 'X', 'Y'])

	N_0 = str(random.randint(0,9))
	N_1 = str(random.randint(0,9))
	N_2 = str(random.randint(0,9))
	N_3 = str(random.randint(0,9))
	N_4 = str(random.randint(0,9))

	s = A_0 + N_0 + N_1 + N_2 + A_1 + A_2 + N_3 + N_4 

	rot_angle = random.randint(-45,45)
	

	IMG_A_0 = Image.open(IMAGE_PATH + "/p/"+ A_0 + ".jpg")

	IMG_N_0 = Image.open(IMAGE_PATH + "/p/"+ N_0 +".jpg")
	IMG_N_1 = Image.open(IMAGE_PATH + "/p/"+ N_1 +".jpg")
	IMG_N_2 = Image.open(IMAGE_PATH + "/p/"+ N_2 +".jpg")

	IMG_A_1 = Image.open(IMAGE_PATH + "/p/"+ A_1 + ".jpg")
	IMG_A_2 = Image.open(IMAGE_PATH + "/p/"+ A_2 + ".jpg")

	IMG_N_3 = Image.open(IMAGE_PATH + "/region/"+ N_3 +".jpg")
	IMG_N_4 = Image.open(IMAGE_PATH + "/region/"+ N_4 +".jpg")


	IMG.paste(IMG_A_0, (90, 72))

	IMG.paste(IMG_N_0 , (227,30))
	IMG.paste(IMG_N_1 , (336,30))
	IMG.paste(IMG_N_2 , (440,30))

	IMG.paste(IMG_A_1, (578, 72))
	IMG.paste(IMG_A_2, (696, 72))

	IMG.paste(IMG_N_3 , (857,24))
	IMG.paste(IMG_N_4 , (949,24))

	filename = filepath + s + IMG_FORMATE
	print (str(x) + " : " + filename)

	IMG.save(filename)

	com = cv2.imread(filename)
	if (x % 2 == 0):
		com = rot(com,15,com.shape,30)
	else:
		com = rot(com,rot_angle,com.shape,180)
	com = rotRandrom(com,10,(com.shape[1],com.shape[0]));
	# com = AddSmudginess(com,smu)
	com = tfactor(com)
	# # com = random_envirment(com,noplates_path);
	com = AddGauss(com, 1+4);
	com = addNoise(com);

	cv2.imwrite(filename, com)

print("Done")

