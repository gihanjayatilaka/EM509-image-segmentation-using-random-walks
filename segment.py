'''
    EM509 : Stochastic Processes Project
    Random Walker Algorithm for Image Segmentation
    E/14/158

    gihanjayatilaka[at]eng[dot]pdn[dot]ac[dot]lk
    2020-07-06
'''
import cv2 as cv
import argparse
import numpy as np
import random

FACTOR = 0.5
COLORS= [[255,0,0],[30,105,210],[169,169,169],[0,255,0]]
#           Blue  , Brown       ,Grey        , Green

def down(x):
    return int(x*FACTOR)

def up(x):
    return int(x/FACTOR)

def mouse_callback(event, x, y, flags, params):

    if event==1:
        clicks.append([x,y])
        print(clicks)

def getVal(y,x,ar):
    if x<0 or y <0 or y >= ar.shape[0] or x >=ar.shape[1]:
        return np.array([-1000.0,-1000.0,-1000.0])
    else:
        return ar[y,x,:]

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("-i","--input",dest="input",type=str)
    args.add_argument("-n","--noSegments",dest="noSegments",type=int)
    args=args.parse_args()

    img=cv.imread(args.input)
    noSegments=args.noSegments

    labelledPixelsXY=[]
    noPixels = int(str(input("No pixels being marked?: ")).strip())


    #>>>>>>> Interactive input for initially marked pixels
    for n in range(noSegments):
        print("NOW WE ARE IN SEGMENT",n)
        cv.imshow("image",img)
        clicks=[]
        cv.setMouseCallback('image', mouse_callback)
        while True:
            if len(clicks)==noPixels:
                break
            cv.waitKey(1)
        labelledPixelsXY.append(clicks)
        clicks=[]

    print(labelledPixelsXY)

    #>>>>>>>> Save the initial markings
    imgCopy=np.array(img)
    for n in range(noSegments):
        for i in range(len(labelledPixelsXY[n])):
            print(imgCopy, labelledPixelsXY[n][i], 2,COLORS[n],3)
            cv.circle(imgCopy, (labelledPixelsXY[n][i][0],\
                labelledPixelsXY[n][i][1]), 2,COLORS[n],3)


    #>>>>>>> Resize the image to save computational time
    imgOriginal=np.array(img)
    img=img/255.0
    img=cv.resize(img, (int(img.shape[1]*FACTOR)+1,\
        int(img.shape[0]*FACTOR)+1))



    initiallyMarked=np.zeros((img.shape[0],img.shape[1]),dtype=np.int)
    initiallyMarked.fill(-1)
    segments=np.zeros((img.shape[0],img.shape[1]),dtype=np.int)
    segments.fill(-1)
    cumilativeProbUpRightDownLeft=np.zeros((img.shape[0],\
        img.shape[1],4),dtype=np.float)



    #Generate the transition probabilites based on pixel similarity
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            urdl=[getVal(y-1,x,img),getVal(y,x+1,img),\
                getVal(y+1,x,img),getVal(y,x-1,img)]
            nonNormalizedProbURDL=[]
            for a in range(4):
                tt=np.mean(np.abs(urdl[a]-img[y,x,:]))
                tt=np.exp(-1*np.power(tt,2))
                nonNormalizedProbURDL.append(tt)
            nonNormalizedProbURDL=np.array(nonNormalizedProbURDL)
            normalizedProbURDL = \
                nonNormalizedProbURDL / np.sum(nonNormalizedProbURDL)
            # print(normalizedProbURDL)
            # print(y,x,cumilativeProbUpRightDownLeft.shape)
            cumilativeProbUpRightDownLeft[y,x,0]=normalizedProbURDL[0]
            for a in range(1,4):
                cumilativeProbUpRightDownLeft[y,x,a]=\
                    cumilativeProbUpRightDownLeft[y,x,a-1]+\
                        normalizedProbURDL[a]

    for s in range(noSegments):
        for a in range(len(labelledPixelsXY[s])):
            print(initiallyMarked.shape,\
                down(labelledPixelsXY[s][a][1]),\
                    down(labelledPixelsXY[s][a][0]))
            initiallyMarked[down(labelledPixelsXY[s][a][1]),\
                down(labelledPixelsXY[s][a][0])]=s
            segments[down(labelledPixelsXY[s][a][1]),\
                down(labelledPixelsXY[s][a][0])]=s
    

    
    #Random Walker Algorithm
    for y in range(segments.shape[0]):
        for x in range(segments.shape[1]):
            if segments[y][x]==-1:
                yy=y
                xx=x

                while(initiallyMarked[yy,xx]==-1):
                    rv = random.random()
                    if cumilativeProbUpRightDownLeft[yy,xx,0]>rv:
                        yy-=1
                    elif cumilativeProbUpRightDownLeft[yy,xx,1]>rv:
                        xx+=1
                    elif cumilativeProbUpRightDownLeft[yy,xx,2]>rv:
                        yy+=1
                    else:
                        xx-=1
                segments[y,x]=initiallyMarked[yy,xx]
            print("Finished marking ",y,x)
    

    outputImg=np.array(imgOriginal)
    for y in range(outputImg.shape[0]):
        for x in range(outputImg.shape[1]):        
            outputImg[y,x]=COLORS[segments[down(y),down(x)]]

    cv.imwrite("{}{}".format(args.input[:4],"initial.jpg"),imgCopy)
    cv.imwrite("{}{}".format(args.input[:4],"segments.jpg"),outputImg)
    cv.imwrite("{}{}".format(args.input[:4],"fullProcess.jpg"),\
        np.concatenate((imgCopy,outputImg),axis=1))