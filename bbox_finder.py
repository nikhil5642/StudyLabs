import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def overlap(a,b):
    mid_a=[(a[0]+a[2])/2,(a[1]+a[3])/2]
    mid_b=[(b[0]+b[2])/2,(b[1]+b[3])/2]
    if(np.abs(mid_a[0]-mid_b[0])*2<(a[2]+b[2]-a[0]-b[0]+200) and np.abs(mid_a[1]-mid_b[1])*2<(a[3]+b[3]-a[1]-b[1])+20):
        return True
    return False
        
def iou(a,b):
    union=[0,0,0,0]
    union[0]=min(a[0],b[0])
    union[2]=max(a[2],b[2])
    union[1]=min(a[1],b[1])
    union[3]=max(a[3],b[3])
    return union
def combine_box_rec(box,array):
    if(len(array)==0):
        return box,array
    array = np.delete(array, np.flatnonzero((array == box).all(1)), axis=0)
    a=0
    
    for x in array:
        if(overlap(box,x)):
            a=1
            box=iou(box,x)
            array= np.delete(array, np.flatnonzero((array == x).all(1)), axis=0)
            break
    if(a==0):
        return box,array
    x,arr=combine_box_rec(box,array)        
    return x,arr
            
def combine_box(arr):
    bbox=[]
    while True:
        val,arr=combine_box_rec(arr[0],arr)
        bbox.append(val)
        if(len(arr)==0):
            break
    return bbox    
def extract_contours(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) 
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations = 13)
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    contour_list=[]
    last_box=None
    for contour in contours:
        [x,y,w,h]=cv2.boundingRect(contour)
        contour_list.append([x,y,x+w,y+h])
    bbox=combine_box(np.array(contour_list))
    n=1
    for x in bbox:
        cv2.rectangle(img,(x[0],x[1]),(x[2],x[3]),(255,0,255),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,str(n),(x[0],x[1]), font, 4,(0,255,255),2,cv2.LINE_AA)
        n=n+1
    return img,pd.DataFrame(bbox)
