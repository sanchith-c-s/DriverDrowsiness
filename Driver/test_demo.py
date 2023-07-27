import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time
import random


video=cv2.VideoCapture(0)

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_default.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

eye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_eye_tree_eyeglasses.xml')


name='C:/Users/Srinath/Eye/value/'
lbl=['Close','Open']

model = load_model('./transfer_model/cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
level=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret,frame=video.read()
    height,width = frame.shape[:2] 
    
    
    
    #faces=facedetect.detectMultiScale(frame,1.1,4)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #HISTOGRAM BY CHANGING ALPHA BETA IN SAME CONVERSION METHOD
    a=cv2.convertScaleAbs(gray,alpha=3.10,beta=5)
    
    #HISTOGRAM FUNCTION TO DISPLAY ALL PIXEL DETAILS OF FRAME WITH PLOT
    #TO CHECK UNCOMMENT THE BELOW FUNCTION CALL
    
    #histogram(frame)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 3 )
    
    
   
       
    eyes = eye_cascade.detectMultiScale(gray)
    for (ex, ey ,ew, eh) in eyes:
        cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
   
    
    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        #cv2.rectangle(frame, (x,y),(x+w,y+h), (0, 255, 0), 2)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break
    

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        #cv2.rectangle(frame, (x,y),(x+w,y+h), (0, 255, 0), 2)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break
       
    if(rpred[0]==0 and lpred[0]==0):
        
        level=level+1
        
        r=random.randint(0,1000)
        k=random.randint(10,80)
        s=str(r)+str(k)
        h=str(level)
        t='image'+s+h+'.jpg'
        t=str(t)
        cv2.imwrite(os.path.join(name,t),frame)
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)
        #cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,0,255) , 3 )
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        level=level-3
        cv2.putText(frame,"Open",(10,height-20), font, 1,(0,255,0),1,cv2.LINE_AA)
        #cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 3 )
     
    if(level<0):
        level=0
        sound.stop()
       
    cv2.putText(frame,'Level:'+str(level),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    if(level>20):
        #person is feeling sleepy so we beep the alarm
        
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    
    cv2.imshow('frame',frame)
    #cv2.imshow('Gray',gray)
    #cv2.imshow('Histogram',a)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
