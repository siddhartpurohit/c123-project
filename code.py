#All the modules
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl

X = np.load('image.npz')['arr_0']
y = pd.read_csv("lables.csv")["lables"]

print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

x_train,x_test,y_train,y_test = train_test_split(X,y,random_state = 9,train_size = 3500,test_size = 500)
x_trainscale = x_train/255.0
x_testscale = x_test/255.0

clf = LogisticRegression(solver="saga",multi_class="multinomial").fit(x_trainscale , y_train)
y_pred = clf.predict(x_testscale)
acc = accuracy_score(y_pred,y_test)
print(acc)

cap = cv2.VideoCapture(1)

while (True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape
        upperLeft = (int(width/2-56),int(height/2-56))
        bottomRight = (int(width/2+56),int(height/2 +56))
        cv2.rectangle(gray,upperLeft,bottomRight,(0,255,0),2)
        roi = gray[upperLeft[1]:bottomRight[1],upperLeft[0]:bottomRight[0]]
        imPIL = Image.fromarray(roi)
        imgbW = imPIL.convert("L")
        imgBwResize = imgbW.resize((28,28),Image.ANTIALIAS)
        imgInverted = PIL.ImageOps.invert(imgBwResize)
        pixelFilter = 20
       
        minPixel = np.percentile(imgInverted,pixelFilter)
        
        imgInvertedScaled = np.clip(imgInverted-minPixel,0,255)
        maxPixel = np.max(imgInverted)
        imgInvertedScaled = np.asarray(imgInvertedScaled)/maxPixel
              
        testSample = np.array(imgInvertedScaled).reshape(1,784)
        test_Pred = clf.predict(testSample)
        print("Predicted Class is : ",test_Pred)

        cv2.imshow("frame",gray)

        if cv2.waitKey(1)& 0xFF == ord("q"):
            break
        
    except Exception as e: 
        pass 

cap.release()
cv2.destroyAllWindows()

