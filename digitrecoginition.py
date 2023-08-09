from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
from PIL import Image
import PIL.ImageOps

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 7500, test_size = 2500)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = LogisticRegression(multi_class='multinomial', solver='saga')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
print(score)

cap=cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Drwaing a box in the center of the video
    height, width = gray.shape
    upper_left = (int(width/2)-56, int(height/2)-56)
    bottom_right = (int(width/2)+56, int(height/2)+56)
    cv2.rectangle(gray, upper_left, bottom_right, (0, 0, 0),2)
    #To consider the area inside the box for detecting the digit
    #ROI = Region of Interest
    roi = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
    #Converting cv2 image to PIL format
    im_pil = Image.fromarray(roi)
    im_bw = im_pil.convert('L')
    im_bw_resized = im_bw.resize((28,28))
    #Invert the image
    im_bw_resized_inverted = PIL.ImageOps.invert(im_bw_resized)
    test_sample = np.array(im_bw_resized_inverted).reshape(1,784)
    test_sample = sc.transform(test_sample)
    test_pred = model.predict(test_sample)
    print('Predicted number is: ', test_pred)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()