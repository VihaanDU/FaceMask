import cv2
import os
from keras_preprocessing.image import img_to_array
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import tkinter as tk
from tkinter import Variable, filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
model = load_model("mask_recog.h5")


my_w = tk.Tk()
my_w.geometry("500x500")# Size of the window
my_w.config(bg='green')
my_w.title('Face Mask Detection')
my_font1=('times', 18, 'bold')

def face_mask_detector():
  frame = cv2.imread(filename)
  frame=cv2.resize(frame,(600,600))
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(60, 60),
                                        flags=cv2.CASCADE_SCALE_IMAGE)

  print(faces)

  faces_list=[]
  preds=[]
  for (x, y, w, h) in faces:
      face_frame = frame[y:y+h,x:x+w]
      face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
      face_frame = cv2.resize(face_frame, (224, 224))
      face_frame = img_to_array(face_frame)
      face_frame = np.expand_dims(face_frame, axis=0)
      face_frame =  preprocess_input(face_frame)
      faces_list.append(face_frame)
      if len(faces_list)>0:
          preds = model.predict(faces_list)
      for pred in preds:
          (mask, withoutMask) = pred
      label = "Mask" if mask > withoutMask else "No Mask"
      color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
      label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
      cv2.putText(frame, label, (x, y- 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

      cv2.rectangle(frame, (x, y), (x + w, y + h),color, 3)
  cv2.imshow('img',frame)



def upload_file():
    global img
    global filename

    print("Upload Button Clicked")
    
    f_types = [('Jpg Files', '.jpg'), ('PNG files', '.png')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = ImageTk.PhotoImage(file=filename)
    b2 =tk.Button(my_w,image=img) # using Button 
    b2.grid(row=15,column=1)


l1 = tk.Label(my_w,text='Add Student Data with Photo',width=30,font=my_font1)  
l1.grid(row=1,column=1)

b1 = tk.Button(my_w, text='Upload File', width=20,command = lambda:upload_file())
b1.grid(row=2,column=1) 

print("Am looping")

b2=tk.Button(my_w, text='Start', width=20, command=face_mask_detector)
b2.grid(row=5,column=1)

b3=tk.Button(my_w, text='End', width=20, command=my_w.destroy)
b3.grid(row=8,column=1)
my_w.mainloop()