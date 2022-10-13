import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from tkinter import messagebox
import tkinter as tk
from PIL import Image, ImageTk
import PIL.Image,PIL.ImageTk
from tkinter import filedialog
import pyodbc
from numpy.core.defchararray import array
import torch
import pandas as pd
import os
net = cv2.dnn.readNetFromDarknet("v4.cfg","v4.weights")
model = torch.hub.load('yolov5', 'custom', path='yolov5\\runs\\train\yolov5s_results\\weights\\best.pt', source='local') # local repo
### Change here for custom classes for trained model

classes = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '4', '5', '6', '7', '8', '9']

bw = 0
distance_or_not = 0
webcam_or_video_or_image = None
name_class_v5 = [
                 "Cam ngc chieu",
                 "Cam o to re phai",
                 "Vong Xoay",
                "Bao di bo sang ngang",
                 "Lan moto",
                 "Lan oto",
                 "Lan oto moto",
                 "Bao hieu huong phai",
                "Giao vs dg k uu tien phai",
                 "Giao vs dg k uu tien trai",
                 "Bao nguy hiem ng di bo",
                 "Tre em",
                 "Cam oto re trai",
                 "Cau vuot",
                 "Cam xe tho so",
                 "Ngoac nguy hiem phai",
                 "Ngoac nguy hiem trai",
                 "Ben xe bus",
                "Cam oto",
                 "Max speed 40",
                 "Max speed 50",
                 "Max speed 60",
                 "Max speed 80",
                 "Cam re trai",
                "Max speed 70",
                 "Cam re phai",
                 "Cam quay dau",
                 "Cam dung do",
                "Cam do ngay chan",
                 "Cam do ngay le",
                "Cam do"
]

name_class_v4 = {
                "0" : "Cam ngc chieu",
                "1" : "Cam o to re phai",
                "10" : "Vong Xoay",
                "11" : "Bao di bo sang ngang",
                "12" : "Lan moto",
                "13" : "Lan oto",
                "14" : "Lan oto moto",
                "15" : "Bao hieu huong phai",
                "16" : "Giao vs dg k uu tien phai",
                "17" : "Giao vs dg k uu tien trai",
                "18" : "Bao nguy hiem ng di bo",
                "19" : "Tre em",
                "2" : "Cam oto re trai",
                "20" : "Cau vuot",
                "21" : "Cam xe tho so",
                "22" : "Ngoac nguy hiem phai",
                "23" : "Ngoac nguy hiem trai",
                "24" : "Ben xe bus",
                "25" : "Cam oto",
                "26" : "Max speed 40",
                "27" : "Max speed 50",
                "28" : "Max speed 60",
                "29" : "Max speed 80",
                "3"  : "Cam re trai",
                "30" : "Max speed 70",
                "4"  : "Cam re phai",
                "5"  : "Cam quay dau",
                "6"  : "Cam dung do",
                "7" : "Cam do ngay chan",
                "8" : "Cam do ngay le",
                "9" : "Cam do"
}







F_cir = 757.14    #w=106
F_tri = 707.14      #w=99
KNOWN_DISTANCE = 500 #cm
WIDTH_cir = 70 #cm
WIDTH_tri = 70 #cm
label_circle = ["Cam ngc chieu","Cam o to re phai", "Cam oto re trai","Cam re trai", "Cam re phai","Cam quay dau","Cam dung do", "Cam do ngay chan","Cam do ngay le","Cam do","Vong Xoay","Bao hieu huong phai","Cam xe tho so","Cam oto","Max speed 70","Max speed 60","Max speed 50","Max speed 80"]
#label_rectang = ["Bao di bo sang ngang","Ben xe bus"]
label_tri = ["Giao vs dg k uu tien phai","Giao vs dg k uu tien trai","Bao nguy hiem ng di bo", "Tre em","Ngoac nguy hiem phai","Ngoac nguy hiem trai"]



image_v4 = 0
image_v5 = 0



class Detect_image_v5():
    def __init__(self,parent):
        self.path = webcam_or_video_or_image
        self.COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        self.GREEN = (0, 120, 255)
        # defining fonts
        self.FONTS = cv2.FONT_HERSHEY_COMPLEX
        self.detect()
    def find_distance(self,focal_length, real_width, width_pixel):
        new_distance = (real_width * focal_length) / width_pixel
        return new_distance
    def detect(self):
        imgf = cv2.imread(self.path)
        imgf = cv2.cvtColor(imgf, cv2.COLOR_BGR2RGB)
        imgf = cv2.resize(imgf, (720, 720))
        # print(type(img))
        output2 = model(imgf)
        output2.imgs                    # array of original images (as np array) passed to model for inference
        output2.render()                # updates results.imgs with boxes and labels
        imgf = cv2.cvtColor(imgf, cv2.COLOR_BGR2RGB)
        frame_out = output2.pandas().xyxy[0]
        # print(output2.pandas().xyxy[0])
        array_detect = frame_out.to_numpy()
        # print(array_detect)
        for i in np.arange(array_detect.shape[0]):
            x = array_detect[i, 0]
            y = array_detect[i, 1]
            w = array_detect[i, 2] - array_detect[i, 0]
            h = array_detect[i, 3] - array_detect[i, 1]
            label = array_detect[i, 5]
            if name_class_v5[label] in label_circle:
                distance = self.find_distance(F_cir, WIDTH_cir, w)
                print("label: ", name_class_v5[label] + " Dien tich BB: ", round(w * h, 2), "D: ",
                      round(distance / 100, 2),
                      "m")
                cv2.putText(imgf, f'D: {round(distance / 100, 2)} m', (int(x), int(y) - 30), self.FONTS, 1, self.GREEN, 2)
            elif name_class_v5[label] in label_tri:
                distance = self.find_distance(F_tri, WIDTH_tri, w)
                print("label: ", name_class_v5[label] + " Dien tich BB: ", round(w * h, 2), "D: ",
                      round(distance / 100, 2),
                      "m")
                cv2.putText(imgf, f'D: {round(distance / 100, 2)} m', (int(x), int(y) - 30), self.FONTS, 1, self.GREEN, 2)

        cv2.imshow('Camera', cv2.resize(imgf, (720, 720)))

class Detect_image_v4():
    def __init__(self,parent):
        self.path = webcam_or_video_or_image
        self.detect()

    def find_distance(self,focal_length, real_width, width_pixel):
        new_distance = (real_width * focal_length) / width_pixel
        return new_distance
    def detect(self):
        image_path = glob.glob(self.path)
        layer_names = net.getLayerNames()                         #lay ra toan bo cac layer cua network
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]      #lay ra lop cuoi cung cua mang => 2 dau yolo
        colors = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        img = cv2.imread(image_path[0])
        h_origin, w_origin, _ = img.shape
        img = cv2.resize(img,(720,720))
        height, width, chanel = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)      #chuyen anh sang dinh dang blob phu hop vs darknet True=> BGR to RGB
        net.setInput(blob)
        outs = net.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.7:
                    # Object detected
                    # print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.6)                #score threshold =0.7, iou threshold=0.6
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                confidence = str(round(confidences[i], 2))
                x, y, w, h = boxes[i]
                label_num = str(classes[class_ids[i]])                              #display label number
                label_name = str(name_class_v5[class_ids[i]])
                if label_name in label_circle:
                    distance = self.find_distance(F_cir,WIDTH_cir,w)
                    color = colors[int(classes[class_ids[i]]) % len(colors)]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label_num, (x - 30, y - 30), font, 2, color, 2)
                    cv2.putText(img, str(confidence), (x + 30, y - 30), font, 2, color, 2)
                    cv2.putText(img, "D: "+ str(round(distance/100,2))+"m",(x,y+150),font,2,color,2)
                elif label_name in label_tri:
                    distance = self.find_distance(F_tri,WIDTH_tri,w)
                    color = colors[int(classes[class_ids[i]]) % len(colors)]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label_num, (x - 30, y - 30), font, 2, color, 2)
                    cv2.putText(img, str(confidence), (x + 30, y - 30), font, 2, color, 2)
                    cv2.putText(img, "D: "+ str(round(distance/100,2))+"m",(x,y+150),font,2,color,2)
        img = cv2.resize(img, dsize=(width, height))
        cv2.imshow("Image", img)



class Browser_v4_deep(tk.Frame):
    def __init__(self,parent):
        self.browser = tk.Toplevel(parent)
        self.filename = "File Explorer using Tkinter"
        frame = tk.Frame.__init__(self, parent)
        self.label_file_explorer = tk.Label(self.browser,
                                    text= self.filename,
                                    width=100, height=4,
                                    fg="blue").pack()

        button_explore = tk.Button(self.browser,
                                text="Browse Files",
                                command=self.browseFiles).pack()

    def browseFiles(self):
        self.filename = tk.filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("Text files",
                                                      "*.txt*"),
                                                     ("all files",
                                                      "*.*")))
        self.label_file_explorer = tk.Label(self.browser,
                                           text=self.filename,
                                           width=100, height=4,
                                           fg="blue").pack()
        # Change label contents
        path_video = str(self.filename)


        os.system(
            "python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny")
        os.system(
            "python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video "+ path_video+" --tiny")

class Browser_v4(tk.Frame):
    def __init__(self,parent):
        self.browser = tk.Toplevel(parent)
        self.filename = "File Explorer using Tkinter"
        frame = tk.Frame.__init__(self, parent)
        self.label_file_explorer = tk.Label(self.browser,
                                    text= self.filename,
                                    width=100, height=4,
                                    fg="blue").pack()

        button_explore = tk.Button(self.browser,
                                text="Browse Files",
                                command=self.browseFiles).pack()

    def browseFiles(self):
        global webcam_or_video_or_image
        self.filename = tk.filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("Text files",
                                                      "*.txt*"),
                                                     ("all files",
                                                      "*.*")))
        self.label_file_explorer = tk.Label(self.browser,
                                           text=self.filename,
                                           width=100, height=4,
                                           fg="blue").pack()
        # Change label contents
        webcam_or_video_or_image = str(self.filename)
        if image_v4 == 1:
            Detect_image_v4(self)
        else:
            with open("path.txt","w+") as f:
                f.write(webcam_or_video_or_image)
            os.system("python yolov4_detect_video_distance.py")
            #CamView_v4(self)


class Browser_v5(tk.Frame):
    def __init__(self,parent):
        self.browser = tk.Toplevel(parent)
        self.filename = "File Explorer using Tkinter"
        frame = tk.Frame.__init__(self, parent)
        self.label_file_explorer = tk.Label(self.browser,
                                    text= self.filename,
                                    width=100, height=4,
                                    fg="blue").pack()

        button_explore = tk.Button(self.browser,
                                text="Browse Files",
                                command=self.browseFiles).pack()

    def browseFiles(self):
        global webcam_or_video_or_image
        self.filename = tk.filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("Text files",
                                                      "*.txt*"),
                                                     ("all files",
                                                      "*.*")))
        self.label_file_explorer = tk.Label(self.browser,
                                           text=self.filename,
                                           width=100, height=4,
                                           fg="blue").pack()
        # Change label contents

        webcam_or_video_or_image = str(self.filename)
        #print("image_v5", image_v5)
        if image_v5 == 1:
            Detect_image_v5(self)
        else:
            with open("path.txt", "w+") as f:
                f.write(str(webcam_or_video_or_image))
            os.system("python yolov5_detect_video_distance.py")



class load_V4(tk.Frame):
    def __init__(self,parent):
        self.load_V4 = tk.Toplevel(parent)
        frame = tk.Frame.__init__(self, parent)
        self.lmain1 = tk.Label(self.load_V4)
        self.lmain1.pack()
        option1 = tk.Button(self.load_V4,text="Image",command=self.load_image).pack()
        option2 = tk.Button(self.load_V4,text="Webcam",command=self.load_webcam).pack()
        option3 = tk.Button(self.load_V4,text="Video",command=self.load_video).pack()
        option4 = tk.Button(self.load_V4, text="deep_sort", command=self.deepsort).pack()
    def load_video(self):
        global image_v4
        image_v4 = 0
        Browser_v4(self)
    def load_webcam(self):
        global webcam_or_video_or_image
        webcam_or_video_or_image = 0
        with open("path.txt", "w+") as f:
            f.write(str(webcam_or_video_or_image))
        os.system("python yolov4_detect_video_distance.py")
        #CamView_v4(self)
    def load_image(self):
        global image_v4
        image_v4 = 1
        Browser_v4(self)
    def deepsort(self):
        Browser_v4_deep(self)

class load_V5(tk.Frame):
    def __init__(self,parent):
        self.load_V5 = tk.Toplevel(parent)
        frame = tk.Frame.__init__(self, parent)
        self.lmain1 = tk.Label(self.load_V5)
        self.lmain1.pack()
        option1 = tk.Button(self.load_V5,text="Image",command=self.load_image).pack()
        option2 = tk.Button(self.load_V5,text="Webcam",command=self.load_webcam).pack()
        option3 = tk.Button(self.load_V5,text="Video",command=self.load_video).pack()

    def load_video(self):
        global image_v5
        image_v5 = 0
        Browser_v5(self)
    def load_webcam(self):
        global webcam_or_video_or_image
        webcam_or_video_or_image = 0
        with open("path.txt", "w+") as f:
            f.write(str(webcam_or_video_or_image))
        os.system("python yolov5_detect_video_distance.py")

    def load_image(self):
        global image_v5
        image_v5 = 1
        Browser_v5(self)





class v4_v5(tk.Frame):
    def __init__(self,parent):
        self.option = tk.Toplevel(parent)
        frame = tk.Frame.__init__(self, parent)
        self.lmain1 = tk.Label(self.option)
        self.lmain1.pack()
        V4 = tk.Button(self.option,text="Yolo-v4tiny",command=self.load_V4).pack()
        V5 = tk.Button(self.option,text="Yolov5",command=self.load_V5).pack()
    def load_V4(self):
        load_V4(self)
    def load_V5(self):
        load_V5(self)




root = tk.Tk()
root.geometry("300x200")
root['background'] = 'Blue'

def load():
    if (str(user.get()) == "admin" and str(password.get()) == "admin"):
        option(root)
    elif (str(user.get()) == None):
        messagebox.showinfo("warning", "please enter user")
    elif (str(password.get())== None):
        messagebox.showinfo("warning", "please enter password")
    else:
        messagebox.showinfo("warning", "error user or password")
def load_winsign():
    win_signup(root)

def Check_database():
    cursor = conx.cursor()
    usename = ['admin']
    passwords = ['admin']
    for row in cursor.execute('select * from Account1'):
        usename.append(row.usename)
        passwords.append(row.passwords)
    if user.get() in usename and password.get() in passwords:
        v4_v5(root)

    elif user.get()== '':
        messagebox.showinfo("warning", "please enter user")
    elif password.get() == '':
        messagebox.showinfo("warning", "please enter password")
    elif user.get() not in usename or password.get() not in passwords:
        messagebox.showinfo("warning", "error user or password")
class win_signup(tk.Frame):
    def __init__(self,parent):
        self.win_sign = tk.Toplevel(parent)
        frame = tk.Frame.__init__(self, parent)
        #self.lmain = tk.Label(self.win_sign)
        #self.lmain.pack(
        tk.Label(self.win_sign, text="UserName").grid(column=0, row=0)
        tk.Label(self.win_sign, text="Password").grid(column=0, row=1)
        tk.Label(self.win_sign, text="Confirm Password").grid(column=0, row=2)
        self.new_user = tk.Entry(self.win_sign, width=20)
        self.new_user.grid(column=1, row=0)

        self.new_password = tk.Entry(self.win_sign, width=20)
        self.new_password.grid(column=1, row=1)
        self.new_password.config(show="*")
        self.new_confirm_password = tk.Entry(self.win_sign, width=20)
        self.new_confirm_password.grid(column=1, row=2)
        self.new_confirm_password.config(show="*")
        button = tk.Button(self.win_sign, text="confirm",command=self.sign_up_data)
        button.grid(column=1, row=3)
    def sign_up_data(self):
        if self.new_user.get() != '' and self.new_password.get() != '' and self.new_password.get() == self.new_confirm_password.get():
            cursor = conx.cursor()
            cursor.execute("Insert Account1 values (?,?)",self.new_user.get(),self.new_password.get())
            conx.commit()
            messagebox.showinfo("Congratulation", "You successfully signed up ")
        else:
            messagebox.showinfo("warning", "Your password didn't match ")





conx = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-G2GES0F\THIEN;Database=socket_account; UID=TamThien;PWD=123456789;')


tk.Label(root,text="UserName").grid(column=0,row=0)
tk.Label(root,text="Password").grid(column=0,row=1)

user = tk.Entry(root,width=20)
user.grid(column=1,row=0)

password = tk.Entry(root,width=20)
password.grid(column=1,row=1)
password.config(show="*")
button = tk.Button(root,text="sign in",command=Check_database)
button.grid(column=1,row = 2)
button = tk.Button(root,text="sign up",command=load_winsign)
button.grid(column=1,row = 3)
button_exit = tk.Button(root,text="exit",command=root.quit).grid(column=1,row=4)

root.mainloop()

# BB hình tròn : Đường kính: 70cm
# BB hình bác giác : Đường kính: 60cm
# BB tam giác: chiều dài cạnh tam giác :70cm

