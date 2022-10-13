import time

from numpy.core.defchararray import array
import torch
import cv2
import numpy as np
import pandas as pd
name_class = [
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


label_circle = ["Cam ngc chieu","Cam o to re phai", "Cam oto re trai","Cam re trai", "Cam re phai","Cam quay dau","Cam dung do", "Cam do ngay chan","Cam do ngay le","Cam do","Vong Xoay","Bao hieu huong phai","Cam xe tho so","Cam oto","Max speed 70","Max speed 60","Max speed 50","Max speed 80"]
#label_rectang = ["Bao di bo sang ngang","Ben xe bus"]
label_tri = ["Giao vs dg k uu tien phai","Giao vs dg k uu tien trai","Bao nguy hiem ng di bo", "Tre em","Ngoac nguy hiem phai","Ngoac nguy hiem trai"]


# Distance constants 
KNOWN_DISTANCE = 500 #cm
WIDTH_cir = 70 #cm
WIDTH_tri = 70 #cm
F_cir = 757.14    #w=106
F_tri = 707.14      #w=99
# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,120,255)

# defining fonts 
FONTS = cv2.FONT_HERSHEY_COMPLEX
model = torch.hub.load('yolov5', 'custom', path=r'yolov5\runs\train\yolov5s_results\weights/best.pt', source='local') # local repo




#F_tri = (width_pix*KNOWN_DISTANCE)/WIDTH_cir
#print(array_out)


# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance


with open(r"D:\NOP_LVTN\yolov4-deepsort-master\path.txt","r") as f:
    path = f.read()
#path = r"yolov4-deepsort-master\1.mp4"          # normal video
try:
    path = int(path)
except:
    pass
cap = cv2.VideoCapture(path)

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(path) # path video detecet

while True:
    ret, frame_final = cap.read()
    start_time = time.time()
    imgf = cv2.cvtColor(frame_final, cv2.COLOR_BGR2RGB)
    output2 = model(imgf)
    output2.imgs                    #array of original images (as np array) passed to model for inference
    output2.render()                #updates results.imgs with boxes and labels
    imgf = cv2.cvtColor(imgf, cv2.COLOR_BGR2RGB)
    frame_out = output2.pandas().xyxy[0]
    #print(output2.pandas().xyxy[0])
    array_detect = frame_out.to_numpy()
    for i in np.arange(array_detect.shape[0]):
        x = array_detect[i, 0]
        y = array_detect[i, 1]
        w = array_detect[i, 2] - array_detect[i, 0]
        h = array_detect[i, 3] - array_detect[i, 1]
        label = array_detect[i, 5]
        if name_class[label] in label_circle and path !=0:
            distance = distance_finder(F_cir, WIDTH_cir, w)
            print("label: ", name_class[label] + " Dien tich BB: ", round(w * h, 2), "D: ", round(distance / 100, 2),
                  "m")
            cv2.putText(imgf, f'D: {round(distance / 100, 2)} m', (int(x), int(y) - 30), FONTS, 1, GREEN, 2)
        elif name_class[label] in label_tri and path !=0:
            distance = distance_finder(F_tri, WIDTH_tri, w)
            print("label: ", name_class[label] + " Dien tich BB: ", round(w * h, 2), "D: ", round(distance / 100, 2),
                  "m")
            cv2.putText(imgf, f'D: {round(distance / 100, 2)} m', (int(x), int(y) - 30), FONTS, 1, GREEN, 2)

    fps = 1 / (time.time() - start_time)
    cv2.putText(imgf, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow('Camera', cv2.resize(imgf, (720, 720)))
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()






