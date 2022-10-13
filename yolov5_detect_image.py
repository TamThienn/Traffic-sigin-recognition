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

label_circle = ["Cam ngc chieu", "Cam o to re phai", "Cam oto re trai", "Cam re trai", "Cam re phai", "Cam quay dau",
                "Cam dung do", "Cam do ngay chan", "Cam do ngay le", "Cam do", "Vong Xoay", "Bao hieu huong phai",
                "Cam xe tho so", "Cam oto", "Max speed 70", "Max speed 60", "Max speed 50", "Max speed 80"]
# label_rectang = ["Bao di bo sang ngang","Ben xe bus"]
label_tri = ["Giao vs dg k uu tien phai", "Giao vs dg k uu tien trai", "Bao nguy hiem ng di bo", "Tre em",
             "Ngoac nguy hiem phai", "Ngoac nguy hiem trai"]

# Distance constants
KNOWN_DISTANCE = 500  # cm
WIDTH_cir = 70  # cm
WIDTH_tri = 70  # cm
# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 120, 255)

# defining fonts
FONTS = cv2.FONT_HERSHEY_COMPLEX

# Compute F Circle
path_cir = r"12.jpg"
model = torch.hub.load('yolov5', 'custom', path=r'yolov5\runs\train\yolov5s_results\weights/best.pt',
                       source='local')  # local repo
img = cv2.imread(path_cir)
img = cv2.resize(img, (720, 720))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
output = model(img)
frame_out = output.pandas().xyxy[0]
array_out = frame_out.to_numpy()
# print(np.shape(array_out))

width_pix_cir = array_out[0, 2] - array_out[0, 0]

F_cir = (width_pix_cir * KNOWN_DISTANCE) / WIDTH_cir
print("Tieu cu dg tron", F_cir)

# Compute F Triangle
path_tri = r"17.jpg"
model = torch.hub.load('yolov5', 'custom', path=r'yolov5\runs\train\yolov5s_results\weights/best.pt',
                       source='local')  # local repo
img = cv2.imread(path_tri)
img = cv2.resize(img, (720, 720))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
output = model(img)
frame_out = output.pandas().xyxy[0]
array_out = frame_out.to_numpy()
# print(np.shape(array_out))

width_pix_tri = array_out[0, 2] - array_out[0, 0]

F_tri = (width_pix_tri * KNOWN_DISTANCE) / WIDTH_tri
print("Tieu cu tam giac", F_tri)




def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance




path_imge = r"C:\Users\Admin\Desktop\tensorflow-yolov4-tflite-master\data\images\12.jpg"
img = cv2.imread(path_imge)
imgf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgf = cv2.resize(imgf,(720,720))
# print(type(img))
output2 = model(imgf)
output2.imgs  # array of original images (as np array) passed to model for inference
output2.render()  # updates results.imgs with boxes and labels
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
    if name_class[label] in label_circle:
        distance = distance_finder(F_cir, WIDTH_cir, w)
        print("width", w)
        print("label: ", name_class[label] + " Dien tich BB: ", round(w * h, 2), "D: ", round(distance / 100, 2),
                  "m")
        cv2.putText(imgf, f'D: {round(distance / 100, 2)} m', (int(x), int(y) - 30), FONTS, 1, GREEN, 2)
    elif name_class[label] in label_tri:
        distance = distance_finder(F_tri, WIDTH_tri, w)
        print("label: ", name_class[label] + " Dien tich BB: ", round(w * h, 2), "D: ", round(distance / 100, 2),
                  "m")
        cv2.putText(imgf, f'D: {round(distance / 100, 2)} m', (int(x), int(y) - 30), FONTS, 1, GREEN, 2)


cv2.imshow('Camera', imgf)
cv2.waitKey(0)
cv2.destroyAllWindows()















