
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

#auther - Jay Shankar Bhatt
# using this code without author's permission other then leaning task is strictly prohibited

## provide the path for testing cofing file and tained model form colab
net = cv2.dnn.readNetFromDarknet("v4.cfg","v4.weights")
### Change here for custom classes for trained model

classes = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '4', '5', '6', '7', '8', '9']
name_class = {
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
label_circle = ["Cam ngc chieu","Cam o to re phai", "Cam oto re trai","Cam re trai", "Cam re phai","Cam quay dau","Cam dung do", "Cam do ngay chan","Cam do ngay le","Cam do","Vong Xoay","Bao hieu huong phai","Cam xe tho so","Cam oto"]
#label_rectang = ["Bao di bo sang ngang","Ben xe bus"]
label_tri = ["Giao vs dg k uu tien phai","Giao vs dg k uu tien trai","Bao nguy hiem ng di bo", "Tre em","Ngoac nguy hiem phai","Ngoac nguy hiem trai"]

# caculate distance
KNOWN_DISTANCE = 500 #cm
WIDTH_cir = 70 #cm
WIDTH_tri = 70 #cm
F_cir = 757.14    #w=106
F_tri = 707.14      #w=99

def find_distance(focal_length,real_width,width_pixel):
    new_distance = (real_width * focal_length)/width_pixel
    return new_distance

#path = r"yolov4-deepsort-master\1.mp4"          # normal video

with open(r"D:\NOP_LVTN\yolov4-deepsort-master\path.txt","r") as f:
    path = f.read()
#path = r"yolov4-deepsort-master\1.mp4"          # normal video
try:
    path = int(path)
except:
    pass
cap = cv2.VideoCapture(path)
frame_num = 0
lis_fps = []
avg_fps = 0

while 1:
    _, img1 = cap.read()
    img= cv2.resize(img1, (720, 720))
    hight, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    start_time = time.time()
    net.setInput(blob)
    output_layers_name = net.getUnconnectedOutLayersNames()
    frame_num += 1
    layerOutputs = net.forward(output_layers_name)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3] * hight)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .7, .6)
    font = cv2.FONT_HERSHEY_PLAIN
    #colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    # colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    colors = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            print("Width ", w)
            color = colors[int(classes[class_ids[i]]) % len(colors)]
            label = str(classes[class_ids[i]])
            #print(label)
            if name_class[label] in label_circle and path != 0:
                distance = find_distance(F_cir,WIDTH_cir,w)
                cv2.putText(img, "D: " + str(round(distance/100,2)) +"m", (x, y - 20), font, 1.5, color, 2)
                print("frame num",frame_num, "label: ", str(label) + " Dien tich BB: ", round(w * h, 2), " D:", round(distance / 100, 2), "m")
            elif name_class[label] in label_tri and path != 0:
                distance = find_distance(F_tri,WIDTH_tri,w)
                cv2.putText("frame num",frame_num, img, "D: " + str(round(distance/100,2))+"m", (x, y - 20), font, 1.5, color, 2)
                print("label: ", str(label) + " Dien tich BB: ", round(w * h, 2), " D:", round(distance / 100, 2), "m")
            confidence = str(round(confidences[i], 2))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, name_class[label] + " " + confidence, (x, y - 40), font, 1.5, color, 2)
            #cv2.putText(img, label + "-" + confidence, (x, y - 10)), 0, 0.75,(255, 255, 255), 2)
            #cv2.putText(img, "D: " + str(distance_predict), (x, y + 140), font, 2, color, 2)
    fps = 1/ (time.time()-start_time)
    lis_fps.append(fps)
    if len(lis_fps) %10 == 0:
        avg_fps = sum(lis_fps)/len(lis_fps)
        lis_fps = []
    cv2.putText(img, str(int(avg_fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow('img', img)
    frame_num += 1
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()