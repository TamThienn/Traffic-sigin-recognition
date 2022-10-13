import cv2
import numpy as np
import glob
import random


# Load Yolo
net = cv2.dnn.readNet("v4.weights", "v4.cfg")

# Name custom object
classes = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '4', '5', '6', '7', '8', '9']
# Images path
#images_path = glob.glob(r"D:\data_newww\train\test_night\*.jpg")
#images_path = glob.glob(r"D:\NOP_LVTN\yolov4-deepsort-master\12.jpg")
images_path = glob.glob(r"D:\NOP_LVTN\image_distance\17.jpg")


layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
random.shuffle(images_path)
# loop through all the images
for img_path in images_path:
    # Loading image
    img = cv2.imread(img_path)
    h_origin,w_origin,_  = img.shape
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    img = cv2.resize(img,(720,720))
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
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
                #print(class_id)
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indexes)
    #print(indexes)
    #print(boxes)
    print(confidences)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            confidence = str(round(confidences[i], 2))
            x, y, w, h = boxes[i]
            print(x,y,w,h)
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            #cv2.rectangle(img, (x+10, y+10), (x-10 + w, y-10 + h), color, 2)
            #cv2.circle(img,(x+10,y+10),radius=2,color=[0,0,255])
            #cv2.circle(img, (x-10+w, y-10+h), radius=2, color=[0, 0, 255])

            cv2.rectangle(img, (x+10, y-5), (x + w-5, y+ h+10), color, 2)

            cv2.putText(img, label, (x-30, y - 30), font, 3, color, 2)
            cv2.putText(img, str(confidence), (x+30, y - 30), font, 3, color, 2)
    img = cv2.resize(img,dsize=(width,height))
    cv2.imshow("Image", img)
    key = cv2.waitKey(0)


cv2.destroyAllWindows()