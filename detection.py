import cv2 as cv
import numpy as np
modelcfg = '/home/epl/tk/yolov3_ts_train.cfg'
modelweight = '/home/epl/tk/yolov3_ts_train_1000.weights'
net = cv.dnn.readNetFromDarknet(modelcfg,modelweight)
if net.empty():
	print("failed to load")
classes = ['prohibitory','danger','mandatory','others']
img = cv.imread('00042.jpg')
img = cv.resize(img,(1280,720))	
hight,width,_ = img.shape
blob = cv.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)
layer = net.getLayerNames()
outputlayers = [layer[i-1] for i in net.getUnconnectedOutLayers()]
net.setInput(blob)
outputs = net.forward(outputlayers)
class_ids = []
confidences =[]
boxes =[]
for output in outputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.1	:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


indexes = cv.dnn.NMSBoxes(boxes,confidences,.5,.4)
font = cv.FONT_HERSHEY_PLAIN
colors = (255,0,0)	
if  len(indexes)>0:
    for i in indexes.flatten():
         x,y,w,h = boxes[i]
         label = str(classes[class_ids[i]])
         confidence = str(round(confidences[i],2))
         color = colors[i]
         cv.rectangle(img,(x,y),(x+w,y+h),color,2)
         cv.putText(img,label + " " + confidence, (x,y+400),font,2,color,2)
cv.namedWindow("img", cv.WINDOW_NORMAL)
  
# Using resizeWindow()
cv.resizeWindow("img", 700, 700)
cv.imshow('img',img)
cv.waitKey(0)


