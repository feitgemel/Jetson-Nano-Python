import cv2
import jetson.inference
import jetson.utils

# Use Python 3.6


#loading the image
img = cv2.imread('/home/feitdemo/github/Jetson-Nano-Python/dog-demo.jpg')

# convert the image format to a Cuda image
frame_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)
cude_frame = jetson.utils.cudaFromNumpy(frame_rgba)

# load the recognition network
net = jetson.inference.imageNet("googlenet")

# classify the image

class_id , confidence = net.Classify(cude_frame)

# find the description of the object
class_desc = net.GetClassDesc(class_id)

#print(class_desc)

#show the image with the class description

cv2.putText(img,class_desc,(30,80), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),4 )

cv2.imshow('img',img)
cv2.moveWindow('img',0,0) # position of the windows and the left corner
cv2.waitKey()




