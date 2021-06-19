import cv2
import jetson.inference
import jetson.utils

# load the video file

cap = cv2.VideoCapture('/home/feitdemo/github/Jetson-Nano-Python/Wildlife.mp4')
cap.set(3,1280)
cap.set(4,720)


#load the network
net = jetson.inference.imageNet("googlenet")

while cap.isOpened():

    re, img = cap.read()

    # convert the image to Cuda
    frame_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)
    cuda_frame = jetson.utils.cudaFromNumpy(frame_rgba)

    # classify the image
    class_id , confidence = net.Classify(cuda_frame)

    # find the object description
    class_desc = net.GetClassDesc(class_id)

    #print(class_desc)

    # more the 40% confidence , than put the text on the video file
    if confidence > 0.4 : 
        cv2.putText(img, class_desc, (30,80), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),3 )



    cv2.imshow('img',img)
    cv2.moveWindow('img',0,0)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()