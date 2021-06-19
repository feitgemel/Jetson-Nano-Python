import cv2
import jetson.inference
import jetson.utils


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

net = jetson.inference.imageNet("googlenet")

while cap.isOpened():

    re, img = cap.read()
    
    # convert the image to Cuda format
    frame_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    cude_frame = jetson.utils.cudaFromNumpy(frame_rgba)

    # classify the image
    class_id , confidence = net.Classify(cude_frame)

    # find the description of the object
    class_desc = net.GetClassDesc(class_id)

    if confidence > 0.4 :
        cv2.putText(img , class_desc , (30,80), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),3)


    cv2.imshow('img',img)
    cv2.moveWindow('img',0,0)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

