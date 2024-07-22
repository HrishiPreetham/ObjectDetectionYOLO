import cv2
import numpy as np


cap = cv2.VideoCapture(0)

kernel= None

# Create the background subtractor object
foog = cv2.createBackgroundSubtractorMOG2(detectShadows = True,varThreshold = 50, history = 2800)

thresh = 1100

while(1):
    
    ret, frame = cap.read() 
    frame = cv2.flip(frame,1)
    if not ret:
        break
      
    # Apply the background object on each frame
    fgmask = foog.apply(frame)
    # Get rid of the shadows
    ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    
    fgmask = cv2.dilate(fgmask,kernel,iterations = 4)

    contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        
        # Get the maximum contour
        cnt = max(contours, key = cv2.contourArea)
 
 
        # make sure the contour area is somewhat hihger than some threshold to make sure its a person and not some noise.
        if cv2.contourArea(cnt) > thresh:
 
            # Draw a bounding box around the person and label it as person detected
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x ,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,'Person Detected',(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
 
 
    # Stack both frames and show the image
    fgmask_3 = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((fgmask_3,frame))
    cv2.imshow('Combined',cv2.resize(stacked,None,fx=0.65,fy=0.65))
 
    k = cv2.waitKey(40) and 0xff
    if k == 27:
        break
 
cap.release()
cv2.destroyAllWindows()