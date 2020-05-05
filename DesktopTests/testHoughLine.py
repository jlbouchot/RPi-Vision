# import the necessary packages
import time
import cv2
import numpy as np

from houghLineEngine import computeLinesFromGRAY

# initialize the camera and grab a reference to the raw camera capture
rawCapture = cv2.VideoCapture(0)

scale = 1
delta = 0
ddepth = cv2.CV_64F

edgeThreshold = 150 # This should be done according to the intensity of the image or something like this. 

while(True): 
	ret, frame = rawCapture.read()	
	
	[toDisplay, image, dx, dy] = computeLinesFromGRAY(frame, edgeThreshold, True, scale, delta)


	cv2.imshow("Image and lines found", np.hstack((toDisplay, frame)))
	# cv2.imshow("Image and lines found", toDisplay2)
	key = cv2.waitKey(1) & 0xFF
	# clear the stream in preparation for the next frame
	# rawCapture.truncate(0)
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

rawCapture.release()
cv2.destroyAllWindows()
