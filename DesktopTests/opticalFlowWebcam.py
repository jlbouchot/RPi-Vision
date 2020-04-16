# import the necessary packages
import time
import cv2
import numpy as np

# Interesting parameters 
widthPx = 640
heightPx = 480

# initialize the camera and grab a reference to the raw camera capture
rawCapture = cv2.VideoCapture(0)
# rawCapture = cv2.VideoCapture(cv2.samples.findFile("vtest.avi"))
# Generate an empty old frame
old_image = None


while(True): 
	ret, frame = rawCapture.read()

	image = frame
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


	if old_image is None: 
		old_image = image

	# Process the optical flow from old_image to image 
	flow = cv2.calcOpticalFlowFarneback(old_image,image, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	# Display the optical flow 
	pop_pop, angles = cv2.cartToPolar(flow[...,0], flow[...,1]) # It's called pop_pop because "pop pop is magnitude"

	flowToDisp = np.zeros_like(frame)

	flowToDisp[:,:,0] = angles*180/np.pi/2
	flowToDisp[:,:,1] = 255
	flowToDisp[:,:,2] = cv2.normalize(pop_pop, None, 0, 255, cv2.NORM_MINMAX)

	toTheDisplay = cv2.cvtColor(flowToDisp, cv2.COLOR_HSV2BGR)
	# # show the frame
	# cv2.imshow("Frame", image)
	cv2.imshow("Optical Flow", toTheDisplay)
	key = cv2.waitKey(1) & 0xFF
	# clear the stream in preparation for the next frame
	# rawCapture.truncate(0)
	old_image = image
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

rawCapture.release()
cv2.destroyAllWindows()
