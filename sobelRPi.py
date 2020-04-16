# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

# Interesting parameters 
widthPx = 640
heightPx = 480

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (widthPx, heightPx)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(widthPx, heightPx))
# allow the camera to warmup
time.sleep(0.1)

scale = 1
delta = 0
ddepth = cv2.CV_64F

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)

	# Process the optical flow from old_image to image 
	grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
	grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

	# Display the optical flow 
	pop_pop, angles = cv2.cartToPolar(grad_x, grad_y) # It's called pop_pop because "pop pop is magnitude"

	flowToDisp = np.zeros_like(frame.array)

	flowToDisp[:,:,0] = angles*180/np.pi/2
	flowToDisp[:,:,1] = 255
	flowToDisp[:,:,2] = cv2.normalize(pop_pop, None, 0, 255, cv2.NORM_MINMAX)
	toTheDisplay = cv2.cvtColor(flowToDisp, cv2.COLOR_HSV2BGR)
	# # show the frame
	# cv2.imshow("Frame", image)
	cv2.imshow("Optical Flow", toTheDisplay)
	key = cv2.waitKey(1) & 0xFF
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
