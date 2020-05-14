# import the necessary packages
import time
import cv2
import numpy as np

from picamera import PiCamera # Pretty much the frame grabber
from picamera.array import PiRGBArray # Pretty much the conversion to numpy array 

from houghLineEngine import computeLinesFromGRAY

# Interesting parameters 
widthPx = 640
heightPx = 480

# Initilize the RPi camera
camera = PiCamera()
camera.resolution = (widthPx, heightPx)
camera.framerate = 30

rawCapture = PiRGBArray(camera, size=(widthPx, heightPx))

# let the camera warm up
time.sleep(0.1)

scale = 1
delta = 0
ddepth = cv2.CV_64F

edgeThreshold = 150 # This should be done according to the intensity of the image or something like this. 

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port = True):
	image = frame.array # cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)
	[toDisplay, grayImg, dx, dy] = computeLinesFromGRAY(image, edgeThreshold, True, scale, delta)


	cv2.imshow("Image and lines found", np.hstack((toDisplay, image)))
	# cv2.imshow("Original image", image)
	# cv2.imshow("Image and lines found", toDisplay2)
	key = cv2.waitKey(1) & 0xFF
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

#rawCapture.release()
cv2.destroyAllWindows()
