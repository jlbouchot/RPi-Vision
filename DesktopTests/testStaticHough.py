# import the necessary packages
import time
import cv2
import numpy as np

import sys

from houghLineEngine import computeLinesFromGRAY

scale = 1
delta = 0
ddepth = cv2.CV_64F

edgeThreshold = 150 # This should be done according to the intensity of the image or something like this. 


frame = cv2.imread(sys.argv[1])

[toDisplay, image, dx, dy] = computeLinesFromGRAY(frame, edgeThreshold, True, scale, delta)

cv2.imshow("Image and lines found", np.hstack((toDisplay, frame)))
# cv2.imshow("Image and lines found", toDisplay2)
wait = True
while wait: 
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		wait = False

cv2.destroyAllWindows()
