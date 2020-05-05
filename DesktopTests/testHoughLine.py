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
	
	[] = computeLinesFromGRAY(inputImg, edgeThreshold, houghP = true, scale = 1, delta = 0)
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Process the optical flow from old_image to image 
	grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
	grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

	# Display the optical flow 
	pop_pop, angles = cv2.cartToPolar(grad_x, grad_y) # It's called pop_pop because "pop pop is magnitude" 
	toDisplay = cv2.cvtColor(pop_pop.astype('uint8'), cv2.COLOR_GRAY2BGR)
	toDisplay2 = cv2.cvtColor(pop_pop.astype('uint8'), cv2.COLOR_GRAY2BGR)
	# edgeImg = np.min(pop_pop, edgeThreshold)
	edgeImg = np.zeros(pop_pop.shape, dtype=float)
	edgeImg[pop_pop >= edgeThreshold] = 255.0

	edges = pop_pop >= edgeThreshold

	lines = cv2.HoughLines(edgeImg.astype('uint8'), 1, np.pi / 180, 150, None, 0, 0)
	linesP = cv2.HoughLinesP(edgeImg.astype('uint8'), 1, np.pi / 180, 50, None, 50, 10) # Corresponds to 

	# Now go through what we've got! 
	if lines is not None: # Assuming at least one line has been found  
		for oneLine in range(0,len(lines)): 
			# Go through the lines, one by one. 
			rho = lines[oneLine][0][0]
			theta = lines[oneLine][0][1]
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a * rho
			y0 = b * rho
			pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
			pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
			cv2.line(toDisplay, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)


	if linesP is not None: 
		for oneLine in range(0,len(linesP)): 
			currentLine = linesP[oneLine][0]
			cv2.line(toDisplay2, (currentLine[0], currentLine[1]), (currentLine[2], currentLine[3]), (0,0,255), 3, cv2.LINE_AA)

	# # show the frame
	# cv2.imshow("Frame", image)
	cv2.imshow("Image and lines found", np.hstack((toDisplay, toDisplay2)))
	# cv2.imshow("Image and lines found", toDisplay2)
	key = cv2.waitKey(1) & 0xFF
	# clear the stream in preparation for the next frame
	# rawCapture.truncate(0)
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

rawCapture.release()
cv2.destroyAllWindows()
