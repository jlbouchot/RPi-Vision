import cv2
import numpy as np
import time

# from matplotlib import pyplot as plt 


rawCapture = cv2.VideoCapture(0)
# rawCapture = cv2.VideoCapture(cv2.samples.findFile("vtest.avi"))
# Generate an empty old frame
old_image = None

# Initialize a keypoint detector
# Need to fiddle around for non xfeatures things
kpDetector = cv2.xfeatures2d.SIFT_create(100)
# kpDetector = cv2.xfeatures2d.SURF_create()
# kpDetector = cv2.xfeatures2d.AffineFeature2D_create() # Does not seem to exist in python
# kpDetector = cv2.xfeatures2d.FREAK_create() # simply detector
# kpDetector = cv2.AKAZE_create()
# kpDetector = cv2.ORB_create()
# kpDetector = cv2.FastFeatureDetector_create() # Does not implement detectAndCompute
# kpDetector = cv2.BRISK_create()

# Some parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

# Is it a match? 
MIN_MATCH_COUNT = 1 # For now, simply display anything (roughly speaking)

ret, frame = rawCapture.read()
old_image = frame
keypointsOld, descriptorsOld = kpDetector.detectAndCompute(old_image,None)

while(True):
	ret, frame = rawCapture.read()

	image = frame
	# image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	keypointsCurrent, descriptorsCurrent = kpDetector.detectAndCompute(image,None)

	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(descriptorsOld,descriptorsCurrent,k=2)
	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)

	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ keypointsOld[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ keypointsCurrent[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()
		h,w,d = image.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)
		old_image = cv2.polylines(old_image,[np.int32(dst)],True,255,3, cv2.LINE_AA)
	else:
		print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
		matchesMask = None


	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
		singlePointColor = None,
		matchesMask = matchesMask, # draw only inliers
		flags = 2)
	toPlot = cv2.drawMatches(old_image,keypointsOld,image,keypointsCurrent,good,None,**draw_params)

	# # show the frame
	# cv2.imshow("Frame", image)
	cv2.imshow("Optical Flow", toPlot)
	key = cv2.waitKey(1) & 0xFF
	# clear the stream in preparation for the next frame
	# rawCapture.truncate(0)
	old_image = image
	keypointsOld = keypointsCurrent
	descriptorsOld = descriptorsCurrent
	time.sleep(0.25) # Wait for the eye to actually see what is being displayed, at the cost of real time processing
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

rawCapture.release()
cv2.destroyAllWindows()
