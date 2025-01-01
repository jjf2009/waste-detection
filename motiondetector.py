# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2

# construct the argument parser and parse the arguments, for interaction/user interface in the CLI
ap = argparse.ArgumentParser() 															# create an argument parser object from argparse module, provides a way to define what arguments the script expects and how to parse them.
ap.add_argument("-v", "--video", help="path to the video file")							# allows user to specify path to a video file (adds an argument to the parser), the argument is a video file, argument could be either -v or --video, with the help as a description by -h
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")	# allows user to specify the minimum area size (adds an argument to the parser) otherwise default 500px, the argument is an integer, argument could be either -a or --min-area, with the help as a description by -h
args = vars(ap.parse_args())															# parse the arguments, convert them to a dictionary, and store them in args, allows the script to use the values provided by the user. Converting to a dictionary makes it convenient to access the argument values using keys.

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()														# start the video stream, src=0 (source = 0) is the webcam av input, start() starts the video stream of webcam
	time.sleep(2.0)																		# pauses the execution for 2 seconds to allow the camera sensor to warm up and stabilize before capturing frames.
# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])												# read the video file, args["video"] is the path to the video file, cv2.VideoCapture is a class from the OpenCV library that captures video from a camera
	
# initialize the first frame in the video stream, to indicate it has not been set yet
firstFrame = None

# loop over the frames of the video
while True:
	# grab the current frame and initialize the ActionOFF/ActionOn text
	frame = vs.read()																	# read the next frame from the video stream, read() returns a tuple, the first element is a boolean indicating if the frame was successfully read, if yes, then the second element will be the frame itself
	frame = frame if args.get("video", None) is None else frame[1]						# if the video argument is None, then we are reading from webcam, so the frame which was already read is the first element of the tuple, otherwise it could be a video source, so then it will the second element, returned by vs.read()
	text = "ActionOn"

	# if the frame could not be grabbed, then we have reached the end of the video
	if frame is None:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)											# resize the frame to have a width of 500 pixels, imutils.resize() is a function from the imutils library that resizes the frame while maintaining the aspect ratio, to improve processing speed and reduce the number of pixels to process
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)										# convert the frame to grayscale, cv2.cvtColor() is a function from the OpenCV library that converts the frame from one color space to another, in this case from BGR to grayscale, as gray videos are faster to process motion detection
	gray = cv2.GaussianBlur(gray, (21, 21), 0)											# apply a Gaussian blur to the frame to remove high-frequency denoise and allow us to focus on the structural objects in the frame, cv2.GaussianBlur() is a function from the OpenCV library that applies a Gaussian blur to the frame, 
																						# the second argument is the kernel size, and the third argument is the standard deviation in the X direction, which is set to 0. basically, it aims to eradicate extra details, in order to only focus on larger frame differences

	# if the first frame is None, initialize it
	if firstFrame is None:																# The first frame is used as a reference for background subtraction. The first frame is the frame that will be compared to all other frames to detect motion.
		firstFrame = gray																# The first frame is the first frame of the video stream, which is converted to grayscale and blurred to remove denoise. This frame will be used as a reference to detect motion.	
		continue																		# skip to next iteration, to ensure that the first frame is set before any motion detection is performed.

	# compute the absolute difference between the current frame and first frame
	frameDelta = cv2.absdiff(firstFrame, gray)											# compute the absolute difference between the first frame and the current frame, cv2.absdiff() is a function from the OpenCV library that computes the absolute difference between two frames, which will help us detect motion in the video stream
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]					# apply a threshold to the frame delta, cv2.threshold() is a function from the OpenCV library that applies a threshold to the frame delta, the threshold value is 25, and the maximum value is 255, cv2.THRESH_BINARY is the thresholding method used, which converts the frame delta to a binary image
																						# frameDelta is the ifference between the first frame and the current frame. 25 being The threshold value. Pixels with a value greater than 25 are set to 255 (white), and those with a value less than or equal to 25 are set to 0 (black).
																						# 255: The maximum value to use with the THRESH_BINARY thresholding type. cv2.THRESH_BINARY converts the image to a binary image
																						#basically aims to creating a binary image where white pixels represent areas of significant change (motion), making it easier to identify and process these areas.

	# dilate the thresholded image to fill in holes, then find contours on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)										# thresh is the input image to be dilated, None is the 3x3 rect kernel used for dilation as default, and iterations=2 is the number of times the dilation is applied, which helps fill in the gaps in the detected motion
																						# Dilation helps fill in small holes and gaps in the thresholded image, making the contours more solid and easier to detect.
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)	# first copies the thresh image in order to compare with original. cv2.RETR_EXTERNAL is the contour retrieval mode, which retrieves only the external contours, 
																						# cv2.CHAIN_APPROX_SIMPLE is the contour approximation method, which compresses horizontal, vertical, and diagonal segments and leaves only their end points.
																						# Finding contours helps identify the boundaries of objects in the image, which is essential for detecting motion.
	cnts = imutils.grab_contours(cnts)													# grab_contours is a function from the imutils library that extracts the contours from the tuple returned by cv2.findContours(), which is necessary for compatibility with different OpenCV versions.
																						# A utility function from the imutils library that simplifies the process of grabbing contours. cnts: The result of cv2.findContours, which may vary depending on the OpenCV version.

	# loop over the contours
	for c in cnts:
		# if the contour is too small (than the specified area), ignore it
		if cv2.contourArea(c) < args["min_area"]:										# cv2.contourArea() is a function from the OpenCV library that computes the area of the contour
			continue

		# computes the bounding box for the contour 'c', draw it on the frame, and update the text
		(x, y, w, h) = cv2.boundingRect(c)												# cv2.boundingRect() is a function from the OpenCV library that computes & returns 4 params, the bounding box vertices for the contour, which is a rectangle that encloses the contour, provides a simple way to represent the area of the contour, which can be used to draw a rectangle around the detected motion.
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)					# cv2.rectangle() is a function from the OpenCV library that draws a rectangle on the frame, the first argument is the frame, the second argument is the top-left vertex of the rectangle, the third argument is the bottom-right vertex of the rectangle, the fourth argument is the color of the rectangle, and the fifth argument is the thickness of the rectangle
		text = "ActionON"
		
        # draw the text and timestamp on the frame
		cv2.putText(frame, "Action: {}".format(text), (10, 20),							# cv2.putText() is a function from the OpenCV library that draws text on the frame, the first argument is the frame, the second argument is the text to be displayed, the third argument is the position of the text, 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)									# the fourth argument is the font face, the fifth argument is the font size, the sixth argument is the color of the text, and the seventh argument is the thickness of the text
		cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),	# the first argument is the frame, the second argument is the he current date and time formatted as a string. The format includes the full weekday name, day of the month, full month name, year, hour, minute, second, and AM/PM., 
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)		# the third argument is the position of the text (means 10 pixels from the left and 10 pixels from the bottom.), the fourth argument is the font face, the fifth argument is the font size, the sixth argument is the color of the text, and the seventh argument is the thickness of the text

		# if the user presses a key, display the process frames in seperate windows, cv2.imshow is a function that displays the image/video feed in a new window
		cv2.imshow("Security Feed", frame)												# the real security/ original video feed			
		cv2.imshow("Thresh", thresh)													# the thresholded image, which is a binary image that highlights areas of significant change (motion)
		cv2.imshow("Frame Delta", frameDelta)											# the difference between the first frame and the current frame, which helps detect motion
		key = cv2.waitKey(1) & 0xFF														# This line waits for a key press for a short period (1 millisecond). cv2.waitKey(1): A function from the OpenCV library that waits for a key event for a specified amount of time (1 millisecond in this case).
																						# This bitwise operation ensures compatibility across different platforms by masking the higher bits of the key code.

		# if the `q` key is pressed, break from the loop
		if key == ord("q"):																# exit the process
			break

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()							# vs.stop() stops video stream if its from a webcam, otherwise vs.release() releases the video file if its from a video file (release the camera sources), 
cv2.destroyAllWindows()																	# cv2.destroyAllWindows() closes all the windows that were opened by OpenCV