# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from math import *
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)

"""
	res contains : 
		name :
			points : list of the points of the named part
			window : rectangle that contain this part
"""
res = {}


for (i, rect) in enumerate(rects):
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		res[name] = {}
		res[name]["points"] = []
		for (x, y) in shape[i:j]:
			res[name]["points"].append((x,y))
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		res[name]["window"] = (x,y,w,h)

"""
	for every part, get the points at left, right top and bottom
"""
for k in res :
	left = res[k]["points"][0]
	right = res[k]["points"][0]
	top = res[k]["points"][0]
	bottom = res[k]["points"][0]

	for i in (res[k]["points"]) :
		if i[0] < left[0] :
			left = i
		if i[0] > right[0] :
			right = i
		if i[1] < bottom[1] :
			bottom = i
		if i[1] > top[1] :
			top = i
	res[k]["limitations"] = [left,top,right,bottom]
	res[k]["center"] = (int((left[0] + right[0] + top[0] + bottom[0])/4),int((left[1] + right[1] + top[1] + bottom[1]) /4))


"""
	draw the delimitation points
"""
clone = image.copy()
for k in res :
	cv2.putText(clone, k, res[k]["center"], cv2.FONT_HERSHEY_SIMPLEX,0.4, (255, 0, 0), 1)
	for i in res[k]["limitations"] :
		cv2.circle(clone, i, 1, (0, 0, 255), -1)
	cv2.circle(clone, res[k]["center"], 1, (255, 0, 0), -1)




"""
	do the same thing for the drawing
"""
dessin = cv2.imread("photo1.png")
dessin = imutils.resize(dessin, width=500)
res_drawing = {
	"right_eye" : {
		"limitations" : [(181,184),(195,150),(203,184),(193,220)],
		"center" : (192,185)
	}
}

for k in res_drawing :
	cv2.putText(dessin, k, res_drawing[k]["limitations"][3], cv2.FONT_HERSHEY_SIMPLEX,0.4, (255, 0, 0), 1)
	for i in res_drawing[k]["limitations"] :
		cv2.circle(dessin, i, 1, (0, 0, 255), -1)
	cv2.circle(dessin, res_drawing[k]["center"], 1, (255, 0, 0), -1)


cv2.imshow("Image", clone)
cv2.imshow("Dessin", dessin)
cv2.waitKey(0)

"""
	compute the correct position  for the features to move
"""

res_move = {}
for k in res_drawing :
	res_move[k] = {}
	res_move[k]["limitations"] = []
	for i in range(len(res[k]["limitations"])) :
		res_move[k]["limitations"].append((res_drawing[k]["limitations"][i][0]+res[k]["center"][0] - res_drawing[k]["center"][0],res_drawing[k]["limitations"][i][1]+res[k]["center"][1] - res_drawing[k]["center"][1]))

new_image = image.copy()
for k in res_move :
	for i in res_move[k]["limitations"] :
		cv2.circle(new_image, i, 1, (0, 0, 255), -1)



cv2.imshow("Dessin2", new_image)
cv2.waitKey(0)

# test déformation
for k in res_move :
	ratio_vert = -(res_drawing[k]["limitations"][3][1] - res_drawing[k]["limitations"][1][1]) / (res[k]["limitations"][3][1] - res[k]["limitations"][1][1])
	ratio_honz = (res_drawing[k]["limitations"][2][0] - res_drawing[k]["limitations"][0][0]) / (res[k]["limitations"][2][0] - res[k]["limitations"][0][0])

	height, width, _ = image.shape
	map_y = np.zeros((height,width),dtype=np.float32)
	map_x = np.zeros((height,width),dtype=np.float32)

	right_eye = res[k]["center"]
	# create index map
	for i in range(height):
		for j in range(width):
			map_y[i][j]=i
			map_x[i][j]=j
	a = res_drawing[k]["limitations"][2][0] - res_drawing[k]["limitations"][0][0]
	b = res_drawing[k]["limitations"][3][1] - res_drawing[k]["limitations"][1][1]
	power = 1.3
	print(a,b)
	radius = 30
	sigma = 1
	for i in range (-right_eye[1], height - right_eye[1]):
		for j in range(-right_eye[0], width - right_eye[0]):
			if ((i/b)**2 + (j/a)**2>1) :
				continue
			if i > 0:
				map_y[right_eye[1] + i][right_eye[0] + j] = right_eye[1] + (i/b)**power * b 
			if i < 0:
				map_y[right_eye[1] + i][right_eye[0] + j] = right_eye[1] - (-i/b)**power * b 
			if j > 0:
				map_x[right_eye[1] + i][right_eye[0] + j] = right_eye[0] + (j/a)**power * a 
			if j < 0:
				map_x[right_eye[1] + i][right_eye[0] + j] = right_eye[0] - (-j/a)**power * a 
	warped=cv2.remap(image,map_x,map_y,cv2.INTER_LINEAR)
	cv2.imshow("fezfze", warped)
	cv2.waitKey(0)
# run python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image photo.png

