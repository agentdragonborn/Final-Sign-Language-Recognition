import cv2
import numpy as np
import math
import pickle
import tensorflow as tf
from cnn_tf import cnn_model_fn
import os
import sqlite3, pyttsx3
from threading import Thread


image_x, image_y = 50, 50

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def init_create_folder_database():
	# create the folder and database if not exist
	if not os.path.exists("gestures"):
		os.mkdir("gestures")
	if not os.path.exists("gesture_db.db"):
		conn = sqlite3.connect("gesture_db.db")
		create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
		conn.execute(create_table_cmd)
		conn.commit()

def create_folder(folder_name):
	if not os.path.exists(folder_name):
		os.mkdir(folder_name)

def create_empty_images(folder_name, n_images):
	create_folder("gestures/"+folder_name)
	black = np.zeros(shape=(image_x, image_y, 1), dtype=np.uint8)
	for i in range(n_images):
		cv2.imwrite("gestures/"+folder_name+"/"+str(i+1)+".jpg", black)

def store_in_db(g_id, g_name):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "INSERT INTO gesture (g_id, g_name) VALUES (%s, \'%s\')" % (g_id, g_name)
	try:
		conn.execute(cmd)
	except sqlite3.IntegrityError:
		choice = input("g_id already exists. Want to change the record? (y/n): ")
		if choice.lower() == 'y':
			cmd = "UPDATE gesture SET g_name = \'%s\' WHERE g_id = %s" % (g_name, g_id)
			conn.execute(cmd)
		else:
			print("Doing nothing...")
			return
	conn.commit()
    
    
    
    
def get_img_contour_thresh(img):
	img = cv2.flip(img, 1)

	#cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
	cv2.rectangle(img, (300,100),(600,400),(0,255,0),0)
	crop_img = img[100:400,300:600]

    # convert to grayscale
	grey=cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)

    # applying gaussian blur
	value = (11, 11)
	blurred = cv2.GaussianBlur(grey,value,0)

    # thresholdin: Otsu's Binarization method
	_, thresh1 = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	cv2.imshow('Thresholded', thresh1)

	contours= cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
	cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # create bounding rectangle around the contour (can skip below two lines)
	x, y, w, h = cv2.boundingRect(cnt)
	cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # finding convex hull
	hull = cv2.convexHull(cnt)

    # drawing contours
	drawing = np.zeros(crop_img.shape,np.uint8)
	cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
	cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    # finding convex hull
	hull = cv2.convexHull(cnt, returnPoints=False)

    # finding convexity defects
	defects = cv2.convexityDefects(cnt, hull)
	count_defects = 0
	cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # applying Cosine Rule to find angle for all defects (between fingers)
    # with angle > 90 degrees and ignore defects
	for i in range(defects.shape[0]):
		s,e,f,d = defects[i,0]

		start = tuple(cnt[s][0])
		end = tuple(cnt[e][0])
		far = tuple(cnt[f][0])

        # find length of all sides of triangle
		a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
		b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
		c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # apply cosine rule here
		angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignore angles > 90 and highlight rest with red dots
		if angle <= 90:
		    count_defects += 1
		    cv2.circle(crop_img, far, 1, [0,0,255], -1)
        #dist = cv2.pointPolygonTest(cnt,far,True)

        # draw a line from start to end i.e. the convex points (finger tips)
        # (can skip this part)
		cv2.line(crop_img,start, end, [0,255,0], 2)
		cv2.circle(crop_img,far,5,[0,0,255],-1)
	thresh1 = thresh1[y:y+h, x:x+w]

	return img, contours, thresh1



def store_images(g_id):
	total_pics = 1200
	if g_id == str(0):
		create_empty_images("0", total_pics)
		return
	hist = get_hand_hist()
	cam = cv2.VideoCapture(1)
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	x, y, w, h = 300, 100, 300, 300

	create_folder("gestures/"+str(g_id))
	pic_no = 0
	flag_start_capturing = False
	frames = 0
	
	while True:
		img = cam.read()[1]
		img, contours, thresh = get_img_contour_thresh(img)

		if len(contours) > 0:
			contour = max(contours, key = lambda x: cv2.contourArea(x))
			if cv2.contourArea(contour) > 10000 and frames > 50:
				x1, y1, w1, h1 = cv2.boundingRect(contour)
				pic_no += 1
				save_img = thresh
				if w1 > h1:
					save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
				elif h1 > w1:
					save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
				save_img = cv2.resize(save_img, (image_x, image_y))
				cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
				cv2.imwrite("gestures/"+str(g_id)+"/"+str(pic_no)+".jpg", save_img)

		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
		cv2.imshow("Capturing gesture", img)
		cv2.imshow("thresh", thresh)
		keypress = cv2.waitKey(1)
		if keypress == ord('c'):
			if flag_start_capturing == False:
				flag_start_capturing = True
			else:
				flag_start_capturing = False
				frames = 0
		if flag_start_capturing == True:
			frames += 1
		if pic_no == total_pics:
			break

init_create_folder_database()
g_id = input("Enter gesture no.: ")
g_name = input("Enter gesture name/text: ")
store_in_db(g_id, g_name)
store_images(g_id)