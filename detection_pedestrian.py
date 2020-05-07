from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
from sys import platform
import flirimageextractor

video_src = 'sample.mp4'

capture = cv2.VideoCapture(video_src)

people_cascade = cv2.CascadeClassifier('pedestrian.xml')

try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON`' \
            'in CMake and have this Python script in the right folder?')
        sys.exit(-1)

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_in", default="./input_image.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--image_out", default="./output_image.jpg", help="Image output")
    parser.add_argument("--reference_px", default="200", help="X reference T position")
    parser.add_argument("--reference_py", default="400", help="Y reference T position")
    parser.add_argument("--head_point", default="2", help="Openpose head point (around it will be created a roi)")
    parser.add_argument("--roi_sizex", default="8", help="Roi size on X")
    parser.add_argument("--roi_sizey", default="8", help="Roi size on Y")
    parser.add_argument("--reference_temperature", default="25.5", help="Reference temperature")
    parser.add_argument("--limit_temperature", default="37.5", help="Limit temperature")
    parser.add_argument("--radiometric", default="False", help="User radiometric temperature, else reference temperature is used")

    parser.add_argument("--openpose_folder", default="/ai_thermometer/openpose/models/",
                        help="Path to the local OpenPose installation directory")

    args = parser.parse_known_args()

    params = dict()

    params["model_folder"] = args[0].openpose_folder
    params["face"] = False
    params["hand"] = False
    params["net_resolution"] = "512x384"

    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    radiometric = True if args[0].radiometric == "True" else False

    if radiometric:
        try:
            flir = flirimageextractor.FlirImageExtractor()
            flir.process_image(args[0].image_in) 
            thermal = flir.get_thermal_np()
        except:
            print("Input image is not radiometric!")
            os._exit(-1)

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()

    imageToProcess = cv2.imread(args[0].image_in)

    if radiometric:
        gray = cv2.cvtColor(imageToProcess, cv2.COLOR_BGR2GRAY)
        
        gray_inverted = cv2.bitwise_not(gray)

        imageToProcess = cv2.cvtColor(gray_inverted, cv2.COLOR_GRAY2BGR)

    datum.cvInputData = imageToProcess 

    opWrapper.emplaceAndPop([datum])

    bodys = np.array(datum.poseKeypoints).tolist()

    imageToShow = datum.cvOutputData

    if type(bodys) is list:
        for body in bodys:
            face = [[int(body[0][0]),int(body[0][1])],
                [int(body[15][0]),int(body[15][1])],
                [int(body[16][0]),int(body[16][1])]]

            if 0 not in face[0] and 0 not in face[1] and 0 not in face[2]:
                
                size_x = int(args[0].roi_sizex)
                size_y = int(args[0].roi_sizey)

                reference_x = face[int(args[0].head_point)][0]
                reference_y = face[int(args[0].head_point)][1]

                reference_px = int(args[0].reference_px)
                reference_py = int(args[0].reference_py)

                offset_x = 0
                offset_y = 0

                counter = 0
                average = 0

                for y in range(reference_x-size_x+offset_x, reference_x+size_x+offset_x):
                    for x in range(reference_y-size_y+offset_y, reference_y+size_y+offset_y):
                        if radiometric:
                            average += thermal[x, y]
                        else:    
                            average += imageToProcess[x, y][0]

                        counter += 1
                
                if counter!=0:
                    average = average / counter
                
                if counter!=0:
                    if radiometric:
                        temperature = average
                        print("Face rect temperature: T:{0:.2f}C".format(temperature))
                    else:
                        reference_temperature = imageToProcess[reference_px, reference_py][0];

                        temperature = (average * float(args[0].reference_temperature))/reference_temperature

                        print("Face rect temperature: T:{0:.2f}C, {1:.2f}".format(temperature, average))
                        print("Reference floor pixel: {0}".format(str(reference_temperature)))

                        cv2.circle(imageToShow, (reference_px, reference_py), 5, (0, 250, 0), -1)
   
                    text_str = 'T:{0:.2f}C'.format(temperature)

                    x1 = reference_x
                    y1 = reference_y

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.9
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                    if temperature > float(args[0].limit_temperature):
                        color = (0, 0, 255)
                    else:
                        color = (255, 0, 0)

                    cv2.rectangle(imageToShow, (reference_x-size_x+offset_x, reference_y-size_y+offset_y), 
                        (reference_x+size_x+offset_x, reference_y+size_y+offset_y), (0, 250, 0), 2)

                    cv2.rectangle(imageToShow, (x1-16, y1-20), ((x1-16) + text_w, (y1-20) - text_h - 4), color, -1)

                    cv2.putText(imageToShow, text_str, (x1-16, y1-20), font_face,
                        font_scale, (255,255,255), font_thickness, cv2.LINE_AA)

    cv2.imwrite(args[0].image_out, imageToShow)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

writer = None

W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0

fps = FPS().start()

while True:
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame
	if args["input"] is not None and frame is None:
		break

	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	status = "Waiting"
	rects = []

	if totalFrames % args["skip_frames"] == 0:
		status = "Detecting"
		trackers = []

		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])

				if CLASSES[idx] != "person":
					continue
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				trackers.append(tracker)

	else:
		# loop over the trackers
		for tracker in trackers:
			status = "Tracking"

			tracker.update(rgb)
			pos = tracker.get_position()

			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			rects.append((startX, startY, endX, endY))

	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

	objects = ct.update(rects)

	for (objectID, centroid) in objects.items():
		to = trackableObjects.get(objectID, None)
		if to is None:
			to = TrackableObject(objectID, centroid)
		else:
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)
			if not to.counted:
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					to.counted = True
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.counted = True
		trackableObjects[objectID] = to

		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
	info = [
		("Up", totalUp),
		("Down", totalDown),
		("Status", status),
	]
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	if writer is not None:
		writer.write(frame)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	totalFrames += 1
	fps.update()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
if writer is not None:
	writer.release()
if not args.get("input", False):
	vs.stop()
else:
	vs.release()


while True:
    rectangle, image = capture.read()
	
    
    if (type(image) == type(None)):
        break
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pedastrian = people_cascade.detectMultiScale(gray,1.3,2)

    for(a,b,c,d) in pedastrian:
        cv2.rectangle(image,(a,b),(a+c,b+d),(0,255,210),4)
    
    cv2.imshow('video', image)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()

