import numpy as np
import argparse
import imutils
import pickle
import cv2 as cv
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
cmdArgs = vars(ap.parse_args())
# cmdArgs ={
# 	"image":"images/lal.jpg"
# }
args = {
	"detector":"model",
	"embedding_model":"nn4.v2.t7",
	"recognizer":"output/recognizer.pickle",
	"le":"output/le.pickle",
	"confidence":0.5,
	"predictionConfidence":0.5,
}


print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv.dnn.readNetFromCaffe(protoPath, modelPath)


print("[INFO] loading face recognizer...")
embedder = cv.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# load the image from the input folders
image = cv.imread(cmdArgs["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# construct a blob from the image
imageBlob = cv.dnn.blobFromImage(
	cv.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

# apply OpenCV's deep learning-based face detector to localize faces in the input image
detector.setInput(imageBlob)
detections = detector.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the face
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# extract the face ROI
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		# ensure the face width and height are sufficiently large
		if fW < 20 or fH < 20:
			continue

		faceBlob = cv.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# perform classification to recognize the face
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name =''
		if proba >= args["predictionConfidence"]:
			name = le.classes_[j]
		else:
			name = 'unknown'
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv.rectangle(image, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		cv.putText(image, text, (startX, y),
			cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

# show the output 
cv.imshow("Image", image)
cv.waitKey(0)