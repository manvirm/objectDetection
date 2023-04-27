import numpy as np
import cv2

image_path = 'roompeople.jpg'
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_math = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

# Generate colors list
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_math)

image = cv2.imread(image_path)
height, weight = image.shape[0], image.shape[1]

# Create Binary Large Object (BLOB)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)