import numpy as np
import cv2

image_path = 'roompeople.jpg'
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_math = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

