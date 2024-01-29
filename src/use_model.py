import os

import cv2
import joblib
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Setup variables
MODEL_DIR=os.path.join("./models",os.listdir('./models')[-1])
INPUT_DIR = "./img"
CATEGORIES = ["1", "2", "3", "4", "5", "6"]
CHAR = [
    imread("./targets/1.png"),
    imread("./targets/2.png"),
    imread("./targets/3.png"),
    imread("./targets/4.png"),
    imread("./targets/5.png"),
    imread("./targets/6.png"),
]


video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 100)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)

SVC_LOADED_MODEL = joblib.load(MODEL_DIR)

def image_show(val: int):
    cv2.imshow("frame", CHAR[val])

while True:
    ret, frame = video.read()
    img = resize(frame, (20, 20))
    predict = SVC_LOADED_MODEL.predict([img.flatten()])[0]
    print(predict)

    image_show(predict)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
