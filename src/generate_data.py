import os

import cv2

SIZE = 10
FOLDER = 1
DATA_LENGHT = 1000

# get number of images in the specified folder

VARIANT = int(len(os.listdir(f"./img/{FOLDER}")))


vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, (16 * SIZE))
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, (9 * SIZE))

print(f"Window : ({16*SIZE}x{9*SIZE})")

# Get frames of video and save it in the spesified folder
for i in range(DATA_LENGHT):
    _ret, frame = vid.read()

    filename = f"./img/{FOLDER}/img_{i+VARIANT}.jpg"

    cv2.imwrite(filename, frame)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
