import socket
import cv2
import numpy as np
import time
from custom_socket import CustomSocket
import json

idx = input('Cam index: ')

host = socket.gethostname()
port = 10002

c = CustomSocket(host, port)
c.clientConnect()

cap = cv2.VideoCapture(int(idx))
# cap.set(4,720)
# cap.set(3,1280)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    cv2.imshow('test', frame)

    # print("Send")
    msg = c.req(frame)
    print(msg)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
