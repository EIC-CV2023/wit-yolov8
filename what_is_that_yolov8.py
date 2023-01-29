import argparse
import json
import os
import socket
from pickle import NONE
from custom_socket import CustomSocket
import numpy as np
from pathlib import Path

import mediapipe as mp
from pose_estimation_module.hand_tracking import HandTracking
from pose_estimation_module.pose_estimation import PoseEstimation
import time
from ultralytics.SORT import *
import cv2
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import select_device
import yaml

mp_hands = mp.solutions.hands

WEIGHT = "cokebest-fs-seg.pt"
# DATASET_NAME = "coco"
DATASET_NAME = {0: "coke"}


class V8Tracker:
    def __init__(self, weight="yolov8s-seg.pt", conf=0.5, dataset_name="coco", sort_max_age=10, sort_min_hits=5, sort_iou_thresh=0.2, show_result=False):
        self.tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits,
                            iou_threshold=sort_iou_thresh)
        self.model = YOLO(weight)
        self.rand_color_list = np.random.rand(20, 3) * 255
        self.conf = conf
        self.show_result = show_result

        if dataset_name == "coco":
            with open("ultralytics/yolo/data/datasets/coco8-seg.yaml", "r") as stream:
                try:
                    datasets = yaml.safe_load(stream)
                    self.datasets_names = datasets['names']
                except:
                    print("No file found")
                    self.datasets_names = ""
        else:
            # In format of {0: name0, 1: name1, ...}
            self.datasets_names = dataset_name

    def draw_box(self, img, bbox, id=None, label=None):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2),
                      self.rand_color_list[id % 20], 3)
        cv2.putText(img, f"{id}:{label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    1, self.rand_color_list[id % 20], 2)
        return img

    def track(self, frame):
        self.res = []
        self.frame = np.copy(frame)
        self.results = self.model.predict(
            source=frame, conf=self.conf, show=self.show_result)[0]
        if self.results.boxes:
            # print(f"DETECT {len(results.boxes)}")
            output = dict()
            dets_to_sort = np.empty((0, 6))

            for i, obj in enumerate(self.results.boxes):
                x1, y1, x2, y2, conf, cls = obj.data.cpu().detach().numpy()[0]
                name = self.datasets_names[int(
                    cls)] if self.datasets_names else 'unknown'

                output[i] = [name, x1, y1, x2, y2]

                dets_to_sort = np.vstack((dets_to_sort,
                                          np.array([x1, y1, x2, y2, conf, cls])))
            # print(dets_to_sort)

            tracked_dets = self.tracker.update(dets_to_sort)
            # print(tracked_dets)
            for tk in tracked_dets:
                x1, y1, x2, y2 = (int(p) for p in tk[:4])
                w, h = x2-x1, y2-y1
                id = int(tk[8])
                cls = tk[4]
                name = self.datasets_names[cls]
                self.draw_box(self.frame, (x1, y1, x2, y2), id, name)
                self.res.append([id, cls, name, x1, y1, w, h])

        return self.res, self.frame


class WhatIsThat:

    def __init__(self):
        self.HT = HandTracking()
        self.PE = PoseEstimation()
        self.start = time.time()

    def what_is_that(self, img, formatted_bbox):

        image = img.copy()
        image = cv2.flip(image, 1)
        image.flags.writeable = False

        # hands detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hands_results = self.HT.track(image)
        pose_results = self.PE.track(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # bbox_list = self.OD.get_bbox(image)
        # formatted_bbox = self.OD.format_bbox(bbox_list)

        image.flags.writeable = True

        self.HT.read_results(image, hands_results)
        self.PE.read_results(image, pose_results)

        # finger_list = [(startindex, midindex, length), ...]
        finger_list = [(7, 8, 200)]
        joint_list = [(13, 15, 200), (14, 16, 200)]

        # define solution list
        obj_list = []

        # check if there is a hand
        if self.HT.hands_results.multi_hand_landmarks:
            self.HT.draw_hand()
            self.HT.draw_hand_label()
            obj_list = self.HT.point_to(formatted_bbox, finger_list)

        elif self.PE.pose_results.pose_landmarks:
            print("Can't detect hands, detecting pose")
            # self.PE.draw_pose()
            obj_list = self.PE.pose_point_to(formatted_bbox, joint_list)

        self.HT.draw_boxes(formatted_bbox)

        # print(obj_list)
        #
        # # get fps
        fps = 1 / (time.time() - self.start)
        self.start = time.time()
        cv2.putText(image, "fps: " + str(round(fps, 2)), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        # cv2.imshow('result image', image)
        # cv2.waitKey(1)

        return obj_list, image


def main():
    HOST = "0.0.0.0"
    # HOST = "192.168.8.99"
    PORT = 10002

    server = CustomSocket(HOST, PORT)
    server.startServer()

    # OT = ObjectTracker()
    WIT = WhatIsThat()
    V8T = V8Tracker(weight=WEIGHT, dataset_name=DATASET_NAME)

    while True:
        conn, addr = server.sock.accept()
        print("Client connected from", addr)
        while True:
            try:
                data = server.recvMsg(conn)
                img = np.frombuffer(data, dtype=np.uint8).reshape(480, 640, 3)
                # img = np.frombuffer(data, dtype=np.uint8).reshape(720, 1280, 3)

                sol, drawn_frame = V8T.track(img)

                out = {}
                obj = []
                formatted_bbox = []

                for s in sol:
                    id, cls, classname, x, y, w, h = s
                    obj.append([id, cls, classname, x, y, w, h])
                    formatted_bbox.append([classname, (x, y, w, h), False])
                out["result"] = obj
                out["n"] = len(obj)
                # server.sendMsg(conn,json.dumps(out, indent = 4))

                results, frame = WIT.what_is_that(
                    cv2.flip(img, 1), formatted_bbox)
                cv2.imshow("Result image", frame)
                cv2.waitKey(1)
                what_is_that = []
                for result in results:
                    what_is_that.append((result))
                res = {"what_is_that": what_is_that}
                print(res)
                server.sendMsg(conn, json.dumps(res))

            except Exception as e:
                print(e)
                print("CONNECTION CLOSED")
                break


if __name__ == '__main__':
    main()
